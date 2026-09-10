"""
Text Encoder
============
Wraps a HuggingFace pre-trained transformer (e.g. ``roberta-base``) for
claim-text encoding.

Supports two pooling strategies:

- ``"cls"``  — take the ``[CLS]`` token hidden state ``(B, H)``
- ``"mean"`` — average over non-padding token hidden states ``(B, H)``

Can also return the **full token sequence** ``(B, seq_len, H)`` for use with
the Cross-Modal Transformer Fusion module.

Usage::

    from src.models.text_encoder import TextEncoder
    from src.utils.config import load_config

    cfg = load_config("experiments/baseline_m3dusa.yaml")
    encoder = TextEncoder(cfg)

    # Pooled representation  (B, H)
    repr = encoder(input_ids, attention_mask)

    # Full sequence          (B, seq_len, H)
    seq  = encoder(input_ids, attention_mask, return_all=True)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from src.utils.config import DotDict
from src.utils.logging import get_logger

log = get_logger(__name__)


class TextEncoder(nn.Module):
    """HuggingFace transformer wrapper for claim-text encoding.

    Args:
        cfg: Loaded config ``DotDict``.  Reads:

            - ``cfg.model.text_encoder``        — HF model identifier
            - ``cfg.model.text_pooling``         — ``"cls"`` or ``"mean"``
            - ``cfg.model.freeze_text_encoder``  — whether to freeze weights
    """

    def __init__(self, cfg: DotDict) -> None:
        super().__init__()
        model_name    = cfg.model.text_encoder
        self.pooling  = cfg.model.text_pooling          # "cls" | "mean"
        self._frozen  = cfg.model.freeze_text_encoder

        log.info(f"Loading text encoder: {model_name}  "
                 f"(pooling={self.pooling}, frozen={self._frozen})")
        self.model = AutoModel.from_pretrained(model_name)

        if self._frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            log.info("Text encoder weights frozen.")

        # Expose hidden size so downstream modules can query it
        self.hidden_size: int = self.model.config.hidden_size
        log.info(f"Text encoder ready — hidden_size={self.hidden_size}")

    # ------------------------------------------------------------------
    # Pooling helpers
    # ------------------------------------------------------------------

    def _cls_pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the [CLS] token embedding.  ``hidden`` shape: ``(B, L, H)``."""
        return hidden[:, 0, :]          # (B, H)

    def _mean_pool(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked mean-pool over non-padding tokens.

        Args:
            hidden:         ``(B, L, H)`` — last hidden states.
            attention_mask: ``(B, L)``    — 1 for real tokens, 0 for padding.

        Returns:
            ``(B, H)`` mean-pooled representation.
        """
        mask   = attention_mask.unsqueeze(-1).float()   # (B, L, 1)
        summed = (hidden * mask).sum(dim=1)             # (B, H)
        denom  = mask.sum(dim=1).clamp(min=1e-9)       # (B, 1)
        return summed / denom                           # (B, H)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        return_all:     bool = False,
    ) -> torch.Tensor:
        """Encode a batch of tokenised claims.

        Args:
            input_ids:      ``(B, seq_len)`` token IDs.
            attention_mask: ``(B, seq_len)`` attention mask.
            return_all:     If ``True``, return the full last-hidden-state
                            sequence ``(B, seq_len, H)`` instead of the pooled
                            vector.  Used by the Cross-Modal Fusion module.

        Returns:
            - ``(B, H)``        — pooled representation (default)
            - ``(B, seq_len, H)``— full sequence if ``return_all=True``
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # (B, L, H)

        if return_all:
            return hidden                   # (B, L, H)

        if self.pooling == "cls":
            return self._cls_pool(hidden)
        else:
            return self._mean_pool(hidden, attention_mask)
