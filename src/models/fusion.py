"""
Fusion Modules
==============
Two fusion strategies for combining text and graph representations:

Late Fusion (``LateFusion``)
----------------------------
Concatenates the pooled text vector ``(B, H_text)`` with the pooled graph
vector ``(B, H_graph)`` and projects to a common ``hidden_dim``::

    fused = Linear(H_text + H_graph → hidden_dim)(concat(text, graph))

Reproduces the M3DUSA baseline fusion approach.

Cross-Modal Transformer Fusion (``CrossModalFusion``) — NOVEL
--------------------------------------------------------------
Stacks ``num_layers`` bidirectional cross-attention blocks.  Each block
applies *two* cross-attention sub-layers in sequence:

1. **Text → Graph**: text tokens attend to graph tokens
   ``MHA(Q=text_seq, K=graph_seq, V=graph_seq)``
2. **Graph → Text**: graph tokens attend to text tokens  (if ``bidirectional``)
   ``MHA(Q=graph_seq, K=text_seq, V=text_seq)``

Both sub-layers have residual connections and LayerNorm.  A position-wise FFN
follows each cross-attention sub-layer.

Inputs to ``CrossModalFusion``:

- ``text_seq``   ``(B, seq_len, H_text)`` — full RoBERTa hidden states
- ``graph_seq``  ``(B, 2, H_graph)``      — [news, hashtag] GATEncoder tokens

Both are projected to a common ``fusion_hidden_dim`` before attention.

Output: pooled ``(B, hidden_dim)`` for the classifier.

Factory
-------
Use :func:`build_fusion` to instantiate the correct module from config::

    fusion = build_fusion(cfg, text_dim=768, graph_dim=256)

Usage::

    from src.models.fusion import build_fusion
    from src.utils.config import load_config

    cfg    = load_config("experiments/cross_modal_fusion.yaml")
    fusion = build_fusion(cfg, text_dim=768, graph_dim=256)

    # Late fusion
    fused  = fusion(text_repr=text_enc_out, graph_repr=gat_out["pooled"])

    # Cross-modal fusion
    fused  = fusion(text_seq=text_enc_out, graph_seq=gat_out["sequence"])
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import DotDict
from src.utils.logging import get_logger

log = get_logger(__name__)


# ===========================================================================
# Late Fusion
# ===========================================================================

class LateFusion(nn.Module):
    """Concatenate-then-project fusion.

    Args:
        text_dim:   Dimension of the text encoder output.
        graph_dim:  Dimension of the graph encoder output.
        hidden_dim: Output dimension after projection.
        dropout:    Dropout probability applied after the projection.
    """

    def __init__(
        self,
        text_dim:   int,
        graph_dim:  int,
        hidden_dim: int,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim + graph_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        log.info(
            f"LateFusion: ({text_dim} + {graph_dim}) -> {hidden_dim}  "
            f"dropout={dropout}"
        )

    def forward(
        self,
        text_repr:  torch.Tensor,
        graph_repr: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        """Fuse pooled text and graph representations.

        Args:
            text_repr:  ``(B, H_text)``  — pooled text encoder output.
            graph_repr: ``(B, H_graph)`` — pooled graph encoder output.

        Returns:
            ``(B, hidden_dim)`` fused representation.
        """
        return self.proj(torch.cat([text_repr, graph_repr], dim=-1))


# ===========================================================================
# Cross-Modal Fusion — novel contribution
# ===========================================================================

class _FFN(nn.Module):
    """Position-wise Feed-Forward Network used inside each fusion block."""

    def __init__(self, dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CrossAttentionBlock(nn.Module):
    """One cross-attention sub-layer: MHA + residual + LN + FFN + residual + LN.

    Args:
        dim:      Common feature dimension (both sequences must be at ``dim``).
        num_heads: Number of attention heads.
        ff_dim:   Feedforward hidden dimension.
        dropout:  Dropout probability.
    """

    def __init__(
        self,
        dim:          int,
        num_heads:    int,
        ff_dim:       int,
        dropout:      float,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # (B, L, H) convention
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn   = _FFN(dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,    # (B, L_q, dim)
        key:   torch.Tensor,    # (B, L_k, dim)
        value: torch.Tensor,    # (B, L_k, dim)
    ) -> torch.Tensor:
        """Apply cross-attention from ``query`` sequence to ``key/value``."""
        attn_out, _ = self.attn(query, key, value)  # (B, L_q, dim)
        if self.use_residual:
            query = self.norm1(query + self.drop(attn_out))
            query = self.norm2(query + self.ffn(query))
        else:
            query = self.norm1(self.drop(attn_out))
            query = self.norm2(self.ffn(query))
        return query


class CrossModalFusion(nn.Module):
    """Bidirectional Cross-Modal Transformer Fusion.

    Stacks ``num_layers`` blocks, each containing:

    - **Text→Graph** cross-attention: text queries attend to graph tokens
    - **Graph→Text** cross-attention (if ``bidirectional``): graph queries
      attend to updated text tokens

    Both text and graph sequences are first projected to ``fusion_hidden_dim``.

    After all layers, text tokens are mean-pooled to ``(B, dim)`` and graph
    tokens are mean-pooled to ``(B, dim)``.  The two vectors are concatenated
    and projected to ``output_dim``.

    Args:
        text_dim:         Input text feature dimension.
        graph_dim:        Input graph feature dimension.
        fusion_hidden_dim:Common dimension inside the fusion transformer.
        output_dim:       Final output dimension (→ classifier input).
        num_layers:       Number of stacked cross-attention blocks.
        num_heads:        Attention heads per block.
        feedforward_dim:  FFN hidden dimension inside each block.
        dropout:          Dropout probability.
        bidirectional:    Whether to also apply Graph→Text cross-attention.
        use_residual:     Whether to use residual connections around attention & FFN.
    """

    def __init__(
        self,
        text_dim:          int,
        graph_dim:         int,
        fusion_hidden_dim: int,
        output_dim:        int,
        num_layers:        int   = 2,
        num_heads:         int   = 4,
        feedforward_dim:   int   = 512,
        dropout:           float = 0.1,
        bidirectional:     bool  = True,
        use_residual:      bool  = True,
    ) -> None:
        super().__init__()
        self.fusion_dim    = fusion_hidden_dim
        self.bidirectional = bidirectional
        self.use_residual  = use_residual

        # ── Input projections: text & graph → fusion_hidden_dim ───────
        self.text_proj  = nn.Linear(text_dim,  fusion_hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, fusion_hidden_dim)

        # ── Stacked cross-attention layers ─────────────────────────────
        self.text_to_graph_layers = nn.ModuleList([
            _CrossAttentionBlock(fusion_hidden_dim, num_heads, feedforward_dim, dropout, use_residual=use_residual)
            for _ in range(num_layers)
        ])
        if bidirectional:
            self.graph_to_text_layers = nn.ModuleList([
                _CrossAttentionBlock(fusion_hidden_dim, num_heads, feedforward_dim, dropout, use_residual=use_residual)
                for _ in range(num_layers)
            ])

        # Output projection: concat(text_pool, graph_pool) -> output_dim
        self.out_proj = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            f"CrossModalFusion: text({text_dim}) + graph({graph_dim}) "
            f"-> fusion_dim={fusion_hidden_dim}  layers={num_layers}  "
            f"heads={num_heads}  bidirectional={bidirectional}  "
            f"residual={use_residual}  output_dim={output_dim}  params={n_params:,}"
        )

    def forward(
        self,
        text_seq:  torch.Tensor,    # (B, seq_len, H_text)
        graph_seq: torch.Tensor,    # (B, 2, H_graph)
        **_,
    ) -> torch.Tensor:
        """Apply bidirectional cross-modal attention and pool.

        Args:
            text_seq:  Full RoBERTa token sequence ``(B, seq_len, H_text)``.
            graph_seq: GATEncoder [news, hashtag] token sequence ``(B, 2, H_graph)``.

        Returns:
            ``(B, output_dim)`` fused representation.
        """
        # Project to common fusion dimension
        T = self.text_proj(text_seq)     # (B, L_text, D)
        G = self.graph_proj(graph_seq)   # (B, 2,      D)

        # Stacked cross-modal attention
        for i, t2g in enumerate(self.text_to_graph_layers):
            # Text queries attend over Graph tokens
            T_new = t2g(query=T, key=G, value=G)

            if self.bidirectional:
                # Graph queries attend over (updated) Text tokens
                G = self.graph_to_text_layers[i](query=G, key=T_new, value=T_new)

            T = T_new

        # Pool both sequences to (B, D)
        text_pool  = T.mean(dim=1)    # (B, D)
        graph_pool = G.mean(dim=1)    # (B, D)

        # Concat and project → (B, output_dim)
        return self.out_proj(torch.cat([text_pool, graph_pool], dim=-1))


# ===========================================================================
# Factory
# ===========================================================================

def build_fusion(
    cfg:       DotDict,
    text_dim:  int,
    graph_dim: int,
) -> nn.Module:
    """Instantiate the correct fusion module from config.

    Args:
        cfg:       Loaded config ``DotDict``.  Reads ``cfg.model.fusion``.
        text_dim:  Dimension of the text encoder output ``(H_text)``.
        graph_dim: Dimension of the graph encoder output ``(H_graph)``.

    Returns:
        A :class:`LateFusion` or :class:`CrossModalFusion` module.

    Raises:
        ValueError: If ``cfg.model.fusion.method`` is not recognised.
    """
    f_cfg  = cfg.model.fusion
    method = f_cfg.method.lower()

    if method == "late_fusion":
        return LateFusion(
            text_dim=text_dim,
            graph_dim=graph_dim,
            hidden_dim=f_cfg.hidden_dim,
            dropout=f_cfg.dropout,
        )

    elif method == "cross_modal":
        use_residual = getattr(f_cfg, "use_residual", True)
        if use_residual is None:
            use_residual = True
        return CrossModalFusion(
            text_dim=text_dim,
            graph_dim=graph_dim,
            fusion_hidden_dim=f_cfg.hidden_dim,
            output_dim=f_cfg.hidden_dim,
            num_layers=f_cfg.num_layers,
            num_heads=f_cfg.num_heads,
            feedforward_dim=f_cfg.feedforward_dim,
            dropout=f_cfg.dropout,
            bidirectional=f_cfg.bidirectional,
            use_residual=bool(use_residual),
        )

    else:
        raise ValueError(
            f"Unknown fusion method: '{method}'. "
            f"Choose 'late_fusion' or 'cross_modal'."
        )
