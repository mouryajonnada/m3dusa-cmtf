"""
FakeNewsDetector — Full Model
==============================
Wires together all four sub-modules into a single ``nn.Module``:

    TextEncoder  ──┐
                   ├──► Fusion ──► ClassifierHead ──► logits
    GATEncoder   ──┘

The ``forward`` method dispatches to the correct fusion path based on
``cfg.model.fusion.method``:

- ``"late_fusion"``  — pooled text + pooled graph → concat → project
- ``"cross_modal"``  — full text sequence + graph [news, hashtag] sequence
                       → bidirectional cross-attention → pool → project

Usage::

    from src.models.model import FakeNewsDetector
    from src.data.graph_builder import GraphBuilder
    from src.utils.config import load_config

    cfg      = load_config("experiments/baseline_m3dusa.yaml")
    metadata = GraphBuilder().metadata()
    model    = FakeNewsDetector(cfg, metadata)

    logits = model(batch)   # batch = dict from DataLoader collate_fn
    # logits: (B, 2)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.text_encoder import TextEncoder
from src.models.gat_encoder  import GATEncoder
from src.models.fusion       import build_fusion
from src.models.classifier   import ClassifierHead
from src.utils.config        import DotDict
from src.utils.logging       import get_logger

log = get_logger(__name__)


def _move_batch(batch: dict, device: torch.device | str) -> dict:
    """Move all tensors and PyG objects in *batch* to *device*."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif hasattr(v, "to"):           # PyG HeteroData / Batch
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class FakeNewsDetector(nn.Module):
    """End-to-end fake-news detection model.

    Args:
        cfg:      Loaded config ``DotDict``.
        metadata: Graph metadata ``(node_types, edge_types)`` from
                  :meth:`~src.data.graph_builder.GraphBuilder.metadata`.
    """

    def __init__(
        self,
        cfg:      DotDict,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
    ) -> None:
        super().__init__()
        self.cfg           = cfg
        self.fusion_method = cfg.model.fusion.method.lower()

        # ── Sub-modules ────────────────────────────────────────────────
        self.text_encoder = TextEncoder(cfg)
        self.gat_encoder  = GATEncoder(cfg, metadata)

        text_dim  = self.text_encoder.hidden_size
        graph_dim = cfg.model.gat.hidden_dim

        self.fusion     = build_fusion(cfg, text_dim=text_dim, graph_dim=graph_dim)
        self.classifier = ClassifierHead(
            cfg,
            input_dim=cfg.model.fusion.hidden_dim,
            num_classes=2,
        )

        total, trainable = self.num_parameters()
        log.info(
            f"FakeNewsDetector ready — fusion={self.fusion_method}  "
            f"total_params={total:,}  trainable={trainable:,}"
        )

    # ------------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------------

    def num_parameters(self) -> tuple[int, int]:
        """Return ``(total_params, trainable_params)``."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> torch.Tensor:
        """Run a forward pass.

        Args:
            batch: Dict produced by :func:`~src.data.dataset.collate_fn`
                   with keys ``"input_ids"``, ``"attention_mask"``,
                   ``"labels"``, ``"graph_batch"``.

        Returns:
            ``(B, 2)`` raw logits (pre-softmax).
        """
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        graph_batch    = batch["graph_batch"]

        # ── Graph encoder (shared by both fusion paths) ────────────────
        graph_out = self.gat_encoder(graph_batch)

        # ── Fusion-specific text encoding + fusion ─────────────────────
        if self.fusion_method == "late_fusion":
            text_repr = self.text_encoder(input_ids, attention_mask)
            fused     = self.fusion(
                text_repr=text_repr,
                graph_repr=graph_out["pooled"],
            )

        else:   # cross_modal
            text_seq = self.text_encoder(input_ids, attention_mask, return_all=True)
            fused    = self.fusion(
                text_seq=text_seq,
                graph_seq=graph_out["sequence"],
            )

        return self.classifier(fused)   # (B, 2)
