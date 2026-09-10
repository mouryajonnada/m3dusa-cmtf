"""
Heterogeneous Graph Attention Encoder (HGTConv)
===============================================
Encodes the partial M3DUSA ``HeteroData`` graphs (News + Hashtag node types)
using stacked **Heterogeneous Graph Transformer** (HGT) convolution layers
(Hu et al., 2020).

Architecture
------------
1. Per-node-type linear **input projection**: 384 → ``hidden_dim``
2. ``num_layers`` × ``HGTConv`` layers (with optional residual connections)
3. Per-graph **pooling** of News and Hashtag node embeddings:
   - ``news_pooled``    ``(B, H)`` — one News node per graph (direct index)
   - ``hashtag_pooled`` ``(B, H)`` — global mean-pool over Hashtag nodes

The ``forward`` method returns a dict with multiple views of the graph
representation so that both ``LateFusion`` and ``CrossModalFusion`` can
consume what they need:

.. code-block:: python

    {
        "pooled":   (B, H)    # news-node repr — for LateFusion
        "sequence": (B, 2, H) # [news, hashtag] stack — for CrossModalFusion
        "news":     (B, H)    # raw news-node embeddings
        "hashtag":  (B, H)    # mean-pooled hashtag embeddings
    }

Usage::

    from src.data.graph_builder import GraphBuilder
    from src.models.gat_encoder import GATEncoder
    from src.utils.config import load_config

    cfg     = load_config("experiments/baseline_m3dusa.yaml")
    builder = GraphBuilder()
    meta    = builder.metadata()

    encoder = GATEncoder(cfg, metadata=meta)
    out     = encoder(graph_batch)      # graph_batch from DataLoader
    print(out["pooled"].shape)          # (B, 256)
    print(out["sequence"].shape)        # (B, 2, 256)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, global_mean_pool

from src.utils.config import DotDict
from src.utils.logging import get_logger

log = get_logger(__name__)

# Sentence-embedding dimension produced by all-MiniLM-L6-v2
_NODE_FEAT_DIM: int = 384


class GATEncoder(nn.Module):
    """Heterogeneous Graph Transformer encoder for News+Hashtag graphs.

    Args:
        cfg:      Loaded config ``DotDict``.  Reads from ``cfg.model.gat``:

                  - ``hidden_dim``   — output feature dimension per node
                  - ``num_layers``   — number of ``HGTConv`` layers
                  - ``num_heads``    — attention heads per layer
                  - ``dropout``      — dropout probability
                  - ``residual``     — add skip-connection per layer

        metadata: Graph metadata ``(node_types, edge_types)`` from
                  :meth:`src.data.graph_builder.GraphBuilder.metadata`.
        in_channels: Input node feature dimension.  Defaults to 384
                     (``all-MiniLM-L6-v2`` embedding size).
    """

    def __init__(
        self,
        cfg:         DotDict,
        metadata:    tuple[list[str], list[tuple[str, str, str]]],
        in_channels: int = _NODE_FEAT_DIM,
    ) -> None:
        super().__init__()

        gat_cfg            = cfg.model.gat
        self.hidden_dim    = gat_cfg.hidden_dim
        self.num_layers    = gat_cfg.num_layers
        self.residual      = gat_cfg.residual
        self.dropout_p     = gat_cfg.dropout

        node_types, edge_types = metadata
        self.node_types    = node_types
        self.edge_types    = edge_types

        # ── 1. Input projection (one per node type) ────────────────────
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels, self.hidden_dim)
            for nt in node_types
        })

        # ── 2. Stacked HGTConv layers ──────────────────────────────────
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                metadata=metadata,
                heads=gat_cfg.num_heads,
            )
            for _ in range(self.num_layers)
        ])

        # ── 3. Layer norms (one per node type per layer) ───────────────
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(self.hidden_dim) for nt in node_types})
            for _ in range(self.num_layers)
        ])

        self.drop = nn.Dropout(p=self.dropout_p)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            f"GATEncoder: {self.num_layers} HGTConv layers  "
            f"hidden_dim={self.hidden_dim}  heads={gat_cfg.num_heads}  "
            f"residual={self.residual}  "
            f"params={n_params:,}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_edge_index_dict(self, graph_batch) -> dict:
        """Build edge_index_dict, filling missing edge types with empty tensors."""
        device = next(iter(self.input_proj.values())).weight.device
        edge_index_dict = {}
        for et in self.edge_types:
            try:
                ei = graph_batch[et].edge_index
            except (KeyError, AttributeError):
                ei = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_index_dict[et] = ei
        return edge_index_dict

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, graph_batch) -> dict[str, torch.Tensor]:
        """Encode a batched ``HeteroData`` graph.

        Args:
            graph_batch: Batched ``torch_geometric.data.Batch`` built from a
                         list of ``HeteroData`` objects (output of
                         :func:`~src.data.dataset.collate_fn`).

        Returns:
            Dict with keys:

            - ``"pooled"``    ``(B, H)`` — primary graph repr for LateFusion
              (news-node embedding, already enriched by HGT message passing)
            - ``"sequence"``  ``(B, 2, H)`` — [news, hashtag] token sequence
              for CrossModalFusion
            - ``"news"``      ``(B, H)`` — raw pooled news embeddings
            - ``"hashtag"``   ``(B, H)`` — raw pooled hashtag embeddings
        """
        B = int(graph_batch.num_graphs)

        # ── Assemble node feature dict ─────────────────────────────────
        x_dict: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            try:
                x = graph_batch[nt].x
            except (KeyError, AttributeError):
                device = next(iter(self.input_proj.values())).weight.device
                x = torch.zeros((0, _NODE_FEAT_DIM), device=device)
            x_dict[nt] = x

        # ── Input projection ───────────────────────────────────────────
        x_dict = {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}

        # ── Edge index dict ────────────────────────────────────────────
        edge_index_dict = self._safe_edge_index_dict(graph_batch)

        # ── HGTConv layers with optional residual + LayerNorm ──────────
        for layer_idx, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            new_x = conv(x_dict, edge_index_dict)

            # HGTConv may return None for node types with no edges; handle
            new_x = {
                nt: (new_x[nt] if new_x[nt] is not None else x_dict[nt])
                for nt in self.node_types
                if nt in new_x
            }

            processed: dict[str, torch.Tensor] = {}
            for nt in self.node_types:
                h = new_x.get(nt, x_dict[nt])  # fallback to input if missing
                if self.residual and h.shape == x_dict[nt].shape:
                    h = h + x_dict[nt]
                h = norm_dict[nt](h)
                h = self.drop(F.leaky_relu(h, negative_slope=0.2))
                processed[nt] = h

            x_dict = processed

        # ── Graph-level pooling ────────────────────────────────────────
        device = x_dict["news"].device

        # News: exactly 1 node per graph → batch vector is [0,1,...,B-1]
        news_x = x_dict["news"]                          # (B, H)
        news_batch = graph_batch["news"].batch           # (B,)
        news_pooled = global_mean_pool(news_x, news_batch, size=B)  # (B, H)

        # Hashtag: 0..many nodes per graph → safe mean pool
        hash_x = x_dict["hashtag"]                      # (total_H, H) or empty
        if hash_x.shape[0] > 0:
            hash_batch = graph_batch["hashtag"].batch
            hashtag_pooled = global_mean_pool(hash_x, hash_batch, size=B)  # (B, H)
        else:
            hashtag_pooled = torch.zeros(
                B, self.hidden_dim, device=device, dtype=news_pooled.dtype
            )

        # ── Sequence view for cross-modal attention ────────────────────
        # Shape: (B, 2, H)  — token 0 = news, token 1 = hashtag aggregate
        graph_sequence = torch.stack([news_pooled, hashtag_pooled], dim=1)

        return {
            "pooled":   news_pooled,      # (B, H)  primary repr
            "sequence": graph_sequence,   # (B, 2, H)
            "news":     news_pooled,      # (B, H)
            "hashtag":  hashtag_pooled,   # (B, H)
        }
