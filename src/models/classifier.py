"""
MLP Classifier Head
===================
A stacked MLP that maps a fused representation vector to class logits.

Hidden layer widths, dropout, and activation are all config-driven.

Usage::

    from src.models.classifier import ClassifierHead
    from src.utils.config import load_config

    cfg = load_config("experiments/baseline_m3dusa.yaml")
    head = ClassifierHead(cfg, input_dim=256, num_classes=2)

    logits = head(fused_repr)   # (B, 2)
    probs  = torch.softmax(logits, dim=-1)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.config import DotDict
from src.utils.logging import get_logger

log = get_logger(__name__)

_ACTIVATIONS = {
    "relu":  nn.ReLU,
    "gelu":  nn.GELU,
    "tanh":  nn.Tanh,
    "selu":  nn.SELU,
}


class ClassifierHead(nn.Module):
    """Stacked MLP classifier head.

    Builds a sequence of linear → activation → dropout layers, one per entry
    in ``cfg.model.classifier.hidden_dims``, followed by a final linear
    projection to ``num_classes`` logits.

    Args:
        cfg:         Loaded config ``DotDict``.  Reads:

                     - ``cfg.model.classifier.hidden_dims`` — list of int
                     - ``cfg.model.classifier.dropout``     — float
                     - ``cfg.model.classifier.activation``  — str
        input_dim:   Dimensionality of the fused input representation.
        num_classes: Number of output classes (default: 2 for binary).
    """

    def __init__(
        self,
        cfg:         DotDict,
        input_dim:   int,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        hidden_dims: list[int] = list(cfg.model.classifier.hidden_dims)
        dropout:     float     = cfg.model.classifier.dropout
        act_name:    str       = cfg.model.classifier.activation.lower()

        if act_name not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{act_name}'. "
                f"Choose from: {list(_ACTIVATIONS)}"
            )
        ActClass = _ACTIVATIONS[act_name]

        # Build MLP
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                ActClass(),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim

        # Final projection → logits
        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        log.info(
            f"ClassifierHead: {input_dim} -> "
            f"{' -> '.join(str(d) for d in hidden_dims)} -> {num_classes}  "
            f"(act={act_name}, dropout={dropout})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify fused representations.

        Args:
            x: ``(B, input_dim)`` fused representation tensor.

        Returns:
            ``(B, num_classes)`` raw logits (pre-softmax / pre-sigmoid).
        """
        return self.mlp(x)
