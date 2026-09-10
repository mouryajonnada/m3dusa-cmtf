"""
Classification Metrics
=======================
Utility functions for computing and formatting evaluation metrics for binary
fake-news classification.

Reported metrics:

- **Accuracy**
- **Macro F1** — primary metric for model selection (handles class imbalance)
- **Weighted F1**
- **Binary F1** (fake class = 1)
- **Macro Precision / Recall**
- **AUC-ROC** (requires predicted probabilities)

Usage::

    from src.utils.metrics import compute_metrics, format_metrics
    import numpy as np

    preds  = np.array([0, 1, 1, 0])
    labels = np.array([0, 1, 0, 0])
    probs  = np.array([[0.9,0.1],[0.2,0.8],[0.3,0.7],[0.8,0.2]])

    m = compute_metrics(preds, labels, probs)
    print(format_metrics(m))
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    preds:  np.ndarray,
    labels: np.ndarray,
    probs:  np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a standard suite of binary classification metrics.

    Args:
        preds:  ``(N,)`` integer predicted class labels.
        labels: ``(N,)`` integer ground-truth labels.
        probs:  ``(N, 2)`` predicted probabilities (optional).  Required for
                AUC-ROC.  If ``None``, ``"auc_roc"`` is omitted.

    Returns:
        Dict mapping metric name → float value.
    """
    metrics: dict[str, float] = {
        "accuracy":         float(accuracy_score(labels, preds)),
        "f1_macro":         float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "f1_weighted":      float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "f1_fake":          float(f1_score(labels, preds, pos_label=1,
                                           average="binary", zero_division=0)),
        "f1_real":          float(f1_score(labels, preds, pos_label=0,
                                           average="binary", zero_division=0)),
        "precision_macro":  float(precision_score(labels, preds, average="macro",
                                                   zero_division=0)),
        "recall_macro":     float(recall_score(labels, preds, average="macro",
                                               zero_division=0)),
    }

    if probs is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(labels, probs[:, 1]))
        except ValueError:
            # Raised when only one class present in labels (e.g. tiny val set)
            metrics["auc_roc"] = float("nan")

    return metrics


def format_metrics(
    metrics:  dict[str, float],
    prefix:   str = "",
    decimals: int = 4,
) -> str:
    """Return a compact one-line string of metric values.

    Args:
        metrics:  Dict from :func:`compute_metrics`.
        prefix:   Optional prefix (e.g. ``"val_"``).
        decimals: Number of decimal places.

    Returns:
        String like ``"acc=0.8523  f1_macro=0.8401  auc=0.9012"``.
    """
    fmt = f".{decimals}f"
    parts = []

    key_labels = [
        ("accuracy",        "acc"),
        ("f1_macro",        "f1_macro"),
        ("f1_fake",         "f1_fake"),
        ("f1_real",         "f1_real"),
        ("auc_roc",         "auc"),
        ("loss",            "loss"),
    ]
    for key, label in key_labels:
        if key in metrics:
            parts.append(f"{prefix}{label}={metrics[key]:{fmt}}")

    return "  ".join(parts)
