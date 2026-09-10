"""
Evaluator
=========
Runs inference on a DataLoader, accumulates logits, and returns a
:func:`~src.utils.metrics.compute_metrics` dict.

Usage::

    from src.training.evaluator import Evaluator

    evaluator = Evaluator(device="cpu")
    metrics   = evaluator.evaluate(model, val_loader)
    print(metrics["f1_macro"])
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.metrics import compute_metrics
from src.utils.logging import get_logger

log = get_logger(__name__)


def _move_batch(batch: dict, device: str | torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class Evaluator:
    """Evaluation loop wrapper.

    Args:
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def evaluate(
        self,
        model:  torch.nn.Module,
        loader: DataLoader,
    ) -> dict[str, float]:
        """Evaluate *model* on all batches in *loader*.

        The model is put in ``eval()`` mode; no gradients are computed.

        Args:
            model:  :class:`~src.models.model.FakeNewsDetector` (or any model
                    whose ``forward(batch)`` returns ``(B, num_classes)`` logits).
            loader: DataLoader to iterate over.

        Returns:
            Dict with keys from :func:`~src.utils.metrics.compute_metrics` plus
            ``"loss"`` (mean cross-entropy over the loader).
        """
        model.eval()

        all_preds:  list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_probs:  list[np.ndarray] = []
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch in loader:
                batch  = _move_batch(batch, self.device)
                labels = batch["labels"]

                logits = model(batch)                         # (B, 2)
                loss   = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                n_batches  += 1

                probs  = F.softmax(logits, dim=-1)            # (B, 2)
                preds  = logits.argmax(dim=-1)                # (B,)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        preds_arr  = np.concatenate(all_preds)
        labels_arr = np.concatenate(all_labels)
        probs_arr  = np.concatenate(all_probs)

        metrics = compute_metrics(preds_arr, labels_arr, probs_arr)
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics
