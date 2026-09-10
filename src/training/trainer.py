"""
Trainer
=======
Manages the full training loop:

- AdamW optimiser with separate param-groups (text encoder vs heads)
- Cosine / linear LR schedule with linear warm-up (via ``transformers``)
- Gradient clipping
- Per-epoch validation with early stopping
- Best-model checkpointing
- Per-epoch metrics saved to CSV history file

Usage::

    from src.training.trainer import Trainer

    trainer = Trainer(
        model        = model,
        cfg          = cfg,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
    )
    test_metrics = trainer.train()
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.training.evaluator import Evaluator
from src.utils.config       import DotDict
from src.utils.logging      import get_logger
from src.utils.metrics      import format_metrics

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _build_param_groups(
    model:        nn.Module,
    weight_decay: float,
    text_lr:      float,
    head_lr:      float,
) -> list[dict]:
    """Split parameters into AdamW param-groups.

    - Text encoder params get ``text_lr`` (usually the smallest LR, since
      the backbone is pre-trained).
    - All other params (GAT, fusion, classifier) get ``head_lr``.
    - Bias, LayerNorm, and layer_norm weights are excluded from weight decay
      (standard HuggingFace practice).
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    text_named  = list(model.text_encoder.named_parameters())
    other_named = [
        (n, p)
        for n, p in model.named_parameters()
        if not n.startswith("text_encoder.") and p.requires_grad
    ]

    groups: list[dict] = []

    # Text encoder (only include trainable params)
    te_decay    = [p for n, p in text_named if p.requires_grad and not any(nd in n for nd in no_decay)]
    te_no_decay = [p for n, p in text_named if p.requires_grad and     any(nd in n for nd in no_decay)]
    if te_decay:
        groups.append({"params": te_decay,    "weight_decay": weight_decay, "lr": text_lr})
    if te_no_decay:
        groups.append({"params": te_no_decay, "weight_decay": 0.0,          "lr": text_lr})

    # Heads (GAT, fusion, classifier)
    h_decay    = [p for n, p in other_named if not any(nd in n for nd in no_decay)]
    h_no_decay = [p for n, p in other_named if     any(nd in n for nd in no_decay)]
    if h_decay:
        groups.append({"params": h_decay,    "weight_decay": weight_decay, "lr": head_lr})
    if h_no_decay:
        groups.append({"params": h_no_decay, "weight_decay": 0.0,          "lr": head_lr})

    return groups


def build_optimizer_scheduler(
    model:       nn.Module,
    cfg:         DotDict,
    total_steps: int,
) -> tuple[AdamW, object]:
    """Create AdamW + warmup LR scheduler.

    Args:
        model:       The full model.
        cfg:         Config DotDict.
        total_steps: Total number of gradient steps across all epochs.

    Returns:
        ``(optimizer, scheduler)``
    """
    t_cfg     = cfg.training
    text_lr   = t_cfg.lr
    head_lr   = text_lr * getattr(t_cfg, "head_lr_scale", 10)
    wd        = t_cfg.weight_decay
    warmup    = int(getattr(t_cfg, "warmup_steps", 100))

    param_groups = _build_param_groups(model, wd, text_lr, head_lr)
    optimizer    = AdamW(param_groups)

    scheduler_name = getattr(t_cfg, "scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )
    elif scheduler_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )
    else:   # "none" — constant LR
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    log.info(
        f"Optimizer: AdamW  text_lr={text_lr:.2e}  head_lr={head_lr:.2e}  "
        f"wd={wd:.2e}  scheduler={scheduler_name}  warmup={warmup}"
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Full training loop with early stopping and best-model checkpointing.

    Args:
        model:        :class:`~src.models.model.FakeNewsDetector`.
        cfg:          Loaded config ``DotDict``.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        test_loader:  Test DataLoader (evaluated after training ends).
        device:       Torch device string.
    """

    def __init__(
        self,
        model:        nn.Module,
        cfg:          DotDict,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        test_loader:  DataLoader,
        device:       str = "cpu",
    ) -> None:
        self.model        = model.to(device)
        self.cfg          = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.device       = device
        self.evaluator    = Evaluator(device)

        # Scheduler needs total_steps upfront
        total_steps = len(train_loader) * cfg.training.epochs
        self.optimizer, self.scheduler = build_optimizer_scheduler(
            model, cfg, total_steps
        )

        # ── Results directories ────────────────────────────────────────
        exp_name        = cfg.experiment.name
        out_dir         = Path(cfg.results.output_dir)
        self.ckpt_dir   = out_dir / "checkpoints" / exp_name
        self.metric_dir = out_dir / "metrics"      / exp_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)

        self.best_ckpt   = self.ckpt_dir / "best_model.pt"
        self.history_csv = self.metric_dir / "history.csv"

        # ── Training state ─────────────────────────────────────────────
        self.best_val_f1       = -1.0
        self.patience_counter  = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Returns:
            Mean cross-entropy loss over all batches.
        """
        self.model.train()
        total_loss = 0.0
        n_batches  = len(self.train_loader)
        log_every  = int(getattr(self.cfg.results, "log_every_n_steps", 10))

        for step, batch in enumerate(self.train_loader, start=1):
            batch  = _move_batch(batch, self.device)
            labels = batch["labels"]

            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.grad_clip,
            )

            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

            if step % log_every == 0 or step == n_batches:
                lr = self.optimizer.param_groups[0]["lr"]
                log.info(
                    f"  Epoch {epoch}  step {step}/{n_batches}  "
                    f"loss={loss.item():.4f}  lr={lr:.2e}"
                )

        return total_loss / n_batches

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, val_f1: float) -> None:
        torch.save(
            {
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_f1":          val_f1,
            },
            self.best_ckpt,
        )
        log.info(f"  Checkpoint saved -> {self.best_ckpt}  (val_f1={val_f1:.4f})")

    def _load_best_checkpoint(self) -> None:
        if not self.best_ckpt.exists():
            log.warning("No checkpoint found; using current model weights.")
            return
        ckpt = torch.load(self.best_ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        log.info(f"Loaded best checkpoint from {self.best_ckpt}")

    # ------------------------------------------------------------------
    # History / results
    # ------------------------------------------------------------------

    def _append_history(self, record: dict) -> None:
        self.history.append(record)
        write_header = not self.history_csv.exists()
        with self.history_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    def _save_test_results(self, test_metrics: dict) -> None:
        out_path = self.metric_dir / "test_metrics.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        log.info(f"Test metrics saved -> {out_path}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """Run training, then evaluate on the test set.

        Returns:
            Test-set metrics dict.
        """
        patience = self.cfg.training.early_stopping_patience
        log.info(
            f"Starting training: {self.cfg.training.epochs} epochs  "
            f"batch_size={self.cfg.training.batch_size}  "
            f"patience={patience}"
        )

        for epoch in range(1, self.cfg.training.epochs + 1):
            t0         = time.time()
            train_loss = self._train_epoch(epoch)
            val_mets   = self.evaluator.evaluate(self.model, self.val_loader)
            elapsed    = time.time() - t0

            log.info(
                f"Epoch {epoch:3d}/{self.cfg.training.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"{format_metrics(val_mets, prefix='val_')}  "
                f"({elapsed:.1f}s)"
            )

            # History record
            record = {"epoch": epoch, "train_loss": round(train_loss, 6)}
            record.update({f"val_{k}": round(v, 6) for k, v in val_mets.items()})
            self._append_history(record)

            # Early stopping + checkpointing
            val_f1 = val_mets.get("f1_macro", 0.0)
            if val_f1 > self.best_val_f1:
                self.best_val_f1      = val_f1
                self.patience_counter = 0
                self._save_checkpoint(val_f1)
            else:
                self.patience_counter += 1
                log.info(
                    f"  No improvement for {self.patience_counter}/{patience} epochs."
                )
                if self.patience_counter >= patience:
                    log.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        # ── Final evaluation on test set ───────────────────────────────
        log.info("Loading best checkpoint for final test evaluation…")
        self._load_best_checkpoint()

        test_mets = self.evaluator.evaluate(self.model, self.test_loader)
        log.info(f"TEST  {format_metrics(test_mets)}")
        self._save_test_results(test_mets)

        return test_mets
