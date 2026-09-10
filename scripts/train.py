"""
Training Entry Point
====================
Run a full training experiment from a YAML config file.

Usage::

    # Baseline (M3DUSA late fusion)
    python scripts/train.py --config experiments/baseline_m3dusa.yaml

    # Novel cross-modal fusion
    python scripts/train.py --config experiments/cross_modal_fusion.yaml

    # Override config values on the command line
    python scripts/train.py --config experiments/baseline_m3dusa.yaml ^
        --override training.epochs=10 training.batch_size=16

    # Resume from existing processed data cache (skip re-embedding)
    python scripts/train.py --config experiments/baseline_m3dusa.yaml

On first run, the data pipeline downloads RoBERTa + all-MiniLM-L6-v2 and
embeds all 1,056 PolitiFact claims. Results are cached to
``data/processed/politifact_processed.pt`` for all subsequent runs.

Output
------
All outputs are written under ``results/``:

    results/checkpoints/{experiment_name}/best_model.pt
    results/metrics/{experiment_name}/test_metrics.json
    results/metrics/{experiment_name}/history.csv
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import os

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Make src/ importable when run from project root ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch

from src.utils.config       import load_config, DotDict
from src.utils.seed         import set_seed
from src.utils.logging      import get_logger
from src.data.dataset       import get_loaders
from src.data.graph_builder import GraphBuilder
from src.models.model       import FakeNewsDetector
from src.training.trainer   import Trainer

log = get_logger("train")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a FakeNewsDetector on PolitiFact."
    )
    p.add_argument(
        "--config", "-c",
        required=True,
        help="Path to a YAML experiment config file.",
    )
    p.add_argument(
        "--override", "-o",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override config values, e.g. "
            "--override training.epochs=5 training.batch_size=16"
        ),
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu).",
    )
    return p.parse_args()


def _apply_overrides(cfg: DotDict, overrides: list[str]) -> DotDict:
    """Apply ``KEY=VALUE`` override strings via :func:`merge_configs`.

    Supports arbitrary nesting (e.g. ``model.freeze_text_encoder=True``)
    and bool/int/float coercion.
    """
    from src.utils.config import merge_configs

    def _coerce(v: str):
        if v.lower() in ("true", "yes"):   return True
        if v.lower() in ("false", "no"):   return False
        try:    return int(v)
        except ValueError: pass
        try:    return float(v)
        except ValueError: pass
        return v

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item!r}")
        key_path, value_str = item.split("=", 1)
        keys  = key_path.split(".")
        value = _coerce(value_str)

        # Build nested dict matching the key path, then deep-merge
        nested: dict = value  # type: ignore[assignment]
        for k in reversed(keys):
            nested = {k: nested}
        cfg = merge_configs(cfg, nested)
        log.info(f"Override: {key_path} = {value!r}")
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.override:
        cfg = _apply_overrides(cfg, args.override)

    log.info(f"Experiment: {cfg.experiment.name}")
    log.info(f"Config:     {args.config}")

    # ── Reproducibility ───────────────────────────────────────────────
    set_seed(cfg.experiment.seed)
    log.info(f"Seed set to {cfg.experiment.seed}")

    # ── Device ────────────────────────────────────────────────────────
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU.")
        device = "cpu"
    log.info(f"Device: {device}")

    # ── Data pipeline ─────────────────────────────────────────────────
    log.info("Building data loaders…")
    train_loader, val_loader, test_loader = get_loaders(cfg)

    log.info(
        f"DataLoaders ready — "
        f"train={len(train_loader.dataset)}  "
        f"val={len(val_loader.dataset)}  "
        f"test={len(test_loader.dataset)}"
    )

    # ── Model ─────────────────────────────────────────────────────────
    log.info("Building model…")
    metadata = GraphBuilder().metadata()
    model    = FakeNewsDetector(cfg, metadata)

    total, trainable = model.num_parameters()
    log.info(f"Parameters — total: {total:,}  trainable: {trainable:,}")

    # ── Train ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    test_metrics = trainer.train()

    # ── Final report ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  FINAL TEST RESULTS — {cfg.experiment.name}")
    print("=" * 60)
    for metric, value in test_metrics.items():
        print(f"  {metric:<20} {value:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
