"""
Multi-Seed Convergence Training & Statistical Evaluation
=========================================================
Runs full training and evaluation over multiple random seeds to measure mean +- std,
implementing the rigorous scientific evaluation protocol from Martirano et al. (2025).

Usage::

    # Run both Baseline and CMTF over 3 seeds (42, 123, 456)
    python scripts/train_multiseed.py --seeds 42 123 456 --epochs 35 --patience 6

    # Run on GPU if available
    python scripts/train_multiseed.py --device cuda --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from pathlib import Path

# UTF-8 stdout fix on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.data.dataset import get_loaders
from src.data.graph_builder import GraphBuilder
from src.models.model import FakeNewsDetector
from src.training.trainer import Trainer
from src.utils.config import load_config, merge_configs
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("train_multiseed")

DEFAULT_SEEDS = [42, 123, 456]
METRICS_ORDER = [
    ("accuracy", "Accuracy"),
    ("f1_macro", "Macro F1"),
    ("f1_weighted", "Weighted F1"),
    ("f1_fake", "Fake News F1"),
    ("f1_real", "Real News F1"),
    ("precision_macro", "Macro Precision"),
    ("recall_macro", "Macro Recall"),
    ("auc_roc", "AUC-ROC"),
    ("loss", "Cross-Entropy Loss"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Seed Training & Statistical Evaluation.")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="List of random seeds (default: 42 123 456)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Maximum training epochs (default: 35)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early stopping patience (default: 6)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["baseline", "cmtf"],
        choices=["baseline", "cmtf"],
        help="Models to train (default: baseline cmtf)",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on ('cpu' or 'cuda')",
    )
    p.add_argument(
        "--freeze_text_encoder",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to freeze RoBERTa text encoder (default: True, recommended for CPU)",
    )
    return p.parse_args()


def calculate_stats(values: list[float]) -> dict[str, float]:
    """Calculate mean and sample standard deviation."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std_val  = math.sqrt(variance)
    return {
        "mean": round(mean_val, 4),
        "std":  round(std_val, 4),
        "min":  round(min(values), 4),
        "max":  round(max(values), 4),
        "n":    n,
    }


def compute_welch_ttest(vals1: list[float], vals2: list[float]) -> float:
    """Compute Welch's t-test p-value approximation."""
    try:
        from scipy import stats
        res = stats.ttest_ind(vals2, vals1, equal_var=False)
        return float(res.pvalue)
    except Exception:
        # Fallback normal approximation
        n1, n2 = len(vals1), len(vals2)
        if n1 < 2 or n2 < 2:
            return 1.0
        m1, m2 = sum(vals1) / n1, sum(vals2) / n2
        s1 = sum((x - m1)**2 for x in vals1) / (n1 - 1)
        s2 = sum((x - m2)**2 for x in vals2) / (n2 - 1)
        se = math.sqrt(s1 / n1 + s2 / n2)
        if se == 0:
            return 1.0
        t_stat = abs(m2 - m1) / se
        # Approximate p-value from t-stat with z-score
        return round(2.0 * (1.0 - 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2)))), 4)


def train_single_run(
    config_path: Path,
    seed: int,
    epochs: int,
    patience: int,
    device: str,
    freeze_text_encoder: bool = True,
) -> dict[str, float]:
    """Run a single training experiment with specified seed."""
    set_seed(seed)
    cfg = load_config(str(config_path))
    cfg = merge_configs(cfg, {
        "experiment": {"seed": seed},
        "training":   {"epochs": epochs, "early_stopping_patience": patience},
        "model":      {"freeze_text_encoder": freeze_text_encoder},
    })

    log.info(
        f"==> Launching {cfg.experiment.name} | Seed {seed} | Device {device} | "
        f"Max Epochs {epochs} | Patience {patience} | FreezeRoBERTa {freeze_text_encoder}"
    )

    train_loader, val_loader, test_loader = get_loaders(cfg)
    metadata = GraphBuilder().metadata()
    model = FakeNewsDetector(cfg, metadata)

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    # Override checkpoint paths for seed tracking
    seed_ckpt = trainer.ckpt_dir / f"seed_{seed}_best_model.pt"
    trainer.best_ckpt = seed_ckpt

    test_metrics = trainer.train()

    # Save individual seed metrics JSON
    seed_metric_path = trainer.metric_dir / f"seed_{seed}_test_metrics.json"
    with open(seed_metric_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    log.info(f"Seed {seed} metrics saved -> {seed_metric_path}")

    # Canonical checkpoint copy for web interface
    canonical_ckpt = trainer.ckpt_dir / "best_model.pt"
    if not canonical_ckpt.exists() or seed == 42:
        import shutil
        shutil.copy2(seed_ckpt, canonical_ckpt)
        log.info(f"Updated canonical checkpoint -> {canonical_ckpt}")

    return test_metrics


def run_multiseed_evaluation(args: argparse.Namespace) -> None:
    config_map = {
        "baseline": PROJECT_ROOT / "experiments" / "baseline_m3dusa.yaml",
        "cmtf":     PROJECT_ROOT / "experiments" / "cross_modal_fusion.yaml",
    }

    all_results: dict[str, dict[str, list[float]]] = {}
    summaries:   dict[str, dict[str, dict[str, float]]] = {}

    for model_key in args.models:
        config_path = config_map[model_key]
        exp_name = "baseline_m3dusa" if model_key == "baseline" else "cross_modal_fusion"
        all_results[exp_name] = {k: [] for k, _ in METRICS_ORDER}

        for seed in args.seeds:
            print(f"\n{'='*70}")
            print(f"  TRAINING {exp_name.upper()} — SEED {seed} ({len(args.seeds)} seeds total)")
            print(f"{'='*70}\n")

            metrics = train_single_run(
                config_path=config_path,
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
                device=args.device,
                freeze_text_encoder=args.freeze_text_encoder,
            )

            for key, _ in METRICS_ORDER:
                if key in metrics:
                    all_results[exp_name][key].append(metrics[key])

        # Compute multi-seed summary
        summaries[exp_name] = {}
        for key, display_name in METRICS_ORDER:
            summaries[exp_name][key] = calculate_stats(all_results[exp_name][key])

        # Save model multi-seed summary
        metric_dir = PROJECT_ROOT / "results" / "metrics" / exp_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        summary_path = metric_dir / "multiseed_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summaries[exp_name], f, indent=2)
        log.info(f"Multi-seed summary saved -> {summary_path}")

    # If both models were run, generate comparison table & export CSV
    if "baseline_m3dusa" in summaries and "cross_modal_fusion" in summaries:
        generate_multiseed_comparison(
            summaries["baseline_m3dusa"],
            summaries["cross_modal_fusion"],
            all_results["baseline_m3dusa"],
            all_results["cross_modal_fusion"],
            args.seeds,
        )


def generate_multiseed_comparison(
    base_summary: dict,
    cmtf_summary: dict,
    base_raw: dict,
    cmtf_raw: dict,
    seeds: list[int],
) -> None:
    """Generate final comparison.csv and print statistical summary."""
    rows = []
    print("\n" + "=" * 80)
    print(f"  MULTI-SEED EVALUATION REPORT ({len(seeds)} Seeds: {seeds})")
    print("=" * 80)
    print(f"  {'Metric':<20} | {'Baseline M3DUSA':<18} | {'Novel CMTF':<18} | {'Delta':<10} | {'p-value'}")
    print("  " + "-" * 76)

    for key, display in METRICS_ORDER:
        b_stat = base_summary[key]
        c_stat = cmtf_summary[key]

        b_mean, b_std = b_stat["mean"], b_stat["std"]
        c_mean, c_std = c_stat["mean"], c_stat["std"]
        delta = round(c_mean - b_mean, 4)

        # Statistical significance test
        p_val = compute_welch_ttest(base_raw[key], cmtf_raw[key])
        sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

        is_loss = (key == "loss")
        delta_str = f"{'+' if delta >= 0 else ''}{delta * 100:.2f}%" if not is_loss else f"{delta:+.4f}"

        # Formatted string: "87.55% ± 0.42%"
        if not is_loss:
            b_str = f"{b_mean * 100:.2f}% ± {b_std * 100:.2f}%"
            c_str = f"{c_mean * 100:.2f}% ± {c_std * 100:.2f}%"
        else:
            b_str = f"{b_mean:.4f} ± {b_std:.4f}"
            c_str = f"{c_mean:.4f} ± {c_std:.4f}"

        rows.append({
            "Metric": display,
            "Baseline_M3DUSA": b_str,
            "CMTF_Novel": c_str,
            "Baseline_Mean": b_mean,
            "Baseline_Std": b_std,
            "CMTF_Mean": c_mean,
            "CMTF_Std": c_std,
            "Delta_Absolute": delta,
            "Delta_Pct": delta_str,
            "p_value": p_val,
            "Significance": sig_marker,
        })

        print(f"  {display:<20} | {b_str:<18} | {c_str:<18} | {delta_str:<10} | p={p_val:.4f} ({sig_marker})")

    print("=" * 80 + "\n")

    # Export CSV for web interface and download
    df = pd.DataFrame(rows)
    csv_path = PROJECT_ROOT / "results" / "metrics" / "comparison.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Comparison CSV saved -> {csv_path}")

    # Also save structured JSON
    comp_json_path = PROJECT_ROOT / "results" / "metrics" / "multiseed_comparison.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    log.info(f"Comparison JSON saved -> {comp_json_path}")


def main():
    args = parse_args()
    run_multiseed_evaluation(args)


if __name__ == "__main__":
    main()
