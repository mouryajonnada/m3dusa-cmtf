"""
Ablation Study: Residual Connections & Cross-Attention Directionality
====================================================================
Runs multi-seed convergence training (Seeds 42, 123, 456) for:
1. cmtf_no_residual:     Bidirectional = True,  Residual = False
2. cmtf_unidirectional:  Bidirectional = False, Residual = True

Compares against Full CMTF (Bidirectional = True, Residual = True) and
Baseline M3DUSA (Late Fusion) with Welch's t-test statistical significance.

Usage::
    python scripts/train_ablation.py --seeds 42 123 456 --epochs 35 --patience 6 --device cpu
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
from scipy import stats
import torch

from src.data.dataset import get_loaders
from src.data.graph_builder import GraphBuilder
from src.models.model import FakeNewsDetector
from src.training.trainer import Trainer
from src.utils.config import load_config, merge_configs
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("train_ablation")

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
    p = argparse.ArgumentParser(description="CMTF Multi-Seed Ablation Study.")
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
        "--variants",
        nargs="+",
        default=["no_residual", "unidirectional"],
        choices=["no_residual", "unidirectional"],
        help="Ablation variants to train (default: no_residual unidirectional)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device to train on ('cpu' or 'cuda')",
    )
    p.add_argument(
        "--freeze_text_encoder",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to freeze RoBERTa text encoder (default: True)",
    )
    return p.parse_args()


def calculate_stats(values: list[float]) -> dict[str, float]:
    """Calculate mean and sample standard deviation."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std_val = math.sqrt(variance)
    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": n,
    }


def compute_welch_from_stats(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> float:
    """Compute Welch's t-test p-value directly from sample statistics."""
    if n1 < 2 or n2 < 2:
        return 1.0
    v1 = (s1 ** 2) / n1
    v2 = (s2 ** 2) / n2
    se = math.sqrt(v1 + v2)
    if se == 0:
        return 1.0
    t_stat = (m2 - m1) / se
    # Welch-Satterthwaite degrees of freedom
    num = (v1 + v2) ** 2
    den = (v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1)
    df = num / den if den > 0 else 1.0
    p_val = float(stats.t.sf(abs(t_stat), df=df) * 2.0)
    return round(p_val, 4)


def train_ablation_run(
    exp_name: str,
    fusion_overrides: dict,
    seed: int,
    epochs: int,
    patience: int,
    device: str,
    freeze_text_encoder: bool = True,
) -> dict[str, float]:
    """Run a single ablation run with specified seed and fusion overrides."""
    set_seed(seed)
    base_cfg_path = PROJECT_ROOT / "experiments" / "cross_modal_fusion.yaml"
    cfg = load_config(str(base_cfg_path))
    cfg = merge_configs(cfg, {
        "experiment": {"name": exp_name, "seed": seed},
        "training":   {"epochs": epochs, "early_stopping_patience": patience},
        "model": {
            "freeze_text_encoder": freeze_text_encoder,
            "fusion": fusion_overrides,
        },
    })

    log.info(
        f"==> Launching Ablation [{exp_name}] | Seed {seed} | Device {device} | "
        f"Max Epochs {epochs} | Patience {patience} | Overrides: {fusion_overrides}"
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

    # Override checkpoint path with seed
    seed_ckpt = trainer.ckpt_dir / f"seed_{seed}_best_model.pt"
    trainer.best_ckpt = seed_ckpt

    test_metrics = trainer.train()

    # Save individual seed metric path
    seed_metric_path = trainer.metric_dir / f"seed_{seed}_test_metrics.json"
    with open(seed_metric_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    log.info(f"[{exp_name}] Seed {seed} metrics saved -> {seed_metric_path}")

    # Canonical checkpoint
    canonical_ckpt = trainer.ckpt_dir / "best_model.pt"
    if not canonical_ckpt.exists() or seed == 42:
        import shutil
        shutil.copy2(seed_ckpt, canonical_ckpt)

    return test_metrics


def run_ablation_study(args: argparse.Namespace) -> None:
    variant_configs = {
        "no_residual": {
            "exp_name": "cmtf_no_residual",
            "overrides": {"use_residual": False, "bidirectional": True},
            "desc": "CMTF without Residual Connections",
        },
        "unidirectional": {
            "exp_name": "cmtf_unidirectional",
            "overrides": {"use_residual": True, "bidirectional": False},
            "desc": "CMTF Unidirectional (Text->Graph only)",
        },
    }

    ablation_results: dict[str, dict[str, list[float]]] = {}
    ablation_summaries: dict[str, dict[str, dict[str, float]]] = {}

    for var_key in args.variants:
        var_info = variant_configs[var_key]
        exp_name = var_info["exp_name"]
        ablation_results[exp_name] = {k: [] for k, _ in METRICS_ORDER}

        for seed in args.seeds:
            print(f"\n{'='*75}")
            print(f"  RUNNING ABLATION: {exp_name.upper()} — SEED {seed} ({len(args.seeds)} seeds)")
            print(f"{'='*75}\n")

            metrics = train_ablation_run(
                exp_name=exp_name,
                fusion_overrides=var_info["overrides"],
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
                device=args.device,
                freeze_text_encoder=args.freeze_text_encoder,
            )

            for key, _ in METRICS_ORDER:
                if key in metrics:
                    ablation_results[exp_name][key].append(metrics[key])

        # Compute summary
        ablation_summaries[exp_name] = {}
        for key, _ in METRICS_ORDER:
            ablation_summaries[exp_name][key] = calculate_stats(ablation_results[exp_name][key])

        # Save summary JSON
        metric_dir = PROJECT_ROOT / "results" / "metrics" / exp_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        summary_path = metric_dir / "multiseed_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(ablation_summaries[exp_name], f, indent=2)
        log.info(f"Summary for {exp_name} saved -> {summary_path}")

    # Load Full CMTF and Baseline M3DUSA summaries for comparative reporting
    full_cmtf_path = PROJECT_ROOT / "results" / "metrics" / "cross_modal_fusion" / "multiseed_summary.json"
    baseline_path = PROJECT_ROOT / "results" / "metrics" / "baseline_m3dusa" / "multiseed_summary.json"

    full_cmtf_summary = json.loads(full_cmtf_path.read_text(encoding="utf-8")) if full_cmtf_path.exists() else {}
    baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}

    generate_ablation_report(
        ablation_summaries=ablation_summaries,
        full_cmtf_summary=full_cmtf_summary,
        baseline_summary=baseline_summary,
        seeds=args.seeds,
    )


def generate_ablation_report(
    ablation_summaries: dict[str, dict],
    full_cmtf_summary: dict,
    baseline_summary: dict,
    seeds: list[int],
) -> None:
    """Generate consolidated CSV, JSON and print statistical ablation table."""
    print("\n" + "=" * 105)
    print(f"  CMTF ABLATION STUDY RESULTS ({len(seeds)} Seeds: {seeds})")
    print("=" * 105)
    header = f"  {'Metric':<18} | {'Baseline':<16} | {'No Residual':<16} | {'Unidirectional':<16} | {'Full CMTF':<16}"
    print(header)
    print("  " + "-" * 100)

    rows = []

    no_res_sum = ablation_summaries.get("cmtf_no_residual", {})
    unidir_sum = ablation_summaries.get("cmtf_unidirectional", {})

    for key, display in METRICS_ORDER:
        is_loss = (key == "loss")

        def fmt(stat):
            if not stat:
                return "N/A"
            m, s = stat["mean"], stat["std"]
            return f"{m:.4f} ± {s:.4f}" if is_loss else f"{m*100:.2f}% ± {s*100:.2f}%"

        b_stat = baseline_summary.get(key, {})
        nr_stat = no_res_sum.get(key, {})
        ud_stat = unidir_sum.get(key, {})
        fc_stat = full_cmtf_summary.get(key, {})

        b_str = fmt(b_stat)
        nr_str = fmt(nr_stat)
        ud_str = fmt(ud_stat)
        fc_str = fmt(fc_stat)

        print(f"  {display:<18} | {b_str:<16} | {nr_str:<16} | {ud_str:<16} | {fc_str:<16}")

        # Statistical significance of drops vs Full CMTF
        p_drop_no_res = compute_welch_from_stats(
            nr_stat.get("mean", 0), nr_stat.get("std", 0), nr_stat.get("n", 3),
            fc_stat.get("mean", 0), fc_stat.get("std", 0), fc_stat.get("n", 3)
        ) if nr_stat and fc_stat else 1.0

        p_drop_unidir = compute_welch_from_stats(
            ud_stat.get("mean", 0), ud_stat.get("std", 0), ud_stat.get("n", 3),
            fc_stat.get("mean", 0), fc_stat.get("std", 0), fc_stat.get("n", 3)
        ) if ud_stat and fc_stat else 1.0

        row_dict = {
            "metric": display,
            "metric_key": key,
            "baseline_m3dusa": b_str,
            "cmtf_no_residual": nr_str,
            "cmtf_unidirectional": ud_str,
            "full_cmtf": fc_str,
            "baseline_mean": b_stat.get("mean"),
            "baseline_std": b_stat.get("std"),
            "no_res_mean": nr_stat.get("mean"),
            "no_res_std": nr_stat.get("std"),
            "unidir_mean": ud_stat.get("mean"),
            "unidir_std": ud_stat.get("std"),
            "full_cmtf_mean": fc_stat.get("mean"),
            "full_cmtf_std": fc_stat.get("std"),
            "p_val_no_res_vs_full": p_drop_no_res,
            "p_val_unidir_vs_full": p_drop_unidir,
        }
        rows.append(row_dict)

    print("=" * 105 + "\n")

    # Export CSV
    df = pd.DataFrame(rows)
    csv_path = PROJECT_ROOT / "results" / "metrics" / "ablation_study.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Ablation study CSV saved -> {csv_path}")

    # Export JSON
    json_path = PROJECT_ROOT / "results" / "metrics" / "ablation_study.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    log.info(f"Ablation study JSON saved -> {json_path}")


def main():
    args = parse_args()
    run_ablation_study(args)


if __name__ == "__main__":
    main()
