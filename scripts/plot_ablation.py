"""
Plot Ablation Study Results
===========================
Generates publication-quality figure comparing:
1. Baseline M3DUSA (Late Fusion)
2. CMTF (No Residuals)
3. CMTF (Unidirectional)
4. Full CMTF (Novel - Bidirectional + Residuals)

Saves to results/plots/ablation_study.png
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

# UTF-8 stdout fix on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_ablation_plot():
    csv_path = PROJECT_ROOT / "results" / "metrics" / "ablation_study.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)

    # Key metrics to plot
    target_metrics = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "Macro F1"),
        ("f1_fake", "Fake News F1"),
        ("recall_macro", "Macro Recall"),
        ("auc_roc", "AUC-ROC"),
    ]

    filtered_df = df[df["metric_key"].isin([m[0] for m in target_metrics])].copy()
    filtered_df["display"] = filtered_df["metric_key"].map(dict(target_metrics))
    # Preserve order
    filtered_df["sort_order"] = filtered_df["metric_key"].map({m[0]: i for i, m in enumerate(target_metrics)})
    filtered_df = filtered_df.sort_values("sort_order").reset_index(drop=True)

    labels = filtered_df["display"].tolist()
    n_groups = len(labels)

    # Values (in percentage except AUC-ROC, or convert all to %)
    # For consistent 0-100 scale:
    base_means = filtered_df["baseline_mean"].values * 100
    base_stds = filtered_df["baseline_std"].values * 100

    no_res_means = filtered_df["no_res_mean"].values * 100
    no_res_stds = filtered_df["no_res_std"].values * 100

    unidir_means = filtered_df["unidir_mean"].values * 100
    unidir_stds = filtered_df["unidir_std"].values * 100

    full_means = filtered_df["full_cmtf_mean"].values * 100
    full_stds = filtered_df["full_cmtf_std"].values * 100

    # Set up publication figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=300)

    indices = np.arange(n_groups)
    width = 0.19

    # High-contrast, clean academic color palette
    c_base = "#64748b"    # Slate Gray (Baseline)
    c_no_res = "#eab308"  # Amber/Yellow (No Residual)
    c_unidir = "#3b82f6"  # Royal Blue (Unidirectional)
    c_full = "#10b981"    # Emerald Green (Full CMTF)

    rects1 = ax.bar(indices - 1.5 * width, base_means, width, yerr=base_stds,
                    capsize=4, label="Baseline M3DUSA (Late Fusion)", color=c_base, alpha=0.9, edgecolor="#334155", linewidth=1.2)
    rects2 = ax.bar(indices - 0.5 * width, no_res_means, width, yerr=no_res_stds,
                    capsize=4, label="CMTF w/o Residuals (No Skip)", color=c_no_res, alpha=0.9, edgecolor="#ca8a04", linewidth=1.2)
    rects3 = ax.bar(indices + 0.5 * width, unidir_means, width, yerr=unidir_stds,
                    capsize=4, label="CMTF Unidirectional (Text→Graph)", color=c_unidir, alpha=0.9, edgecolor="#1d4ed8", linewidth=1.2)
    rects4 = ax.bar(indices + 1.5 * width, full_means, width, yerr=full_stds,
                    capsize=4, label="Full CMTF (Bidirectional + Residuals)", color=c_full, alpha=0.95, edgecolor="#047857", linewidth=1.5)

    # Add numeric value badges above bars
    for rects, color in [(rects1, "#334155"), (rects2, "#854d0e"), (rects3, "#1e3a8a"), (rects4, "#064e3b")]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color=color)

    ax.set_ylabel("Score (%) — Mean ± 1 Std across 3 Seeds", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Ablation Study: Impact of Residual Skips & Bidirectional Attention on CMTF\n(Evaluated on PolitiFact Test Split across Seeds 42, 123, 456)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9.5)

    # Dynamic y-limits focused on informative region
    min_val = min(base_means.min(), no_res_means.min(), unidir_means.min())
    ax.set_ylim(max(75.0, min_val - 4.0), 101.0)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "results" / "plots" / "ablation_study.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Ablation figure saved successfully -> {plot_path}")


if __name__ == "__main__":
    generate_ablation_plot()
