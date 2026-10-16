"""
Multi-Part CSV Training Script
==============================
Trains the FakeNewsDetector (Cross-Modal Transformer Fusion or Late Fusion)
across all 5 split CSV parts of the FakeNewsCorpus dataset located in NEW_DATA/:

    NEW_DATA/news_cleaned_2018_02_13_part1.csv
    NEW_DATA/news_cleaned_2018_02_13_part2.csv
    NEW_DATA/news_cleaned_2018_02_13_part3.csv
    NEW_DATA/news_cleaned_2018_02_13_part4.csv
    NEW_DATA/news_cleaned_2018_02_13_part5.csv

Supported Training Modes:
-------------------------
1. Sequential (default): Continual learning across Part 1 -> Part 2 -> ... -> Part 5.
   Model weights carry forward across partitions, tracking progression and saving
   per-part checkpoints and a final cumulative model.
2. Combined: Pools extracted data from all 5 parts into a unified dataset and trains.
3. Single: Trains on a single specified part.

Usage Examples:
---------------
    # Train sequentially across all 5 CSVs (default)
    python scripts/train_parts.py

    # Train with custom epochs and batch size
    python scripts/train_parts.py --epochs 3 --batch-size 16 --samples-per-class 1000

    # Train in combined mode across all 5 parts
    python scripts/train_parts.py --mode combined --epochs 5

    # Train on specific parts only (e.g. parts 1 and 2)
    python scripts/train_parts.py --parts 1 2

    # Force device (e.g. cuda or cpu)
    python scripts/train_parts.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Windows UTF-8 console output fix ──────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Add project root to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

from src.data.dataset import (
    LABEL_FAKE,
    LABEL_REAL,
    PolitiFactDataset,
    PolitiFactSample,
    collate_fn,
    stratified_split,
)
from src.data.graph_builder import GraphBuilder, extract_hashtags
from src.data.preprocess import Tokenizer, clean_text
from src.models.model import FakeNewsDetector
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer, build_optimizer_scheduler
from src.utils.config import DotDict, load_config, merge_configs
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("train_parts")

# Allow arbitrarily large CSV fields for news content
csv.field_size_limit(min(2147483647, sys.maxsize))


# ---------------------------------------------------------------------------
# Data Extraction & Batched Graph Preprocessing
# ---------------------------------------------------------------------------

def extract_part_records(
    csv_path: Path,
    target_per_class: int = 1000,
    include_misinformation: bool = False,
) -> list[dict]:
    """Stream through a large partitioned CSV and extract balanced records."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Partition CSV not found: {csv_path}")

    log.info(f"Extracting balanced samples from: {csv_path.name}")
    log.info(f"Target: {target_per_class} fake + {target_per_class} real articles...")

    fake_records = []
    real_records = []

    misinfo_types = {"fake", "conspiracy", "junksci", "hate", "unreliable", "rumor"}

    t0 = time.time()
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            news_type = (row.get("type") or "").strip().lower()
            title = (row.get("title") or "").strip()

            if len(title) < 8 or title.lower() in ("none", "null", "untitled"):
                continue

            rec = {
                "id": row.get("id") or str(i),
                "title": title,
                "url": row.get("url", ""),
            }

            # Label assignment
            is_fake = (news_type in misinfo_types) if include_misinformation else (news_type == "fake")
            is_real = (news_type == "reliable")

            if is_fake and len(fake_records) < target_per_class:
                rec["label"] = LABEL_FAKE
                fake_records.append(rec)
            elif is_real and len(real_records) < target_per_class:
                rec["label"] = LABEL_REAL
                real_records.append(rec)

            if len(fake_records) >= target_per_class and len(real_records) >= target_per_class:
                break

            if (i + 1) % 50000 == 0:
                elapsed = time.time() - t0
                log.info(
                    f"  Read {i + 1:,} rows | Found {len(fake_records):,} fake, "
                    f"{len(real_records):,} real ({elapsed:.1f}s)"
                )

    log.info(
        f"Extraction complete for {csv_path.name}: "
        f"{len(fake_records):,} fake, {len(real_records):,} real in {time.time() - t0:.2f}s"
    )
    return fake_records + real_records


def build_samples_batched(
    records: list[dict],
    cfg: DotDict,
    graph_builder: GraphBuilder,
    tokenizer: Tokenizer,
) -> list[PolitiFactSample]:
    """Tokenize and construct HeteroData graphs using fast batched embedding."""
    t0 = time.time()
    valid_records = []
    titles = []

    for r in records:
        cleaned = clean_text(r["title"])
        if cleaned:
            valid_records.append(r)
            titles.append(cleaned)

    if not valid_records:
        return []

    log.info(f"Tokenizing {len(titles):,} titles...")
    token_list = [tokenizer.tokenize([t]) for t in titles]

    log.info(f"Computing batched sentence embeddings for {len(titles):,} claim titles...")
    news_embeddings = graph_builder._embedder.encode(
        titles,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu().float()

    # Pre-extract all hashtags and embed unique hashtags in one batch
    log.info("Extracting hashtags and building evidence graphs...")
    all_sample_tags = [extract_hashtags(t) for t in titles]
    unique_tags = sorted({tag for tags in all_sample_tags for tag in tags})

    tag_to_emb = {}
    if unique_tags:
        log.info(f"Embedding {len(unique_tags):,} unique hashtags in batched pass...")
        tag_embeddings = graph_builder._embedder.encode(
            unique_tags,
            batch_size=64,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).cpu().float()
        for tag, emb in zip(unique_tags, tag_embeddings):
            tag_to_emb[tag] = emb

    samples: list[PolitiFactSample] = []
    dim = news_embeddings.size(1)

    for i, (rec, title, tokens, tags) in enumerate(
        zip(valid_records, titles, token_list, all_sample_tags)
    ):
        graph = HeteroData()
        graph["news"].x = news_embeddings[i : i + 1]

        if not tags:
            graph["hashtag"].x = torch.zeros((0, dim), dtype=torch.float)
            graph["news", "has_hashtag", "hashtag"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )
            graph["hashtag", "rev_has_hashtag", "news"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )
        else:
            h_embs = torch.stack([tag_to_emb[t] for t in tags], dim=0)
            graph["hashtag"].x = h_embs

            H = len(tags)
            src = torch.zeros(H, dtype=torch.long)
            dst = torch.arange(H, dtype=torch.long)
            graph["news", "has_hashtag", "hashtag"].edge_index = torch.stack(
                [src, dst], dim=0
            )
            graph["hashtag", "rev_has_hashtag", "news"].edge_index = torch.stack(
                [dst, src], dim=0
            )

        samples.append(
            PolitiFactSample(
                id=rec["id"],
                title=title,
                label=rec["label"],
                tokens=tokens,
                graph=graph,
            )
        )

    log.info(f"Built {len(samples):,} graph samples in {time.time() - t0:.2f}s")
    return samples


def get_part_samples(
    part_num: int,
    csv_path: Path,
    cfg: DotDict,
    graph_builder: GraphBuilder,
    tokenizer: Tokenizer,
    cache_dir: Path,
    target_per_class: int = 1000,
    include_misinformation: bool = False,
    rebuild_cache: bool = False,
) -> list[PolitiFactSample]:
    """Retrieve processed samples for a part, loading from cache if available."""
    cache_file = cache_dir / f"part_{part_num}_processed.pt"
    if cache_file.exists() and not rebuild_cache:
        log.info(f"Loading cached samples for Part {part_num} from: {cache_file.name}")
        return torch.load(cache_file, map_location="cpu", weights_only=False)

    records = extract_part_records(
        csv_path,
        target_per_class=target_per_class,
        include_misinformation=include_misinformation,
    )
    samples = build_samples_batched(records, cfg, graph_builder, tokenizer)

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples, cache_file)
    log.info(f"Saved cache for Part {part_num} -> {cache_file}")
    return samples


def create_loaders(
    samples: list[PolitiFactSample],
    cfg: DotDict,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders from samples."""
    train_samples, val_samples, test_samples = stratified_split(
        samples,
        splits=list(cfg.data.splits),
        seed=cfg.experiment.seed,
    )
    shared_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    return (
        DataLoader(PolitiFactDataset(train_samples), shuffle=True, **shared_kwargs),
        DataLoader(PolitiFactDataset(val_samples), shuffle=False, **shared_kwargs),
        DataLoader(PolitiFactDataset(test_samples), shuffle=False, **shared_kwargs),
    )


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def run_sequential_training(
    parts: list[int],
    data_dir: Path,
    cfg: DotDict,
    output_dir: Path,
    cache_dir: Path,
    target_per_class: int,
    epochs_per_part: int,
    batch_size: int,
    device: str,
    include_misinformation: bool,
    rebuild_cache: bool,
) -> dict:
    """Train sequentially across parts: Part 1 -> Part 2 -> ... -> Part 5."""
    log.info("=" * 60)
    log.info("MODE: SEQUENTIAL CONTINUAL TRAINING ACROSS PARTS")
    log.info("=" * 60)

    graph_builder = GraphBuilder()
    tokenizer = Tokenizer(
        model_name=cfg.model.text_encoder,
        max_length=cfg.data.max_text_length,
    )

    metadata = graph_builder.metadata()
    model = FakeNewsDetector(cfg, metadata).to(device)
    evaluator = Evaluator(device)

    summary_records = []
    best_overall_f1 = -1.0
    best_part = None

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    test_loaders = {}

    for part_num in parts:
        csv_file = data_dir / f"news_cleaned_2018_02_13_part{part_num}.csv"
        log.info(f"\n{'='*25} PART {part_num}/{len(parts)} {'='*25}")
        log.info(f"Source file: {csv_file.name}")

        samples = get_part_samples(
            part_num=part_num,
            csv_path=csv_file,
            cfg=cfg,
            graph_builder=graph_builder,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            target_per_class=target_per_class,
            include_misinformation=include_misinformation,
            rebuild_cache=rebuild_cache,
        )

        train_loader, val_loader, test_loader = create_loaders(
            samples, cfg, batch_size=batch_size
        )
        test_loaders[part_num] = test_loader

        # Create trainer for this part with updated total steps
        part_cfg = DotDict(dict(cfg))
        part_cfg.training.epochs = epochs_per_part
        part_cfg.training.batch_size = batch_size
        part_cfg.results.output_dir = str(output_dir)
        part_cfg.experiment.name = f"part_{part_num}"

        trainer = Trainer(
            model=model,
            cfg=part_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )

        log.info(f"Training Part {part_num} for {epochs_per_part} epochs...")
        metrics = trainer.train()

        part_ckpt = ckpt_dir / f"part_{part_num}_model.pt"
        torch.save(model.state_dict(), part_ckpt)
        log.info(f"Saved Part {part_num} checkpoint -> {part_ckpt.name}")

        test_f1 = metrics.get("f1_macro", 0.0)
        test_acc = metrics.get("accuracy", 0.0)
        test_auc = metrics.get("auc_roc", 0.0)

        if test_f1 > best_overall_f1:
            best_overall_f1 = test_f1
            best_part = part_num
            overall_ckpt = ckpt_dir / "overall_best_model.pt"
            torch.save(model.state_dict(), overall_ckpt)
            log.info(f"*** New overall best model! (Part {part_num} F1={test_f1:.4f}) ***")

        summary_records.append({
            "part": f"Part {part_num}",
            "samples": len(samples),
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
            "test_auc_roc": test_auc,
            "test_loss": metrics.get("loss", 0.0),
        })

    # Final evaluation: evaluate current model on test sets of all parts
    log.info("\n" + "=" * 60)
    log.info("EVALUATING FINAL MODEL ON ALL TEST SETS")
    log.info("=" * 60)
    for p_num, t_loader in test_loaders.items():
        eval_metrics = evaluator.evaluate(model, t_loader)
        log.info(
            f"Final Model on Part {p_num} Test Set: "
            f"Acc={eval_metrics['accuracy']:.4f}  "
            f"F1={eval_metrics['f1_macro']:.4f}  "
            f"AUC={eval_metrics['auc_roc']:.4f}"
        )

    return {
        "summary": summary_records,
        "best_part": best_part,
        "best_f1": best_overall_f1,
    }


def run_combined_training(
    parts: list[int],
    data_dir: Path,
    cfg: DotDict,
    output_dir: Path,
    cache_dir: Path,
    target_per_class: int,
    epochs: int,
    batch_size: int,
    device: str,
    include_misinformation: bool,
    rebuild_cache: bool,
) -> dict:
    """Pool extracted samples across all 5 parts and train unified model."""
    log.info("=" * 60)
    log.info("MODE: UNIFIED COMBINED MULTI-PART TRAINING")
    log.info("=" * 60)

    graph_builder = GraphBuilder()
    tokenizer = Tokenizer(
        model_name=cfg.model.text_encoder,
        max_length=cfg.data.max_text_length,
    )

    combined_samples = []
    for part_num in parts:
        csv_file = data_dir / f"news_cleaned_2018_02_13_part{part_num}.csv"
        samples = get_part_samples(
            part_num=part_num,
            csv_path=csv_file,
            cfg=cfg,
            graph_builder=graph_builder,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            target_per_class=target_per_class,
            include_misinformation=include_misinformation,
            rebuild_cache=rebuild_cache,
        )
        combined_samples.extend(samples)

    log.info(f"Total combined samples from {len(parts)} parts: {len(combined_samples):,}")

    train_loader, val_loader, test_loader = create_loaders(
        combined_samples, cfg, batch_size=batch_size
    )

    metadata = graph_builder.metadata()
    model = FakeNewsDetector(cfg, metadata).to(device)

    cfg.training.epochs = epochs
    cfg.training.batch_size = batch_size
    cfg.results.output_dir = str(output_dir)
    cfg.experiment.name = "all_parts_combined"

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    test_metrics = trainer.train()
    return {
        "summary": [{
            "part": "All Parts Combined",
            "samples": len(combined_samples),
            "test_accuracy": test_metrics.get("accuracy", 0.0),
            "test_f1_macro": test_metrics.get("f1_macro", 0.0),
            "test_auc_roc": test_metrics.get("auc_roc", 0.0),
            "test_loss": test_metrics.get("loss", 0.0),
        }],
        "best_part": "Combined",
        "best_f1": test_metrics.get("f1_macro", 0.0),
    }


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train FakeNewsDetector across 5 split CSV parts."
    )
    p.add_argument(
        "--config", "-c",
        default="experiments/fakenewscorpus_cmtf.yaml",
        help="Path to YAML experiment config file.",
    )
    p.add_argument(
        "--data-dir", "-d",
        default="NEW_DATA",
        help="Directory containing the split CSV files.",
    )
    p.add_argument(
        "--parts", "-p",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Part numbers to include in training (default: 1 2 3 4 5).",
    )
    p.add_argument(
        "--mode", "-m",
        choices=["sequential", "combined", "single"],
        default="sequential",
        help="Training mode: 'sequential' (continual across parts), 'combined', or 'single'.",
    )
    p.add_argument(
        "--single-part",
        type=int,
        default=1,
        help="Part number to train on when mode='single'.",
    )
    p.add_argument(
        "--samples-per-class", "-s",
        type=int,
        default=1000,
        help="Number of samples per class (fake/real) extracted from each part (default: 1000).",
    )
    p.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Epochs per part (sequential) or total epochs (combined). Default: 3.",
    )
    p.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size for DataLoader (default: 16).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional learning rate override.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device ('cuda' or 'cpu'). Auto-detected if not specified.",
    )
    p.add_argument(
        "--include-misinformation",
        action="store_true",
        help="Include conspiracy, junksci, rumor, etc., as Fake (label 1).",
    )
    p.add_argument(
        "--output-dir", "-o",
        default="results/parts_training",
        help="Directory for saving checkpoints, logs, and metrics.",
    )
    p.add_argument(
        "--cache-dir",
        default="data/processed/parts",
        help="Directory for cached preprocessed samples.",
    )
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force re-extraction and re-embedding instead of loading cache.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Apply command-line overrides
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.lr = args.lr

    set_seed(cfg.experiment.seed)

    # Device auto-detection
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    log.info(f"Using device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    parts_to_run = [args.single_part] if args.mode == "single" else sorted(args.parts)

    t_start = time.time()
    if args.mode in ("sequential", "single"):
        results = run_sequential_training(
            parts=parts_to_run,
            data_dir=data_dir,
            cfg=cfg,
            output_dir=output_dir,
            cache_dir=cache_dir,
            target_per_class=args.samples_per_class,
            epochs_per_part=args.epochs,
            batch_size=args.batch_size,
            device=device,
            include_misinformation=args.include_misinformation,
            rebuild_cache=args.rebuild_cache,
        )
    else:  # combined
        results = run_combined_training(
            parts=parts_to_run,
            data_dir=data_dir,
            cfg=cfg,
            output_dir=output_dir,
            cache_dir=cache_dir,
            target_per_class=args.samples_per_class,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            include_misinformation=args.include_misinformation,
            rebuild_cache=args.rebuild_cache,
        )

    t_total = time.time() - t_start

    # Save summary report
    summary_df = pd.DataFrame(results["summary"])
    summary_csv = output_dir / "parts_summary.csv"
    summary_json = output_dir / "parts_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Formatted console output
    print("\n" + "=" * 70)
    print("  MULTI-PART TRAINING EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total time elapsed: {t_total:.1f}s ({t_total / 60:.1f} min)")
    print(f"Mode:               {args.mode.upper()}")
    print(f"Best Partition:     {results['best_part']} (F1 = {results['best_f1']:.4f})")
    print(f"Summary CSV:        {summary_csv}")
    print(f"Summary JSON:       {summary_json}")
    print("\n" + summary_df.to_string(index=False))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
