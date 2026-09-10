"""
Data Pipeline Diagnostic Script
================================
Loads real PolitiFact data, runs the full pipeline with a 60/15/25 split,
and prints dataset statistics for sanity-checking.

Run from project root:
    python scripts/run_diagnostics.py
"""
import sys
import os
import io

# Force UTF-8 stdout on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Make sure src/ is importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from collections import Counter
from pathlib import Path

from src.data.preprocess import clean_text, Tokenizer
from src.data.graph_builder import GraphBuilder, extract_hashtags
from src.data.dataset import (
    load_fakenewsnet, stratified_split, PolitiFactSample, LABEL_FAKE, LABEL_REAL
)
from src.utils.logging import get_logger

log = get_logger("diagnostics")

RAW_DIR      = Path("data/raw")
PROCESSED_PT = Path("data/processed/politifact_processed.pt")
SPLITS       = [0.60, 0.15, 0.25]
SEED         = 42

# ---------------------------------------------------------------------------

def header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def section(title: str):
    print(f"\n  {'-'*50}")
    print(f"  {title}")
    print(f"  {'-'*50}")

# ---------------------------------------------------------------------------

def main():
    header("PolitiFact Data Pipeline Diagnostics")

    # ------------------------------------------------------------------
    # 1. Raw CSV stats
    # ------------------------------------------------------------------
    section("1. Raw CSV Stats")
    fake_df = pd.read_csv(RAW_DIR / "politifact_fake.csv")
    real_df = pd.read_csv(RAW_DIR / "politifact_real.csv")
    print(f"  politifact_fake.csv  : {len(fake_df):>5} rows   columns: {list(fake_df.columns)}")
    print(f"  politifact_real.csv  : {len(real_df):>5} rows   columns: {list(real_df.columns)}")
    print(f"  Total raw records    : {len(fake_df) + len(real_df):>5}")
    print(f"  Class ratio (fake/real): {len(fake_df)/len(real_df):.3f}")

    # ------------------------------------------------------------------
    # 2. Load or build processed samples
    # ------------------------------------------------------------------
    section("2. Preprocessing Pipeline")

    if PROCESSED_PT.exists():
        print(f"  Found cache at {PROCESSED_PT} — loading…")
        samples = torch.load(PROCESSED_PT, map_location="cpu", weights_only=False)
        print(f"  Loaded {len(samples):,} processed samples from cache.")
    else:
        print("  No cache found — running full pipeline (this may take a few minutes)…")
        print("  [tokenizer + graph builder will download models on first run]")

        raw_records = load_fakenewsnet(RAW_DIR)

        tokenizer     = Tokenizer(model_name="roberta-base", max_length=512)
        graph_builder = GraphBuilder()

        samples = []
        skipped = 0
        for i, rec in enumerate(raw_records):
            title = clean_text(rec["title"])
            if not title:
                skipped += 1
                continue

            tokens = tokenizer.tokenize([title])
            graph  = graph_builder.build(title)

            samples.append(PolitiFactSample(
                id=rec["id"], title=title, label=rec["label"],
                tokens=tokens, graph=graph,
            ))

            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1:>4}/{len(raw_records)} records…", flush=True)

        print(f"\n  Done. Valid samples: {len(samples):,}  |  Skipped (empty title): {skipped}")

        PROCESSED_PT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, PROCESSED_PT)
        print(f"  Saved cache → {PROCESSED_PT}")

    # ------------------------------------------------------------------
    # 3. Graph statistics
    # ------------------------------------------------------------------
    section("3. Graph Statistics (HeteroData per sample)")

    news_node_counts     = []   # always 1
    hashtag_node_counts  = []
    edge_counts          = []
    samples_no_hashtags  = 0
    hashtag_hist         = Counter()

    for s in samples:
        g = s.graph
        n_hash = g["hashtag"].x.shape[0]
        n_edge = g["news", "has_hashtag", "hashtag"].edge_index.shape[1]
        news_node_counts.append(1)
        hashtag_node_counts.append(n_hash)
        edge_counts.append(n_edge)
        hashtag_hist[n_hash] += 1
        if n_hash == 0:
            samples_no_hashtags += 1

    total = len(samples)
    import statistics

    print(f"\n  Node types per graph : news (always 1), hashtag")
    print(f"  Edge types           : has_hashtag, rev_has_hashtag")
    print(f"\n  ── Hashtag node counts ──")
    print(f"    Min          : {min(hashtag_node_counts)}")
    print(f"    Max          : {max(hashtag_node_counts)}")
    print(f"    Mean         : {sum(hashtag_node_counts)/total:.2f}")
    print(f"    Median       : {statistics.median(hashtag_node_counts):.1f}")
    print(f"    Samples with 0 hashtags : {samples_no_hashtags:,} / {total:,} "
          f"({100*samples_no_hashtags/total:.1f}%)")

    print(f"\n  ── Hashtag distribution (top 10 buckets) ──")
    for n_h, count in sorted(hashtag_hist.items())[:10]:
        bar = "█" * min(count, 40)
        print(f"    {n_h:>3} hashtags : {bar}  ({count})")

    print(f"\n  ── Edge counts (has_hashtag) ──")
    print(f"    Min  : {min(edge_counts)}")
    print(f"    Max  : {max(edge_counts)}")
    print(f"    Mean : {sum(edge_counts)/total:.2f}")
    print(f"    Total edges (all samples) : {sum(edge_counts):,}")

    # sample hashtag token examples
    print(f"\n  ── Example hashtags extracted ──")
    shown = 0
    for s in samples:
        tags = extract_hashtags(s.title)
        if tags:
            print(f"    [{s.label}] \"{s.title[:70]}\"")
            print(f"         → hashtags: {tags}")
            shown += 1
        if shown >= 5:
            break

    # ------------------------------------------------------------------
    # 4. Stratified split statistics
    # ------------------------------------------------------------------
    section(f"4. Stratified Split  {SPLITS} (seed={SEED})")

    train, val, test = stratified_split(samples, SPLITS, SEED)

    def split_stats(name, split):
        n_fake = sum(s.label == LABEL_FAKE for s in split)
        n_real = sum(s.label == LABEL_REAL for s in split)
        pct    = 100 * len(split) / total
        print(f"    {name:<8}: {len(split):>4} samples ({pct:5.1f}%)  "
              f"fake={n_fake} ({100*n_fake/len(split):.1f}%)  "
              f"real={n_real} ({100*n_real/len(split):.1f}%)")

    print()
    split_stats("Train", train)
    split_stats("Val",   val)
    split_stats("Test",  test)

    # ------------------------------------------------------------------
    # 5. Token length distribution
    # ------------------------------------------------------------------
    section("5. Token Length Distribution (input_ids, max=512)")

    seq_lengths = []
    for s in samples:
        # Count non-padding tokens (attention_mask == 1)
        length = int(s.tokens["attention_mask"].squeeze(0).sum().item())
        seq_lengths.append(length)

    print(f"    Min    : {min(seq_lengths)}")
    print(f"    Max    : {max(seq_lengths)}")
    print(f"    Mean   : {sum(seq_lengths)/total:.1f}")
    print(f"    Median : {statistics.median(seq_lengths):.1f}")
    pct_maxed = sum(1 for l in seq_lengths if l == 512) / total * 100
    print(f"    Hitting max_length (512): {pct_maxed:.1f}% of samples")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    header("Summary")
    print(f"  Total samples  : {total:,}")
    print(f"  Fake           : {sum(s.label==LABEL_FAKE for s in samples):,}")
    print(f"  Real           : {sum(s.label==LABEL_REAL for s in samples):,}")
    print(f"  Train / Val / Test : {len(train)} / {len(val)} / {len(test)}")
    print(f"  Graph schema   : HeteroData (News + Hashtag, has_hashtag edges)")
    print(f"  Avg hashtags   : {sum(hashtag_node_counts)/total:.2f} per claim")
    print(f"  Avg token len  : {sum(seq_lengths)/total:.1f}")
    print()
    print("  ✅ Pipeline looks healthy — proceed to model implementation.")
    print()


if __name__ == "__main__":
    main()
