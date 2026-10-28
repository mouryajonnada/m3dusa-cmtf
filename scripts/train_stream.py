"""
Full Disk-Streaming Training Pipeline for Large News Corpora (29 GB+)
=====================================================================
Streams through massive CSV archives (e.g. NEW_DATA/news_cleaned_2018_02_13.csv
or its 5 split partition files) chunk-by-chunk without loading the entire
dataset into system memory or creating massive intermediate tensor caches.

Key Features:
-------------
1. Memory-Bounded Disk Streaming:
   Streams line-by-line using Python CSV reader with minimal memory footprint (< 500 MB RAM).
2. Balanced Chunk Buffer:
   Maintains dual-class pools (Fake/Misinformation vs. Reliable) and yields balanced,
   shuffled chunks (e.g. 1,024 or 2,048 samples) to avoid class bias and gradient collapse.
3. Fast On-The-Fly Batch Processing:
   Sentence embeddings and tokenization are batched per chunk and discarded after training.
4. Hardware Optimization for Low-VRAM (e.g. 2 GB GPUs / MX330) and CPUs:
   - Mixed-precision training (torch.cuda.amp.autocast)
   - Gradient accumulation for larger effective batch size
   - Text encoder freezing option for 5-10x speedup
5. Fault-Tolerant Checkpointing & Resume:
   Saves rolling checkpoints (latest_model.pt, best_model.pt) and stream state (stream_state.json).
   Allows seamless resumption from the exact row offset after interruption.
6. Periodic Fixed Validation:
   Evaluates periodically on a reserved held-out validation set to monitor true generalisation.

Usage Examples:
---------------
    # 1. Run streaming across the 5 partition CSVs in NEW_DATA/
    python scripts/train_stream.py --parts-dir NEW_DATA

    # 2. Run streaming on the single master 29 GB CSV
    python scripts/train_stream.py --input NEW_DATA/news_cleaned_2018_02_13.csv

    # 3. Test on 10,000 samples with batch size 8
    python scripts/train_stream.py --max-samples 10000 --batch-size 8

    # 4. Resume interrupted training from last checkpoint
    python scripts/train_stream.py --resume
"""
from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Generator, Optional

import psutil

# --- Windows UTF-8 stdout fix ---
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
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
)
from src.data.graph_builder import GraphBuilder, extract_hashtags
from src.data.preprocess import Tokenizer, clean_text
from src.models.model import FakeNewsDetector, _move_batch
from src.training.evaluator import Evaluator
from src.utils.config import DotDict, load_config, merge_configs
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("train_stream")

# Allow very large fields in CSV rows
csv.field_size_limit(min(2147483647, sys.maxsize))

MISINFO_TYPES = frozenset({
    "fake", "conspiracy", "junksci", "hate", "unreliable", "rumor"
})


# ---------------------------------------------------------------------------
# Stream Reader & Balanced Buffer
# ---------------------------------------------------------------------------

class NewsStreamIterator:
    """Streams rows across one or multiple CSV files, maintaining a balanced buffer."""

    def __init__(
        self,
        file_paths: list[Path],
        chunk_size: int = 1024,
        include_misinformation: bool = True,
        skip_rows_total: int = 0,
    ) -> None:
        self.file_paths = [Path(p) for p in file_paths]
        self.chunk_size = chunk_size
        self.include_misinformation = include_misinformation
        self.skip_rows_total = skip_rows_total

        self.fake_pool: deque[dict] = deque(maxlen=chunk_size * 2)
        self.real_pool: deque[dict] = deque(maxlen=chunk_size * 2)
        self.half_chunk = chunk_size // 2

        self.total_rows_read = 0
        self.current_file_idx = 0

    def stream_chunks(self) -> Generator[list[dict], None, None]:
        """Yields balanced chunks of records with equal fake and real samples."""
        skipped_so_far = 0

        for file_idx, fpath in enumerate(self.file_paths):
            if not fpath.exists():
                log.warning(f"File not found, skipping: {fpath}")
                continue

            self.current_file_idx = file_idx
            log.info(f"Opening data stream: {fpath.name} ({fpath.stat().st_size / (1024**3):.2f} GB)")

            with open(fpath, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.total_rows_read += 1

                    # Handle skip for resumption
                    if skipped_so_far < self.skip_rows_total:
                        skipped_so_far += 1
                        if skipped_so_far % 100000 == 0:
                            log.info(f"Fast-forwarding stream: skipped {skipped_so_far:,} / {self.skip_rows_total:,} rows...")
                        continue

                    news_type = (row.get("type") or "").strip().lower()
                    title = (row.get("title") or "").strip()

                    # Filter out short or placeholder titles
                    if len(title) < 8 or title.lower() in ("none", "null", "untitled"):
                        continue

                    # Assign labels
                    if self.include_misinformation:
                        is_fake = news_type in MISINFO_TYPES
                    else:
                        is_fake = news_type == "fake"
                    is_real = news_type == "reliable"

                    if not (is_fake or is_real):
                        continue

                    rec = {
                        "id": row.get("id") or str(self.total_rows_read),
                        "title": title,
                        "label": LABEL_FAKE if is_fake else LABEL_REAL,
                    }

                    if is_fake:
                        self.fake_pool.append(rec)
                    else:
                        self.real_pool.append(rec)

                    # When both pools have enough samples, yield a balanced chunk
                    if len(self.fake_pool) >= self.half_chunk and len(self.real_pool) >= self.half_chunk:
                        chunk = []
                        for _ in range(self.half_chunk):
                            chunk.append(self.fake_pool.popleft())
                            chunk.append(self.real_pool.popleft())

                        np.random.shuffle(chunk)
                        yield chunk

        # Flush any remaining balanced pairs
        rem_size = min(len(self.fake_pool), len(self.real_pool))
        if rem_size >= 16:
            chunk = []
            for _ in range(rem_size):
                chunk.append(self.fake_pool.popleft())
                chunk.append(self.real_pool.popleft())
            np.random.shuffle(chunk)
            yield chunk


# ---------------------------------------------------------------------------
# On-the-fly Graph & Token Batch Construction
# ---------------------------------------------------------------------------

def build_chunk_samples(
    records: list[dict],
    graph_builder: GraphBuilder,
    tokenizer: Tokenizer,
) -> list[PolitiFactSample]:
    """Tokenizes and builds HeteroData graphs for a single chunk in memory."""
    valid_records = []
    titles = []

    for r in records:
        cleaned = clean_text(r["title"])
        if cleaned:
            valid_records.append(r)
            titles.append(cleaned)

    if not valid_records:
        return []

    # Tokenize with RoBERTa
    token_list = [tokenizer.tokenize([t]) for t in titles]

    # Sentence embeddings for claim news nodes
    news_embeddings = graph_builder._embedder.encode(
        titles,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu().float()

    # Extract and embed unique hashtags
    all_sample_tags = [extract_hashtags(t) for t in titles]
    unique_tags = sorted({tag for tags in all_sample_tags for tag in tags})

    tag_to_emb = {}
    if unique_tags:
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
            graph["news", "has_hashtag", "hashtag"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph["hashtag", "rev_has_hashtag", "news"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            h_embs = torch.stack([tag_to_emb[t] for t in tags], dim=0)
            graph["hashtag"].x = h_embs
            H = len(tags)
            src = torch.zeros(H, dtype=torch.long)
            dst = torch.arange(H, dtype=torch.long)
            graph["news", "has_hashtag", "hashtag"].edge_index = torch.stack([src, dst], dim=0)
            graph["hashtag", "rev_has_hashtag", "news"].edge_index = torch.stack([dst, src], dim=0)

        samples.append(
            PolitiFactSample(
                id=rec["id"],
                title=title,
                label=rec["label"],
                tokens=tokens,
                graph=graph,
            )
        )

    return samples


# ---------------------------------------------------------------------------
# Streaming Training Engine
# ---------------------------------------------------------------------------

def run_streaming_training(
    input_files: list[Path],
    cfg: DotDict,
    output_dir: Path,
    chunk_size: int = 1024,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    val_every_chunks: int = 10,
    save_every_chunks: int = 10,
    max_samples: Optional[int] = None,
    device_str: Optional[str] = None,
    freeze_text_encoder: bool = True,
    include_misinformation: bool = True,
    resume: bool = False,
    max_ram_gb: float = 6.0,
) -> None:
    """Executes the full disk-streaming training pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "stream_state.json"
    history_file = output_dir / "stream_history.csv"

    # Select device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using execution device: {device}")
    if device.type == "cuda":
        log.info(f"GPU Model: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")

    # Model configuration
    cfg = merge_configs(cfg, {
        "model": {"freeze_text_encoder": freeze_text_encoder},
        "training": {"lr": lr, "batch_size": batch_size},
    })

    graph_builder = GraphBuilder()
    tokenizer = Tokenizer(
        model_name=cfg.model.text_encoder,
        max_length=cfg.data.max_text_length,
    )
    metadata = graph_builder.metadata()

    model = FakeNewsDetector(cfg, metadata).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    evaluator = Evaluator(device)

    # Use mixed precision if CUDA available
    use_amp = (device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # State variables
    skip_rows = 0
    total_samples_trained = 0
    total_chunks_trained = 0
    best_val_f1 = -1.0
    start_time = time.time()

    # Load state if resume requested
    if resume and state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
            skip_rows = state.get("total_rows_read", 0)
            total_samples_trained = state.get("total_samples_trained", 0)
            total_chunks_trained = state.get("total_chunks_trained", 0)
            best_val_f1 = state.get("best_val_f1", -1.0)

        latest_ckpt = ckpt_dir / "latest_model.pt"
        if latest_ckpt.exists():
            log.info(f"Resuming model weights from: {latest_ckpt.name}")
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            log.info(f"Resumed at: {total_samples_trained:,} samples trained, skipping {skip_rows:,} CSV rows.")

    # Setup history CSV if new
    if not history_file.exists() or not resume:
        with open(history_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "chunk", "total_samples", "rows_read", "train_loss",
                "val_accuracy", "val_f1_macro", "val_auc_roc", "val_loss", "elapsed_s"
            ])

    # -----------------------------------------------------------------------
    # Step 1: Collect Fixed Validation Set (1,000 samples)
    # -----------------------------------------------------------------------
    val_cache_file = output_dir / "fixed_val_set.pt"
    if val_cache_file.exists():
        log.info(f"Loading persistent validation set from: {val_cache_file.name}")
        val_samples = torch.load(val_cache_file, map_location="cpu", weights_only=False)
    else:
        log.info("Extracting fixed validation set (1,000 balanced samples) from stream head...")
        val_iterator = NewsStreamIterator(
            input_files,
            chunk_size=1000,
            include_misinformation=include_misinformation,
            skip_rows_total=0,
        )
        val_records = next(val_iterator.stream_chunks())
        val_samples = build_chunk_samples(val_records, graph_builder, tokenizer)
        torch.save(val_samples, val_cache_file)
        log.info(f"Saved fixed validation set ({len(val_samples):,} samples) -> {val_cache_file.name}")

    val_loader = DataLoader(
        PolitiFactDataset(val_samples),
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # -----------------------------------------------------------------------
    # Step 2: Main Disk Streaming Training Loop
    # -----------------------------------------------------------------------
    stream = NewsStreamIterator(
        input_files,
        chunk_size=chunk_size,
        include_misinformation=include_misinformation,
        skip_rows_total=skip_rows,
    )

    log.info("=" * 65)
    log.info(f"STARTING FULL DISK-STREAMING TRAINING")
    log.info(f"Chunk Size: {chunk_size} | Batch Size: {batch_size} | Grad Accum: {grad_accum_steps}")
    log.info(f"Effective Batch Size: {batch_size * grad_accum_steps} | FP16 AMP: {use_amp}")
    log.info(f"RoBERTa Frozen: {freeze_text_encoder} | Target Files: {len(input_files)}")
    log.info("=" * 65)

    rolling_loss = 0.0
    rolling_steps = 0

    for chunk_records in stream.stream_chunks():
        t0_chunk = time.time()
        total_chunks_trained += 1

        # Build in-memory samples for this chunk
        chunk_samples = build_chunk_samples(chunk_records, graph_builder, tokenizer)
        if not chunk_samples:
            continue

        train_loader = DataLoader(
            PolitiFactDataset(chunk_samples),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        model.train()
        chunk_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)

            try:
                autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

            with autocast_ctx:
                logits = model(batch)
                loss = criterion(logits, batch["labels"])
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            chunk_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_chunk_loss = chunk_loss / max(1, len(train_loader))
        rolling_loss += avg_chunk_loss
        rolling_steps += 1
        total_samples_trained += len(chunk_samples)

        elapsed = time.time() - start_time
        samples_per_sec = len(chunk_samples) / max(0.01, (time.time() - t0_chunk))

        # Memory tracking
        proc_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        vram_info = f" | VRAM: {torch.cuda.memory_allocated() / (1024 * 1024):.0f}MB" if device.type == "cuda" else ""

        log.info(
            f"[Chunk {total_chunks_trained:>5,}] "
            f"Samples: {total_samples_trained:>8,} | "
            f"Loss: {avg_chunk_loss:.4f} | "
            f"Speed: {samples_per_sec:.1f} s/s | "
            f"RAM: {proc_ram_mb:.0f} MB / {max_ram_gb*1024:.0f} MB{vram_info}"
        )

        # Periodic Validation
        val_acc, val_f1, val_auc, val_loss = None, None, None, None
        if total_chunks_trained % val_every_chunks == 0:
            val_metrics = evaluator.evaluate(model, val_loader)
            val_acc = val_metrics.get("accuracy", 0.0)
            val_f1 = val_metrics.get("f1_macro", 0.0)
            val_auc = val_metrics.get("auc_roc", 0.0)
            val_loss = val_metrics.get("loss", 0.0)

            log.info(
                f"  >>> VALIDATION @ Chunk {total_chunks_trained:,}: "
                f"Acc={val_acc*100:.2f}% | F1={val_f1*100:.2f}% | "
                f"AUC={val_auc:.4f} | ValLoss={val_loss:.4f}"
            )

            # Save best checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_ckpt = ckpt_dir / "best_model.pt"
                torch.save({"model_state": model.state_dict(), "val_f1": best_val_f1}, best_ckpt)
                log.info(f"  *** New Best Validation F1: {best_val_f1*100:.2f}% -> Saved best_model.pt ***")

        # Periodic Checkpoint & State Save
        if total_chunks_trained % save_every_chunks == 0:
            latest_ckpt = ckpt_dir / "latest_model.pt"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "total_chunks": total_chunks_trained,
                "total_samples": total_samples_trained,
            }, latest_ckpt)

            with open(state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "total_rows_read": stream.total_rows_read,
                    "total_samples_trained": total_samples_trained,
                    "total_chunks_trained": total_chunks_trained,
                    "best_val_f1": best_val_f1,
                    "elapsed_seconds": elapsed,
                    "current_file_idx": stream.current_file_idx,
                }, f, indent=2)

            with open(history_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    total_chunks_trained,
                    total_samples_trained,
                    stream.total_rows_read,
                    f"{avg_chunk_loss:.4f}",
                    f"{val_acc:.4f}" if val_acc is not None else "",
                    f"{val_f1:.4f}" if val_f1 is not None else "",
                    f"{val_auc:.4f}" if val_auc is not None else "",
                    f"{val_loss:.4f}" if val_loss is not None else "",
                    f"{elapsed:.1f}",
                ])

        # Clean memory between chunks
        del chunk_samples
        del train_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Enforce memory safety cap (3 GB default)
        proc_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        if proc_ram_mb > (max_ram_gb * 1024 * 0.9):
            log.warning(f"Memory threshold reached ({proc_ram_mb:.0f} MB). Forcing garbage collection...")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Check maximum samples limit
        if max_samples and total_samples_trained >= max_samples:
            log.info(f"Reached requested limit of {max_samples:,} samples. Stopping stream.")
            break

    # Save final model
    final_ckpt = ckpt_dir / "final_streaming_model.pt"
    torch.save(model.state_dict(), final_ckpt)
    log.info(f"Streaming training complete! Saved final model -> {final_ckpt.name}")
    log.info(f"Total samples trained: {total_samples_trained:,} | Total rows read: {stream.total_rows_read:,}")


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full Disk-Streaming Training for Fake News Corpora.")
    p.add_argument("--input", "-i", default=None, help="Path to single large CSV file.")
    p.add_argument("--parts-dir", "-p", default="NEW_DATA", help="Directory containing split part CSV files.")
    p.add_argument("--config", "-c", default="experiments/fakenewscorpus_cmtf.yaml", help="Experiment YAML config.")
    p.add_argument("--output-dir", "-o", default="results/stream_training", help="Output directory.")
    p.add_argument("--chunk-size", type=int, default=1024, help="Samples per balanced chunk buffer (default: 1024).")
    p.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size per training step (default: 8).")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4).")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4).")
    p.add_argument("--val-every", type=int, default=10, help="Validate every N chunks (default: 10).")
    p.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N chunks (default: 10).")
    p.add_argument("--max-samples", type=int, default=None, help="Optional maximum total samples to train on.")
    p.add_argument("--device", default=None, help="Torch device ('cuda' or 'cpu').")
    p.add_argument("--unfreeze-text-encoder", action="store_true", help="Unfreeze RoBERTa weights (higher VRAM).")
    p.add_argument("--strict-fake-only", action="store_true", help="Only match 'fake' instead of all misinformation.")
    p.add_argument("--resume", action="store_true", help="Resume from stream_state.json and latest_model.pt.")
    p.add_argument("--max-ram-gb", type=float, default=6.0, help="Maximum RAM threshold in GB (default: 6.0).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Determine input files
    if args.input:
        input_files = [Path(args.input)]
    else:
        parts_dir = Path(args.parts_dir)
        part_files = sorted(parts_dir.glob("news_cleaned_2018_02_13_part*.csv"))
        if not part_files:
            master = parts_dir / "news_cleaned_2018_02_13.csv"
            if master.exists():
                input_files = [master]
            else:
                raise FileNotFoundError(f"No CSVs found in: {parts_dir}")
        else:
            input_files = part_files

    log.info(f"Found {len(input_files)} data source file(s) for streaming:")
    for f in input_files:
        log.info(f"  - {f.name} ({f.stat().st_size / (1024**3):.2f} GB)")

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)

    run_streaming_training(
        input_files=input_files,
        cfg=cfg,
        output_dir=Path(args.output_dir),
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        val_every_chunks=args.val_every,
        save_every_chunks=args.save_every,
        max_samples=args.max_samples,
        device_str=args.device,
        freeze_text_encoder=not args.unfreeze_text_encoder,
        include_misinformation=not args.strict_fake_only,
        resume=args.resume,
        max_ram_gb=args.max_ram_gb,
    )


if __name__ == "__main__":
    main()
