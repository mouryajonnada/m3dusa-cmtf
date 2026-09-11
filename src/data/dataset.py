"""
PolitiFact Dataset Loader
=========================
Orchestrates the full data pipeline:

1. ``load_fakenewsnet()``    — reads raw FakeNewsNet CSVs, assigns labels
2. ``PolitiFactSample``      — dataclass holding one processed claim
3. ``stratified_split()``    — stratified train / val / test split
4. ``PolitiFactDataset``     — ``torch.utils.data.Dataset`` wrapper
5. ``collate_fn()``          — custom collate merging tokens + PyG graphs
6. ``get_loaders()``         — full pipeline entry-point, returns DataLoaders

Expected raw data layout::

    data/raw/
        politifact_fake.csv   # columns: id, url, title, tweet_ids
        politifact_real.csv   # columns: id, url, title, tweet_ids

Label convention: ``real = 0``, ``fake = 1``.

Processing results are cached to ``cfg.data.processed_dir/politifact_processed.pt``
after the first run and reloaded on subsequent calls, skipping re-tokenisation
and re-embedding.

Usage::

    from src.utils.config import load_config
    from src.data.dataset import get_loaders

    cfg = load_config("experiments/baseline_m3dusa.yaml")
    train_loader, val_loader, test_loader = get_loaders(cfg)

    batch = next(iter(train_loader))
    print(batch["input_ids"].shape)   # (B, 512)
    print(batch["labels"].shape)      # (B,)
    print(batch["graph_batch"])       # PyG Batch object
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, HeteroData

from src.data.graph_builder import GraphBuilder
from src.data.preprocess import Tokenizer, clean_text
from src.utils.config import DotDict
from src.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
LABEL_REAL: int = 0
LABEL_FAKE: int = 1


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PolitiFactSample:
    """One processed PolitiFact claim with all pre-computed features.

    Attributes:
        id:     Unique sample identifier from the raw CSV.
        title:  Cleaned claim title string.
        label:  Integer label (0 = real, 1 = fake).
        tokens: Dict of tokenised tensors from :class:`~src.data.preprocess.Tokenizer`.
                Keys: ``"input_ids"`` ``(1, seq_len)``,
                      ``"attention_mask"`` ``(1, seq_len)``.
        graph:  PyG ``HeteroData`` object matching the partial M3DUSA schema
                (News + Hashtag node types, has_hashtag edges).
    """
    id: str
    title: str
    label: int
    tokens: dict[str, torch.Tensor]
    graph: HeteroData


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first candidate that exists in *df* (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_fakenewsnet(raw_dir: Union[str, Path]) -> list[dict]:
    """Load raw FakeNewsNet PolitiFact CSVs and return normalised records.

    Reads ``politifact_fake.csv`` and ``politifact_real.csv`` from *raw_dir*,
    assigns binary labels (0 = real, 1 = fake), and applies robust column
    detection to handle minor naming variations.

    Args:
        raw_dir: Directory containing ``politifact_fake.csv`` and
                 ``politifact_real.csv``.

    Returns:
        List of dicts with keys ``"id"``, ``"title"``, ``"label"``.

    Raises:
        FileNotFoundError: If either CSV file is missing.
        ValueError:        If no recognisable text column is found in a CSV.
    """
    raw_dir = Path(raw_dir)
    records: list[dict] = []

    for csv_name, label in [
        ("politifact_fake.csv", LABEL_FAKE),
        ("politifact_real.csv", LABEL_REAL),
    ]:
        path = raw_dir / csv_name
        if not path.exists():
            raise FileNotFoundError(
                f"Expected FakeNewsNet CSV not found: {path}\n"
                "Download instructions: https://github.com/KaiDMML/FakeNewsNet"
            )

        df = pd.read_csv(path)
        log.info(f"Loaded {len(df):,} rows from {csv_name}  (label={label})")

        # --- Robust column detection ---
        id_col    = _find_column(df, ["id", "news_id", "index"])
        title_col = _find_column(df, ["title", "text", "statement", "claim", "headline"])

        if title_col is None:
            raise ValueError(
                f"Cannot find a text column in {csv_name}. "
                f"Columns present: {list(df.columns)}"
            )

        for row_idx, row in df.iterrows():
            records.append({
                "id":    str(row[id_col]) if id_col else str(row_idx),
                "title": str(row[title_col]),
                "label": label,
            })

    n_fake = sum(r["label"] == LABEL_FAKE for r in records)
    n_real = sum(r["label"] == LABEL_REAL for r in records)
    log.info(f"Total records: {len(records):,}  (fake={n_fake}, real={n_real})")
    return records


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------

def stratified_split(
    samples: list[PolitiFactSample],
    splits: list[float],
    seed: int,
) -> tuple[list[PolitiFactSample], list[PolitiFactSample], list[PolitiFactSample]]:
    """Split *samples* into train / val / test sets, stratified by label.

    Args:
        samples: Full list of :class:`PolitiFactSample` objects.
        splits:  Three floats ``[train, val, test]`` summing to 1.0.
        seed:    Random seed for reproducibility.

    Returns:
        ``(train_samples, val_samples, test_samples)``

    Raises:
        AssertionError: If *splits* does not sum to 1.0 or has wrong length.
    """
    assert len(splits) == 3, f"Expected 3 split fractions, got {len(splits)}"
    assert abs(sum(splits) - 1.0) < 1e-6, f"Splits must sum to 1.0, got {sum(splits)}"

    train_frac, val_frac, test_frac = splits
    labels = [s.label for s in samples]

    # Step 1: separate train from (val + test)
    train, rest, _, rest_labels = train_test_split(
        samples,
        labels,
        test_size=val_frac + test_frac,
        stratify=labels,
        random_state=seed,
    )

    # Step 2: split (val + test) → val and test
    val_fraction_of_rest = val_frac / (val_frac + test_frac)
    val, test = train_test_split(
        rest,
        test_size=1.0 - val_fraction_of_rest,
        stratify=rest_labels,
        random_state=seed,
    )

    log.info(
        f"Split sizes — train: {len(train):,}, "
        f"val: {len(val):,}, test: {len(test):,}"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PolitiFactDataset(Dataset):
    """``torch.utils.data.Dataset`` wrapping a list of ``PolitiFactSample`` objects.

    Args:
        samples: Pre-built list of :class:`PolitiFactSample`.
    """

    def __init__(self, samples: list[PolitiFactSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PolitiFactSample:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list[PolitiFactSample]) -> dict:
    """Collate a list of ``PolitiFactSample`` objects into a model-ready batch.

    Stacks token tensors along the batch dimension and merges ``HeteroData``
    objects into a single batched ``HeteroData`` using
    ``torch_geometric.data.Batch.from_data_list``.

    Args:
        batch: List of :class:`PolitiFactSample` objects from a DataLoader.

    Returns:
        Dict with keys:

        - ``"input_ids"``       ``(B, seq_len)``  — token IDs
        - ``"attention_mask"``  ``(B, seq_len)``  — attention masks
        - ``"labels"``          ``(B,)``           — integer class labels
        - ``"graph_batch"``     ``HeteroData``     — merged PyG heterogeneous batch
    """
    # tokens were stored as (1, seq_len); squeeze to (seq_len,) then stack → (B, seq_len)
    input_ids      = torch.stack([s.tokens["input_ids"].squeeze(0)      for s in batch])
    attention_mask = torch.stack([s.tokens["attention_mask"].squeeze(0) for s in batch])
    labels         = torch.tensor([s.label for s in batch], dtype=torch.long)
    graph_batch    = Batch.from_data_list([s.graph for s in batch])

    # Dynamic padding trimming: slice away unused trailing padding across batch
    # Eliminates hundreds of empty tokens with 0% data loss, speeding up attention dramatically
    max_len = int(attention_mask.sum(dim=-1).max().item())
    max_len = max(max_len, 8)
    input_ids      = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "graph_batch":    graph_batch,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str, min_length: int = 10) -> list[str]:
    """Naively split *text* into sentences by punctuation or newline boundaries.

    Only sentences at least *min_length* characters long are kept.

    Args:
        text:       Input string.
        min_length: Minimum character count for a sentence to be included.

    Returns:
        List of sentence strings.  Falls back to ``[text]`` if nothing is
        long enough.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [p.strip() for p in parts if len(p.strip()) >= min_length]
    return sentences if sentences else [text]


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def get_loaders(
    cfg: DotDict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Run the full data pipeline and return train / val / test DataLoaders.

    On first call, this function:

    1. Reads raw FakeNewsNet CSVs.
    2. Cleans text with :func:`~src.data.preprocess.clean_text`.
    3. Tokenises with :class:`~src.data.preprocess.Tokenizer`.
    4. Builds k-NN evidence graphs with :class:`~src.data.graph_builder.GraphBuilder`.
    5. Caches all :class:`PolitiFactSample` objects to
       ``cfg.data.processed_dir/politifact_processed.pt``.

    On subsequent calls the cache is reloaded, skipping steps 1-4.

    Args:
        cfg: Dot-accessible config dict (from :func:`~src.utils.config.load_config`).

    Returns:
        ``(train_loader, val_loader, test_loader)``
    """
    cache_path = Path(cfg.data.processed_dir) / "politifact_processed.pt"

    # ------------------------------------------------------------------
    # Load or build processed samples
    # ------------------------------------------------------------------
    if cache_path.exists():
        log.info(f"Loading cached processed samples from {cache_path}")
        samples: list[PolitiFactSample] = torch.load(
            cache_path, map_location="cpu", weights_only=False
        )
        log.info(f"Loaded {len(samples):,} samples from cache.")
    else:
        log.info("Cache not found — running full preprocessing pipeline…")

        # 1. Load raw records
        raw_records = load_fakenewsnet(cfg.data.raw_dir)

        # 2. Initialise preprocessors
        tokenizer = Tokenizer(
            model_name=cfg.model.text_encoder,
            max_length=cfg.data.max_text_length,
        )
        graph_builder = GraphBuilder()

        # 3. Build PolitiFactSample objects
        samples = []
        for i, rec in enumerate(raw_records):
            title = clean_text(rec["title"])
            if not title:
                log.warning(f"Empty title for record id={rec['id']} — skipping.")
                continue

            # Tokenise: returns dict of (1, seq_len) tensors
            tokens = tokenizer.tokenize([title])

            # Build partial M3DUSA HeteroData graph (News + Hashtag)
            graph = graph_builder.build(title)

            samples.append(PolitiFactSample(
                id=rec["id"],
                title=title,
                label=rec["label"],
                tokens=tokens,
                graph=graph,
            ))

            if (i + 1) % 100 == 0:
                log.info(f"  Processed {i + 1:,} / {len(raw_records):,} records…")

        log.info(f"Preprocessing complete: {len(samples):,} valid samples.")

        # 4. Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, cache_path)
        log.info(f"Saved processed samples → {cache_path}")

    # ------------------------------------------------------------------
    # Stratified split
    # ------------------------------------------------------------------
    train_samples, val_samples, test_samples = stratified_split(
        samples,
        splits=list(cfg.data.splits),
        seed=cfg.experiment.seed,
    )

    # ------------------------------------------------------------------
    # Build DataLoaders
    # ------------------------------------------------------------------
    shared_kwargs = dict(
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        num_workers=0,    # CPU-only; increase if you add multiprocessing later
        pin_memory=False, # No CUDA
    )

    train_loader = DataLoader(
        PolitiFactDataset(train_samples), shuffle=True, **shared_kwargs
    )
    val_loader = DataLoader(
        PolitiFactDataset(val_samples), shuffle=False, **shared_kwargs
    )
    test_loader = DataLoader(
        PolitiFactDataset(test_samples), shuffle=False, **shared_kwargs
    )

    log.info(
        f"DataLoaders ready — "
        f"train: {len(train_loader.dataset):,} samples, "
        f"val: {len(val_loader.dataset):,} samples, "
        f"test: {len(test_loader.dataset):,} samples."
    )
    return train_loader, val_loader, test_loader
