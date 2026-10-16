"""
Prepare New Data (FakeNewsCorpus)
=================================
Extracts a balanced subset (1,000 fake and 1,000 reliable articles)
from NEW_DATA/news_cleaned_2018_02_13.csv and saves them to
data/raw/fakenewscorpus_fake.csv and data/raw/fakenewscorpus_real.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
import pandas as pd

# Allow large fields in CSV
csv.field_size_limit(min(2147483647, sys.maxsize))

def extract_fakenewscorpus(
    source_csv: str | Path,
    out_dir: str | Path,
    target_per_class: int = 1000,
) -> tuple[Path, Path]:
    source_path = Path(source_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fake_out = out_dir / "fakenewscorpus_fake.csv"
    real_out = out_dir / "fakenewscorpus_real.csv"

    if fake_out.exists() and real_out.exists():
        print(f"Extraction already exists:\n  {fake_out}\n  {real_out}")
        return fake_out, real_out

    print(f"Reading from {source_path}...")
    print(f"Target: {target_per_class} fake and {target_per_class} reliable articles...")

    fake_records = []
    real_records = []

    t0 = time.time()
    with open(source_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            news_type = (row.get("type") or "").strip().lower()
            title = (row.get("title") or "").strip()
            
            # Filter out empty or too-short titles
            if len(title) < 8 or title.lower() in ("none", "null", "untitled"):
                continue

            rec = {
                "id": row.get("id") or str(i),
                "title": title,
                "url": row.get("url", ""),
                "type": news_type,
            }

            if news_type == "fake" and len(fake_records) < target_per_class:
                fake_records.append(rec)
            elif news_type == "reliable" and len(real_records) < target_per_class:
                real_records.append(rec)

            if len(fake_records) >= target_per_class and len(real_records) >= target_per_class:
                break

            if (i + 1) % 50000 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {i + 1:,} rows: {len(fake_records):,} fake, {len(real_records):,} reliable ({elapsed:.1f}s)")

    print(f"Extracted {len(fake_records)} fake, {len(real_records)} reliable in {time.time() - t0:.2f}s")

    # Save to CSV
    pd.DataFrame(fake_records).to_csv(fake_out, index=False)
    pd.DataFrame(real_records).to_csv(real_out, index=False)
    print(f"Saved:\n  {fake_out}\n  {real_out}")
    return fake_out, real_out

if __name__ == "__main__":
    extract_fakenewscorpus(
        source_csv="NEW_DATA/news_cleaned_2018_02_13.csv",
        out_dir="data/raw",
        target_per_class=1000,
    )
