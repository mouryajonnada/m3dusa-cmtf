"""
Split Large CSV into N Parts
============================
Splits large CSV files (such as NEW_DATA/news_cleaned_2018_02_13.csv)
into N parts with balanced size, maintaining RFC 4180 multiline quote integrity
and including the original CSV header in every split part.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def split_csv(
    input_path: str | Path,
    num_parts: int = 5,
    output_dir: str | Path | None = None,
    buffer_size: int = 16 * 1024 * 1024,  # 16 MB
) -> list[Path]:
    input_file = Path(input_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Source file not found: {input_file}")

    out_dir = Path(output_dir).resolve() if output_dir else input_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = input_file.stat().st_size
    print(f"Source CSV: {input_file}")
    print(f"Total Size: {total_bytes / (1024**3):.2f} GB ({total_bytes:,} bytes)")
    print(f"Splitting into {num_parts} parts...")

    stem = input_file.stem
    ext = input_file.suffix

    output_paths = [
        out_dir / f"{stem}_part{i + 1}{ext}" for i in range(num_parts)
    ]

    target_bytes_per_part = total_bytes // num_parts

    t_start = time.time()
    last_print_time = t_start

    current_part = 1
    current_part_bytes = 0
    current_part_records = 0
    total_records = 0
    total_bytes_processed = 0

    in_quotes = False

    with open(input_file, "rb") as fin:
        header = fin.readline()
        header_len = len(header)
        total_bytes_processed += header_len

        fout = open(output_paths[0], "wb", buffering=buffer_size)
        fout.write(header)
        current_part_bytes += header_len

        print(f"\n--- Writing Part 1: {output_paths[0].name} ---")

        while True:
            line = fin.readline()
            if not line:
                break

            line_len = len(line)
            total_bytes_processed += line_len
            current_part_bytes += line_len
            fout.write(line)

            # RFC 4180 quote parity tracking
            q = line.count(b'"')
            if q % 2 != 0:
                in_quotes = not in_quotes

            if not in_quotes:
                current_part_records += 1
                total_records += 1

                # Check if we should advance to the next part
                if (
                    current_part < num_parts
                    and current_part_bytes >= target_bytes_per_part
                ):
                    fout.close()
                    elapsed_part = time.time() - t_start
                    print(
                        f"Completed Part {current_part}: "
                        f"{current_part_bytes / (1024**3):.2f} GB, "
                        f"{current_part_records:,} records."
                    )

                    current_part += 1
                    current_part_bytes = 0
                    current_part_records = 0

                    print(f"\n--- Writing Part {current_part}: {output_paths[current_part - 1].name} ---")
                    fout = open(
                        output_paths[current_part - 1], "wb", buffering=buffer_size
                    )
                    fout.write(header)
                    current_part_bytes += header_len

            # Progress update every 5 seconds
            now = time.time()
            if now - last_print_time >= 5.0:
                last_print_time = now
                pct = (total_bytes_processed / total_bytes) * 100
                speed_mb = (total_bytes_processed / (1024**2)) / (now - t_start)
                remaining_bytes = max(0, total_bytes - total_bytes_processed)
                eta_s = (remaining_bytes / (1024**2)) / speed_mb if speed_mb > 0 else 0
                print(
                    f"Progress: {pct:5.1f}% | "
                    f"Processed: {total_bytes_processed / (1024**3):.2f}/{total_bytes / (1024**3):.2f} GB | "
                    f"Speed: {speed_mb:.1f} MB/s | "
                    f"ETA: {eta_s:.0f}s | "
                    f"Part {current_part}/{num_parts}"
                )

        fout.close()
        print(
            f"Completed Part {current_part}: "
            f"{current_part_bytes / (1024**3):.2f} GB, "
            f"{current_part_records:,} records."
        )

    t_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total time elapsed: {t_total:.1f}s ({t_total / 60:.1f} minutes)")
    print(f"Total records processed: {total_records:,}")
    print("\nGenerated files:")
    for i, path in enumerate(output_paths, 1):
        size_gb = path.stat().st_size / (1024**3)
        print(f"  Part {i}: {path.name} ({size_gb:.2f} GB, {path.stat().st_size:,} bytes)")

    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CSV into N parts.")
    parser.add_argument(
        "--input",
        "-i",
        default=r"NEW_DATA/news_cleaned_2018_02_13.csv",
        help="Path to source CSV file",
    )
    parser.add_argument(
        "--parts",
        "-n",
        type=int,
        default=5,
        help="Number of parts to split into (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory (default: same as input file)",
    )

    args = parser.parse_args()
    split_csv(
        input_path=args.input,
        num_parts=args.parts,
        output_dir=args.output_dir,
    )
