#!/usr/bin/env python3
"""Scrape val_MPJPE / val_loss per epoch from a training log.

Usage
-----
    python scripts/scrape_val_mpjpe.py outputs/v46_svg_smoke_local_4090.log

Output is a CSV with columns: epoch, val_loss, val_mpjpe_mm.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def scrape_log(log_path: str) -> list[dict[str, float]]:
    """Return a list of rows extracted from a training log."""
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    rows: list[dict[str, float]] = []

    # Matches lines like:
    #   Epoch 1: train_loss=5.849611, val_loss=0.029057, val_MPJPE=32.77mm
    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+):\s+"
        r"train_loss=(?P<train_loss>[\d.]+),\s+"
        r"val_loss=(?P<val_loss>[\d.]+),\s+"
        r"val_MPJPE=(?P<val_mpjpe>[\d.]+)mm"
    )

    for match in pattern.finditer(text):
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "val_loss": float(match.group("val_loss")),
                "val_mpjpe_mm": float(match.group("val_mpjpe")),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape val_MPJPE from a training log")
    parser.add_argument("log_path", help="Path to the training log file")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    args = parser.parse_args()

    rows = scrape_log(args.log_path)
    if not rows:
        print(f"No epoch summary lines found in {args.log_path}", file=sys.stderr)
        return 1

    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=["epoch", "val_loss", "val_mpjpe_mm"])
        writer.writeheader()
        writer.writerows(rows)
    else:
        for row in rows:
            print(f"Epoch {row['epoch']}: val_loss={row['val_loss']:.6f}, val_MPJPE={row['val_mpjpe_mm']:.2f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
