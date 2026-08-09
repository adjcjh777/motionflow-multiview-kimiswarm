#!/usr/bin/env python3
"""Aggregate val_MPJPE from the v46/v47/v48 local smoke chain.

After the smoke chain in `scripts/run_v46_v47_v48_smoke_chain_local_4090.sh`
finishes, run this script to produce a Markdown table of epoch-1/epoch-2
validation MPJPE for each variant.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.scrape_val_mpjpe import scrape_log


LOG_FILES = {
    "v46": "outputs/v46_svg_smoke_local_4090.log",
    "v47": "outputs/v47_temporal_svg_smoke_local_4090.log",
    "v48": "outputs/v48_domain_smoke_local_4090.log",
}


def _scrape_to_dict(log_path: str) -> dict:
    """Return {epoch: mpjpe} from a log file using the scraper."""
    if not Path(log_path).exists():
        return {}
    try:
        rows = scrape_log(log_path)
    except Exception:  # pylint: disable=broad-except
        return {}
    return {int(row["epoch"]): float(row["val_mpjpe_mm"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate v46/v47/v48 smoke results.")
    parser.add_argument("--csv", action="store_true", help="Output CSV instead of Markdown.")
    args = parser.parse_args()

    results = {name: _scrape_to_dict(path) for name, path in LOG_FILES.items()}
    epochs = sorted({e for r in results.values() for e in r})

    if not epochs:
        print("No validation results found yet.")
        return

    if args.csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(["epoch"] + list(LOG_FILES.keys()))
        for e in epochs:
            writer.writerow([e] + [f"{results[n].get(e, float('nan')):.2f}" for n in LOG_FILES])
    else:
        print("| epoch | v46 (mm) | v47 (mm) | v48 (mm) |")
        print("|-------|----------|----------|----------|")
        for e in epochs:
            row = [f"{results[n].get(e, float('nan')):.2f}" for n in LOG_FILES]
            print(f"| {e:5d} | {row[0]:>8s} | {row[1]:>8s} | {row[2]:>8s} |")


if __name__ == "__main__":
    main()
