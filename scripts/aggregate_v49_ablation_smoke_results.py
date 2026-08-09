#!/usr/bin/env python3
"""Aggregate val_MPJPE from the v49 ablation matrix smoke runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.scrape_val_mpjpe import scrape_log


VARIANTS = ["v45_only", "v46_on_v45", "v47_on_v46", "v48_on_v47", "v49_lite_on_v46"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate v49 ablation smoke results.")
    parser.add_argument("--csv", action="store_true", help="Output CSV instead of Markdown.")
    args = parser.parse_args()

    rows = {}
    for name in VARIANTS:
        log_path = f"outputs/v49_ablation_{name}.log"
        try:
            log_rows = scrape_log(log_path)
        except FileNotFoundError:
            log_rows = []
        rows[name] = {r["epoch"]: r["val_mpjpe_mm"] for r in log_rows}

    epochs = sorted({e for r in rows.values() for e in r})
    if not epochs:
        print("No validation results found yet.")
        return

    if args.csv:
        import csv

        writer = csv.writer(sys.stdout)
        writer.writerow(["epoch"] + VARIANTS)
        for e in epochs:
            writer.writerow([e] + [f"{rows[n].get(e, float('nan')):.2f}" for n in VARIANTS])
    else:
        print("| epoch | " + " | ".join(VARIANTS) + " |")
        print("|-------|" + "|".join(["------"] * len(VARIANTS)) + "|")
        for e in epochs:
            vals = [f"{rows[n].get(e, float('nan')):>7.2f}" for n in VARIANTS]
            print(f"| {e:5d} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
