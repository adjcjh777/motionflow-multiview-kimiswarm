#!/usr/bin/env python
"""Aggregate training logs into a leaderboard CSV.

Scans a directory for ``omniview_fusion_*.log`` files, extracts the best
validation MPJPE for each run, and writes a sorted CSV.  Useful for quickly
comparing v25/v26/UDP/GMM variants.

Usage
-----
    python scripts/aggregate_run_results.py
    python scripts/aggregate_run_results.py --log-dir outputs --out results/leaderboard.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional


def parse_log(log_path: Path) -> Optional[Dict[str, object]]:
    """Return a dict with run name, best val_MPJPE, and final epoch count."""
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    # Find all val_MPJPE lines.
    pattern = re.compile(r"Epoch\s+(\d+):\s+train_loss=[\d.]+,\s+val_loss=[\d.]+,\s+val_MPJPE=([\d.]+)mm")
    matches = pattern.findall(text)
    if not matches:
        return None

    best_epoch, best_mpjpe = min(matches, key=lambda m: float(m[1]))
    best_mpjpe = float(best_mpjpe)
    last_epoch = int(matches[-1][0])

    # Also try to extract the run variant from the log name.
    name = log_path.stem.replace("omniview_fusion_", "")

    return {
        "run": name,
        "log": log_path.name,
        "last_epoch": last_epoch,
        "best_epoch": int(best_epoch),
        "best_val_mpjpe_mm": best_mpjpe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate training logs into a leaderboard CSV")
    parser.add_argument("--log-dir", type=str, default="outputs", help="Directory containing omniview_fusion_*.log files")
    parser.add_argument("--out", type=str, default="outputs/leaderboard.csv", help="Output CSV path")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return

    logs = sorted(log_dir.glob("omniview_fusion_*.log"))
    if not logs:
        print(f"No omniview_fusion_*.log files found in {log_dir}")
        return

    rows: List[Dict[str, object]] = []
    for log in logs:
        parsed = parse_log(log)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        print("No val_MPJPE entries found in any log.")
        return

    rows.sort(key=lambda r: r["best_val_mpjpe_mm"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "best_val_mpjpe_mm", "best_epoch", "last_epoch", "log"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Leaderboard ({len(rows)} runs) -> {out_path}")
    for row in rows:
        print(f"  {row['best_val_mpjpe_mm']:.2f} mm  epoch={row['best_epoch']}  {row['run']}")


if __name__ == "__main__":
    main()
