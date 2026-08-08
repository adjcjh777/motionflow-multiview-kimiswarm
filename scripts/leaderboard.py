"""Parse all OmniMultiViewFusion training logs and emit a live leaderboard.

Usage:
    python scripts/leaderboard.py --watch 60
    python scripts/leaderboard.py --once --json outputs/leaderboard.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional


def _parse_val_mpjpe(text: str) -> Optional[float]:
    vals = [float(m) for m in re.findall(r"val_MPJPE=([\d.]+)mm", text)]
    if not vals:
        return None
    return min(vals)


def _parse_last_step(text: str) -> Optional[int]:
    matches = re.findall(r"train step (\d+): loss=([\d.]+)", text)
    if not matches:
        return None
    return int(matches[-1][0])


def _parse_last_loss(text: str) -> Optional[float]:
    matches = re.findall(r"train step (\d+): loss=([\d.]+)", text)
    if not matches:
        return None
    return float(matches[-1][1])


def _experiment_name(path: Path) -> str:
    return path.stem


def build_leaderboard(log_dir: Path) -> List[Dict[str, object]]:
    rows = []
    for log in sorted(log_dir.glob("omniview_fusion_*.log")):
        text = log.read_text(errors="ignore")
        rows.append(
            {
                "experiment": _experiment_name(log),
                "best_val_mpjpe_mm": _parse_val_mpjpe(text),
                "last_step": _parse_last_step(text),
                "last_train_loss": _parse_last_loss(text),
                "log": str(log),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Live leaderboard for training runs.")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs"), help="Directory containing .log files")
    parser.add_argument("--json", type=Path, help="Write leaderboard to JSON")
    parser.add_argument("--watch", type=int, default=0, help="Refresh every N seconds (0 = once)")
    args = parser.parse_args()

    while True:
        rows = build_leaderboard(args.log_dir)
        rows_sorted = sorted(rows, key=lambda r: (r["best_val_mpjpe_mm"] is None, r["best_val_mpjpe_mm"] or 1e9))
        if args.json:
            args.json.write_text(json.dumps(rows_sorted, indent=2))
        else:
            print(f"{'Experiment':<40} {'Step':>8} {'TrainLoss':>12} {'BestValMPJPE':>14}")
            print("-" * 80)
            for r in rows_sorted:
                val = f"{r['best_val_mpjpe_mm']:.2f}mm" if r["best_val_mpjpe_mm"] is not None else "---"
                step = r["last_step"] if r["last_step"] is not None else "---"
                loss = f"{r['last_train_loss']:.4f}" if r["last_train_loss"] is not None else "---"
                print(f"{r['experiment']:<40} {step:>8} {loss:>12} {val:>14}")
            print()
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
