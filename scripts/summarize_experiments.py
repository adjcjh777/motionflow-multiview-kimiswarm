"""Parse experiment logs and produce a Markdown summary table.

Usage:
    python scripts/summarize_experiments.py outputs/*.log

Outputs a table with experiment name, model params (if available),
last train loss, best/last val MPJPE, and whether a checkpoint/config exists.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _extract_last_train_loss(text: str) -> Optional[float]:
    matches = re.findall(r"train step \d+: loss=([\d.]+)", text)
    if matches:
        return float(matches[-1])
    return None


def _extract_best_val_mpjpe(text: str) -> Optional[float]:
    matches = re.findall(r"Best val MPJPE: ([\d.]+)mm", text)
    if matches:
        return float(matches[-1])
    return None


def _extract_last_val_mpjpe(text: str) -> Optional[float]:
    matches = re.findall(r"val_MPJPE=([\d.]+)mm", text)
    if matches:
        return float(matches[-1])
    return None


def _extract_model_params(text: str) -> Optional[int]:
    m = re.search(r"Model params: (\d+)", text)
    if m:
        return int(m.group(1))
    return None


def summarize_log(path: Path) -> Dict[str, Any]:
    text = path.read_text(errors="ignore")
    name = path.stem
    checkpoint = path.with_suffix(".pth")
    config = path.with_suffix(".config.json")
    return {
        "experiment": name,
        "params": _extract_model_params(text),
        "last_train_loss": _extract_last_train_loss(text),
        "best_val_mpjpe_mm": _extract_best_val_mpjpe(text),
        "last_val_mpjpe_mm": _extract_last_val_mpjpe(text),
        "checkpoint_exists": checkpoint.exists(),
        "config_exists": config.exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment logs")
    parser.add_argument("logs", type=Path, nargs="+", help="Log files to summarize")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of Markdown")
    args = parser.parse_args()

    rows = [summarize_log(p) for p in args.logs]

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    key_to_header = {
        "experiment": "Experiment",
        "params": "Params",
        "last_train_loss": "Last Train Loss",
        "best_val_mpjpe_mm": "Best Val MPJPE (mm)",
        "checkpoint_exists": "Checkpoint",
        "config_exists": "Config",
    }
    keys = list(key_to_header.keys())
    headers = list(key_to_header.values())
    col_widths = [
        max(len(key_to_header[k]), max((len(str(r[k])) for r in rows), default=0))
        for k in keys
    ]

    def fmt(row: Dict[str, Any]) -> List[str]:
        val_mpjpe = (
            row["best_val_mpjpe_mm"]
            if row["best_val_mpjpe_mm"] is not None
            else row["last_val_mpjpe_mm"]
        )
        return [
            str(row["experiment"]),
            str(row["params"] if row["params"] is not None else ""),
            f"{row['last_train_loss']:.4f}" if row["last_train_loss"] is not None else "",
            f"{val_mpjpe:.2f}" if val_mpjpe is not None else "",
            "yes" if row["checkpoint_exists"] else "no",
            "yes" if row["config_exists"] else "no",
        ]

    def line(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    print(line(headers))
    print(line(["-" * w for w in col_widths]))
    for row in rows:
        print(line(fmt(row)))


if __name__ == "__main__":
    main()
