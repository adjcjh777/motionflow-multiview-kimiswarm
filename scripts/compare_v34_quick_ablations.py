#!/usr/bin/env python3
"""Compare v34 VJGN vs geometry-aware VJGN quick ablations."""

from __future__ import annotations

import re
from pathlib import Path


LOGS = [
    ("v34_vjgn", Path("outputs/v34_vjgn_quick_ablation_local_4090.log")),
    ("v34_gvjgn", Path("outputs/v34_gvjgn_quick_ablation_local_4090.log")),
]


def _extract_epochs(text: str) -> list[dict]:
    rows = []
    # "Epoch N: train_loss=..., val_loss=..., val_MPJPE=XX.XXmm"
    for m in re.finditer(
        r"Epoch\s+(\d+):\s+train_loss=([0-9.eE+-]+),\s+val_loss=([0-9.eE+-]+),\s+val_MPJPE=([0-9.]+)mm",
        text,
    ):
        rows.append(
            {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss": float(m.group(3)),
                "val_mpjpe_mm": float(m.group(4)),
            }
        )
    return rows


def main() -> None:
    print("| Run | Epoch | train_loss | val_loss | val_MPJPE (mm) |")
    print("| --- | ----- | --- | --- | --- |")
    for name, path in LOGS:
        if not path.exists():
            print(f"| {name} | log not found | - | - | - |")
            continue
        text = path.read_text(errors="ignore")
        rows = _extract_epochs(text)
        if not rows:
            print(f"| {name} | no val yet | - | - | - |")
            continue
        for row in rows:
            print(
                f"| {name} | {row['epoch']} | {row['train_loss']:.6f} | "
                f"{row['val_loss']:.6f} | {row['val_mpjpe_mm']:.2f} |"
            )
        best = min(rows, key=lambda r: r["val_mpjpe_mm"])
        print(f"| {name} | best | - | - | {best['val_mpjpe_mm']:.2f}mm (epoch {best['epoch']}) |")


if __name__ == "__main__":
    main()
