#!/usr/bin/env python3
"""Parse v31 A800 logs and print a compact val_MPJPE table."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"


def a800_run(cmd: str) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "a800-D", cmd],
        text=True,
    )


def parse_log(text: str) -> List[Tuple[int, float]]:
    results: List[Tuple[int, float]] = []
    for line in text.splitlines():
        match = re.search(r"Epoch\s+(\d+):\s+.*val_MPJPE=([0-9.]+)mm", line)
        if match:
            results.append((int(match.group(1)), float(match.group(2))))
    return results


def main() -> None:
    logs = a800_run(f"ls {A800_REPO}/outputs/omniview_fusion_v31_*_a800.log 2>/dev/null")
    rows: List[Tuple[str, int, float]] = []
    for path in logs.splitlines():
        if not path.strip():
            continue
        text = a800_run(f"cat {path}")
        vals = parse_log(text)
        name = Path(path).stem
        if not vals:
            rows.append((name, 0, float("nan")))
            continue
        epoch, val = vals[-1]
        rows.append((name, epoch, val))

    print(f"{'run':<60} {'epoch':>6} {'val_MPJPE (mm)':>15}")
    print("-" * 90)
    for name, epoch, val in sorted(rows):
        val_str = f"{val:>15.2f}" if val == val else "no val"
        epoch_str = str(epoch) if epoch > 0 else "-"
        print(f"{name:<60} {epoch_str:>6} {val_str}")


if __name__ == "__main__":
    main()
