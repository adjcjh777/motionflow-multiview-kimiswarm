#!/usr/bin/env python3
"""Parse v29 A800 sweep logs and print a sorted val_MPJPE table.

Usage:
    python scripts/analyze_v29_sweep.py
    python scripts/analyze_v29_sweep.py --remote  # fetch from a800-D via ssh
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple


def fetch_remote() -> str:
    cmd = (
        r"ssh a800-D 'cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20 "
        r"&& for f in outputs/omniview_fusion_v29*_a800.log; do "
        r"echo --- $f; grep val_MPJPE= ${f} || true; done'"
    )
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def parse_text(text: str) -> List[Tuple[str, float, int]]:
    """Return list of (run_name, val_mpjpe, epoch)."""
    results: List[Tuple[str, float, int]] = []
    current_file = None
    for line in text.splitlines():
        if line.startswith("--- outputs/"):
            current_file = line.split("/")[-1].replace(".log", "")
            continue
        match = re.search(r"Epoch\s+(\d+):\s+.*val_MPJPE=([0-9.]+)mm", line)
        if match and current_file:
            epoch = int(match.group(1))
            val = float(match.group(2))
            results.append((current_file, val, epoch))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v29 sweep val_MPJPE")
    parser.add_argument("--remote", action="store_true", help="Fetch logs from a800-D")
    args = parser.parse_args()

    if args.remote:
        text = fetch_remote()
    else:
        outputs = Path("outputs")
        text = ""
        for log in outputs.glob("omniview_fusion_v29*_a800.log"):
            text += f"--- {log}\n"
            text += log.read_text(errors="ignore")

    results = parse_text(text)
    # Keep only the latest epoch per run.
    best: dict[str, Tuple[float, int]] = {}
    for name, val, epoch in results:
        if name not in best or epoch > best[name][1]:
            best[name] = (val, epoch)

    sorted_results = sorted(best.items(), key=lambda x: x[1][0])
    print(f"{'run':<50} {'epoch':>8} {'val_MPJPE (mm)':>15}")
    print("-" * 80)
    for name, (val, epoch) in sorted_results:
        print(f"{name:<50} {epoch:>8} {val:>15.2f}")


if __name__ == "__main__":
    main()
