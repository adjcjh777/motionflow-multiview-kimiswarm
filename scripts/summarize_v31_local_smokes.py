#!/usr/bin/env python3
"""Parse local v30/v31 smoke logs and print a compact val_MPJPE table."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_log(path: Path) -> List[Tuple[int, float]]:
    """Return list of (epoch, val_mpjpe) from a log file."""
    results: List[Tuple[int, float]] = []
    text = path.read_text(errors="ignore")
    for line in text.splitlines():
        match = re.search(r"Epoch\s+(\d+):\s+.*val_MPJPE=([0-9.]+)mm", line)
        if match:
            results.append((int(match.group(1)), float(match.group(2))))
    return results


def main() -> None:
    outputs = Path("outputs")
    # v30 baseline with val_stride=1; v31/v32 smokes.
    patterns = [
        "omniview_fusion_v30_smoke_local_4090_val1.log",
        "omniview_fusion_v31_*_smoke_local_4090.log",
        "omniview_fusion_v32_*_smoke_local_4090.log",
    ]
    logs: Dict[str, Path] = {}
    for pat in patterns:
        for log in outputs.glob(pat):
            logs[log.stem] = log

    rows: List[Tuple[str, int, float]] = []
    for name, path in sorted(logs.items()):
        vals = parse_log(path)
        if not vals:
            rows.append((name, 0, float("nan")))
            continue
        # Use latest epoch per run.
        epoch, val = vals[-1]
        rows.append((name, epoch, val))

    print(f"{'run':<60} {'epoch':>6} {'val_MPJPE (mm)':>15}")
    print("-" * 90)
    for name, epoch, val in rows:
        val_str = f"{val:>15.2f}" if not (val != val) else "no val"
        epoch_str = str(epoch) if epoch > 0 else "-"
        print(f"{name:<60} {epoch_str:>6} {val_str}")


if __name__ == "__main__":
    main()
