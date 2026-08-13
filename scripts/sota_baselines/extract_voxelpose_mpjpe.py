#!/usr/bin/env python
"""Extract the final/best MPJPE from a VoxelPose log and write a JSON summary.

VoxelPose logs lines like::

    MPJPE: 31.41mm

This script collects all such lines, reports the minimum (best) and last values,
and writes a JSON result file compatible with the project's baseline outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


def find_mpjpe_values(log_text: str) -> List[float]:
    """Return all MPJPE values (mm) found in *log_text*, in order."""
    # Match lines such as "MPJPE: 31.41mm" or "MPJPE: 31.41 mm"
    pattern = re.compile(r"MPJPE:\s*([0-9]+\.[0-9]+)\s*mm", re.IGNORECASE)
    values = []
    for match in pattern.finditer(log_text):
        values.append(float(match.group(1)))
    return values


def build_json_result(
    values: List[float],
    run_config: Path,
    output_json: Path,
    checkpoint_path: Optional[Path] = None,
) -> dict:
    best = min(values) if values else None
    last = values[-1] if values else None

    return {
        "method": "VoxelPose",
        "dataset": "h36m_true_gt_v2",
        "protocol": "S1,5,6,7,8 -> S9/S11",
        "unit": "mm",
        "mpjpe_mm": best,
        "mpjpe_m": best / 1000.0 if best is not None else None,
        "mpjpe_last_mm": last,
        "mpjpe_last_m": last / 1000.0 if last is not None else None,
        "run_config": str(run_config.resolve()),
        "checkpoint": str(checkpoint_path.resolve()) if checkpoint_path else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Path to VoxelPose log file.")
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("configs/sota_baselines/voxelpose_h36m_true_gt_v2.yaml"),
        help="Path to the VoxelPose run config (recorded in output JSON).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the trained checkpoint (recorded in output JSON).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"ERROR: log file not found: {args.log}", file=sys.stderr)
        return 1

    log_text = args.log.read_text(encoding="utf-8", errors="ignore")
    values = find_mpjpe_values(log_text)

    if not values:
        print(f"ERROR: no MPJPE values found in {args.log}", file=sys.stderr)
        return 1

    result = build_json_result(
        values,
        run_config=args.run_config,
        output_json=args.output,
        checkpoint_path=args.checkpoint,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted MPJPE values: best={min(values):.2f}mm, last={values[-1]:.2f}mm")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
