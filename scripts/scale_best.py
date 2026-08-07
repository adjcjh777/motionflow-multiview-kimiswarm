"""Pick the best finished baseline and launch its full-scale run.

Usage:
    python scripts/scale_best.py outputs/omniview_fusion_v10_aleatoric_outlier.log ...

The script parses each log, looks for the best val MPJPE, maps the experiment
name to the corresponding *_fullscale.sh launch script, and starts it in a
tmux session on the requested GPU.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _best_val_mpjpe(text: str) -> Optional[float]:
    """Return the best (minimum) validation MPJPE in mm from the log."""
    vals = [float(m) for m in re.findall(r"val_MPJPE=([\d.]+)mm", text)]
    if not vals:
        return None
    return min(vals)


def _last_train_step(text: str) -> Optional[int]:
    matches = re.findall(r"train step (\d+):", text)
    if matches:
        return int(matches[-1])
    return None


def _map_to_fullscale(experiment: str) -> Optional[Path]:
    """Map an experiment name to its full-scale launch script."""
    script_dir = Path(__file__).parent
    # Strip the "omniview_fusion_" prefix if present.
    if experiment.startswith("omniview_fusion_"):
        experiment = experiment[len("omniview_fusion_")]
    candidates = [
        script_dir / f"run_{experiment}_fullscale.sh",
        script_dir / f"run_{experiment}.sh",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def summarize_log(path: Path) -> Dict[str, Optional[float]]:
    text = path.read_text(errors="ignore")
    return {
        "experiment": path.stem,
        "best_val_mpjpe_mm": _best_val_mpjpe(text),
        "last_train_step": _last_train_step(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch full-scale run for the best baseline.")
    parser.add_argument("logs", type=Path, nargs="+", help="Baseline log files to compare")
    parser.add_argument("--gpu", type=int, default=4, help="GPU index for the full-scale run")
    parser.add_argument("--tmux_session", type=str, default="best_fullscale", help="tmux session name")
    parser.add_argument("--dry_run", action="store_true", help="Only print what would be launched")
    args = parser.parse_args()

    rows = [summarize_log(p) for p in args.logs]
    valid_rows = [r for r in rows if r["best_val_mpjpe_mm"] is not None]

    if not valid_rows:
        print("No validation MPJPE available yet; waiting for first epoch to finish.")
        sys.exit(0)

    best = min(valid_rows, key=lambda r: r["best_val_mpjpe_mm"])  # type: ignore
    experiment = best["experiment"]
    script = _map_to_fullscale(experiment)
    if script is None:
        print(f"Could not find a launch script for {experiment}")
        sys.exit(1)

    print(f"Best baseline: {experiment} (val MPJPE = {best['best_val_mpjpe_mm']:.2f}mm)")
    print(f"Launching: CUDA_VISIBLE_DEVICES={args.gpu} bash {script}")

    if args.dry_run:
        print("Dry run; not starting tmux session.")
        sys.exit(0)

    cmd = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        args.tmux_session,
        "-n",
        "main",
        f"source .venv/bin/activate && CUDA_VISIBLE_DEVICES={args.gpu} bash {script}",
    ]
    subprocess.run(cmd, check=True)
    print(f"Started tmux session '{args.tmux_session}' on GPU {args.gpu}")


if __name__ == "__main__":
    main()
