#!/usr/bin/env python3
"""Plot training loss curves from A800-D logs.

Reads the ``outputs/omniview_fusion_v*.log`` files on A800-D, extracts the
``train step X: loss=Y`` lines, and produces a loss-vs-step PNG for quick
visual comparison of concurrent runs.

Examples
--------
    python scripts/plot_a800_training_curves.py \
        --runs v25_geometry_fusion_small v25_geometry_fusion_full \
        --output training_curves.png
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"


def fetch_log(log_name: str) -> str:
    """Return the contents of an A800 outputs/ log file."""
    return subprocess.check_output(
        ["ssh", "a800-D", f"cat {A800_REPO}/outputs/{log_name}"],
        text=True,
        stderr=subprocess.STDOUT,
    )


def parse_loss(text: str) -> list[tuple[int, float]]:
    """Parse (step, loss) tuples from a training log."""
    out: list[tuple[int, float]] = []
    for line in text.splitlines():
        match = re.search(r"train step\s+(\d+):\s*loss=([\d.]+)", line)
        if match:
            out.append((int(match.group(1)), float(match.group(2))))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot A800 training curves.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Log file names (without path) to plot.",
    )
    parser.add_argument("--output", default="a800_training_curves.png", help="Output PNG path.")
    args = parser.parse_args(argv)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        raise SystemExit(1) from exc

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in args.runs:
        try:
            text = fetch_log(name)
        except subprocess.CalledProcessError:
            print(f"Warning: could not fetch {name}", file=sys.stderr)
            continue
        data = parse_loss(text)
        if not data:
            print(f"Warning: no train step data in {name}", file=sys.stderr)
            continue
        steps, losses = zip(*data)
        ax.plot(steps, losses, label=name, alpha=0.8)

    ax.set_xlabel("Train Step")
    ax.set_ylabel("Loss")
    ax.set_title("A800 Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
