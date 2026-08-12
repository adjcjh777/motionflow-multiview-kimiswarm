#!/usr/bin/env python3
"""Audit all generated H36M true-GT v2 .npz files for circularity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    npz_dir = Path("data/h36m_true_gt_v2")
    if not npz_dir.exists():
        print(f"Directory not found: {npz_dir}")
        sys.exit(1)

    files = sorted(npz_dir.glob("*_multiview_m.npz"))
    if not files:
        print("No *_multiview_m.npz files found.")
        sys.exit(1)

    for path in files:
        print(f"Auditing {path.name} ...")
        subprocess.run([sys.executable, "scripts/diagnose_circular_labels.py", str(path)], check=True)
        print()


if __name__ == "__main__":
    main()
