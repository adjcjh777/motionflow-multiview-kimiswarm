"""Smoke test for the v51 TTSER evaluation script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_eval_v51_tta_smoke() -> None:
    result = subprocess.run(
        [sys.executable, "experiments/eval_v51_test_time_self_evolution.py", "--smoke"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "baseline_no_tta" in result.stdout
    assert "v51_tta" in result.stdout
