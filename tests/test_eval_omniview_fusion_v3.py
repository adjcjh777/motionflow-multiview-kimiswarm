"""Smoke tests for experiments/eval_omniview_fusion_v3_mpiinf3dhp.py"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_eval_v3_smoke() -> None:
    out_json = REPO_ROOT / "outputs" / "test_eval_omniview_fusion_v3_smoke.json"
    out_csv = REPO_ROOT / "outputs" / "test_eval_omniview_fusion_v3_smoke.csv"
    out_json.unlink(missing_ok=True)
    out_csv.unlink(missing_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "eval_omniview_fusion_v3_mpiinf3dhp.py"),
            "--smoke",
            f"--out_json={out_json}",
            f"--out_csv={out_csv}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_json.exists()
    assert out_csv.exists()

    data = json.loads(out_json.read_text())
    assert "clean" in data
    assert "mpjpe" in data["clean"]
    assert "pa_mpjpe" in data["clean"]
    assert "robustness" in data
    assert "variable_views" in data
