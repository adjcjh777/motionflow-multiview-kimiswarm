"""Smoke test for the OmniMultiViewFusionV2 evaluation script.

Runs the evaluation script in smoke mode on synthetic data and checks that the
produced JSON and CSV outputs are well-formed.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _run_smoke(tmp_path: Path) -> Path:
    out_json = tmp_path / "eval_smoke.json"
    out_csv = tmp_path / "eval_smoke.csv"
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "eval_omniview_fusion_v2_mpiinf3dhp.py"),
        "--smoke",
        "--seed", "42",
        "--out_json", str(out_json),
        "--out_csv", str(out_csv),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return out_json, out_csv


def test_eval_omniview_fusion_v2_smoke(tmp_path: Path):
    out_json, out_csv = _run_smoke(tmp_path)

    assert out_json.exists()
    assert out_csv.exists()

    with open(out_json) as f:
        results = json.load(f)

    assert "clean" in results
    assert "mpjpe" in results["clean"]
    assert "pa_mpjpe" in results["clean"]
    assert results["clean"]["mpjpe"] < float("inf")
    assert results["clean"]["pa_mpjpe"] < float("inf")

    assert "robustness" in results
    assert "rot_0.5_deg" in results["robustness"]
    assert "mpjpe" in results["robustness"]["rot_0.5_deg"]

    assert "variable_views" in results
    assert "2" in results["variable_views"]
    assert "mean_mm" in results["variable_views"]["2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
