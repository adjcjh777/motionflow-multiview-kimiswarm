"""CPU smoke tests for the v4 robustness matrix harness."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "robustness_matrix_v4.py"


def _run_smoke(model: str, tmp_path: Path) -> Path:
    out_dir = tmp_path / f"robustness_matrix_v4_{model}"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--smoke",
        "--model",
        model,
        "--device",
        "cpu",
        "--output_dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return out_dir


@pytest.mark.parametrize("model", ["v4", "v3"])
def test_smoke_outputs(model: str, tmp_path: Path) -> None:
    """Smoke test should produce JSON and CSV with the expected columns/rows."""
    out_dir = _run_smoke(model, tmp_path)
    json_path = out_dir / "robustness_matrix_v4.json"
    csv_path = out_dir / "robustness_matrix_v4.csv"

    assert json_path.exists()
    assert csv_path.exists()

    data = json.loads(json_path.read_text())
    assert "clean" in data
    assert "robustness" in data
    assert "per_joint" in data
    assert "variable_views" in data

    # CSV should have a markdown-compatible robustness table section.
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    robustness_rows = [r for r in rows if r["section"] == "robustness"]
    assert len(robustness_rows) > 0
    expected_columns = {
        "condition",
        "mpjpe",
        "pa_mpjpe",
        "pck@50mm",
        "pck@100mm",
        "pck@150mm",
        "pck_auc",
    }
    assert expected_columns.issubset(set(rows[0].keys()))
    for r in robustness_rows:
        assert r["condition"]
        assert float(r["mpjpe"]) >= 0.0
        assert float(r["pa_mpjpe"]) >= 0.0

    variable_rows = [r for r in rows if r["section"] == "variable_views"]
    assert len(variable_rows) >= 2
    for r in variable_rows:
        assert r["mpjpe"]
        assert r["pa_mpjpe"]

    per_joint_rows = [r for r in rows if r["section"] == "per_joint"]
    assert len(per_joint_rows) > 0
