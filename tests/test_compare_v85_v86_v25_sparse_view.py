"""Tests for scripts/compare_v85_v86_v25_sparse_view.py.

Covers JSON parsing, missing-file handling, table generation, and the produced
PNG/JSON/Markdown artifacts.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parent.parent / "scripts" / "compare_v85_v86_v25_sparse_view.py"


def _make_json(path: Path, values_s9: dict, values_s11: dict) -> None:
    """Write a synthetic variable-view JSON in the per_dataset format."""
    data = {"per_dataset": {"S9": {}, "S11": {}}}
    for k, val in values_s9.items():
        data["per_dataset"]["S9"][str(k)] = {"mpjpe_at_k": val}
    for k, val in values_s11.items():
        data["per_dataset"]["S11"][str(k)] = {"mpjpe_at_k": val}
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def synthetic_inputs(tmp_path: Path):
    v85 = tmp_path / "v85_no_fallback.json"
    v85_dlt = tmp_path / "v85_dlt_fallback.json"
    v25 = tmp_path / "v25_dlt_fallback.json"
    v86 = tmp_path / "v86.json"

    _make_json(v85, {2: 2310.27, 3: 1119.45, 4: 83.52}, {2: 2308.80, 3: 1118.18, 4: 77.07})
    _make_json(v85_dlt, {2: 60.0, 3: 35.0, 4: 85.0}, {2: 50.0, 3: 28.0, 4: 80.0})
    _make_json(v86, {2: 2200.0, 3: 1000.0, 4: 70.0}, {2: 2100.0, 3: 950.0, 4: 65.0})
    _make_json(v25, {2: 58.18, 3: 33.32, 4: 116.98}, {2: 49.35, 3: 25.28, 4: 110.58})

    return v85, v85_dlt, v86, v25


def test_script_creates_all_artifacts(synthetic_inputs, tmp_path: Path):
    v85, v85_dlt, v86, v25 = synthetic_inputs
    out_dir = tmp_path / "out"
    md_path = tmp_path / "report.md"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--v85-no-fallback",
        str(v85),
        "--v85-dlt-fallback",
        str(v85_dlt),
        "--v86",
        str(v86),
        "--v25-dlt-fallback",
        str(v25),
        "--out-dir",
        str(out_dir),
        "--md-path",
        str(md_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}\n{result.stdout}"

    png_path = out_dir / "v85_v86_v25_sparse_view_comparison.png"
    json_path = out_dir / "v85_v86_v25_sparse_view_comparison.json"

    assert png_path.exists(), "PNG plot was not created"
    assert png_path.stat().st_size > 0, "PNG plot is empty"
    assert json_path.exists(), "JSON report was not created"
    assert md_path.exists(), "Markdown report was not created"

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert not report["v85 no-fallback"]["missing"]
    assert report["v25 DLT-fallback"]["S9"]["2"] == pytest.approx(58.18)
    assert report["v25 DLT-fallback"]["S11"]["4"] == pytest.approx(110.58)
    assert report["v86"]["S9"]["3"] == pytest.approx(1000.0)

    stdout = result.stdout
    assert "| 2 | S9 |" in stdout
    assert "58.18" in stdout
    assert "v85 no-fallback" in stdout


def test_handles_missing_v86_gracefully(synthetic_inputs, tmp_path: Path):
    v85, v85_dlt, _v86, v25 = synthetic_inputs
    out_dir = tmp_path / "out"
    md_path = tmp_path / "report.md"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--v85-no-fallback",
        str(v85),
        "--v85-dlt-fallback",
        str(v85_dlt),
        "--v25-dlt-fallback",
        str(v25),
        "--out-dir",
        str(out_dir),
        "--md-path",
        str(md_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}\n{result.stdout}"

    json_path = out_dir / "v85_v86_v25_sparse_view_comparison.json"
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["v86"]["missing"] is True
    assert report["v86"]["S9"]["2"] is None

    # The table should still be printed and contain the available rows.
    assert "v25 DLT-fallback" in result.stdout
    assert "v86" in result.stdout


def test_csv_input(synthetic_inputs, tmp_path: Path):
    v86 = tmp_path / "v86.csv"
    v86.write_text(
        "dataset,k,mpjpe_at_k\nS9,2,2200.0\nS9,3,1000.0\nS11,4,65.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    md_path = tmp_path / "report.md"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--v86",
        str(v86),
        "--out-dir",
        str(out_dir),
        "--md-path",
        str(md_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}\n{result.stdout}"

    report = json.loads(
        (out_dir / "v85_v86_v25_sparse_view_comparison.json").read_text(encoding="utf-8")
    )
    assert report["v86"]["S9"]["2"] == pytest.approx(2200.0)
    assert report["v86"]["S11"]["4"] == pytest.approx(65.0)
    assert report["v86"]["S9"]["4"] is None  # Not in CSV


def test_nonexistent_input_is_missing(tmp_path: Path):
    out_dir = tmp_path / "out"
    md_path = tmp_path / "report.md"
    fake = tmp_path / "does_not_exist.json"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--v85-no-fallback",
        str(fake),
        "--v25-dlt-fallback",
        str(fake),
        "--out-dir",
        str(out_dir),
        "--md-path",
        str(md_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}\n{result.stdout}"

    report = json.loads(
        (out_dir / "v85_v86_v25_sparse_view_comparison.json").read_text(encoding="utf-8")
    )
    assert report["v85 no-fallback"]["missing"] is True
    assert report["v25 DLT-fallback"]["missing"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
