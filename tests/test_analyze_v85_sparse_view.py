"""Tests for scripts/analyze_v85_sparse_view.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


V85_JSON = {
    "per_dataset": {
        "S9": {
            "2": {
                "mpjpe_at_k": 100.0,
                "mean_mm": 100.0,
                "std_mm": 5.0,
                "n_subsets": 6,
                "temporal_jerk": 10.0,
            }
        },
        "S11": {
            "2": {
                "mpjpe_at_k": 120.0,
                "mean_mm": 120.0,
                "std_mm": 6.0,
                "n_subsets": 6,
                "temporal_jerk": 12.0,
            }
        },
    }
}


BASELINE_JSON = {
    "per_dataset": {
        "S9": {
            "2": {
                "mpjpe_at_k": 50.0,
                "mean_mm": 50.0,
                "std_mm": 3.0,
                "n_subsets": 6,
                "temporal_jerk": 5.0,
            },
            "3": {
                "mpjpe_at_k": 30.0,
                "mean_mm": 30.0,
                "std_mm": 2.0,
                "n_subsets": 4,
                "temporal_jerk": 3.0,
            },
        },
        "S11": {
            "2": {
                "mpjpe_at_k": 60.0,
                "mean_mm": 60.0,
                "std_mm": 4.0,
                "n_subsets": 6,
                "temporal_jerk": 6.0,
            },
            "3": {
                "mpjpe_at_k": 25.0,
                "mean_mm": 25.0,
                "std_mm": 1.5,
                "n_subsets": 4,
                "temporal_jerk": 2.5,
            },
        },
    }
}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


class TestAnalyzeV85SparseView:
    def test_single_json_comparison(self, tmp_path: Path) -> None:
        v85_path = tmp_path / "v85.json"
        baseline_path = tmp_path / "baseline.json"
        out_dir = tmp_path / "output"

        _write_json(v85_path, V85_JSON)
        _write_json(baseline_path, BASELINE_JSON)

        result = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_v85_sparse_view.py",
                "--v85_json",
                str(v85_path),
                "--baseline_json",
                str(baseline_path),
                "--out_dir",
                str(out_dir),
                "--no_plot",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert "v85 vs baseline summary" in result.stdout
        comparison_json = out_dir / "comparison.json"
        assert comparison_json.exists()
        data = json.loads(comparison_json.read_text())
        assert "per_dataset" in data
        assert "combined" in data
        assert data["per_dataset"]["S9"]["2"]["v85_mpjpe_mm"] == 100.0
        assert data["per_dataset"]["S9"]["2"]["baseline_mpjpe_mm"] == 50.0
        assert data["per_dataset"]["S9"]["2"]["delta_mm"] == 50.0
        # improvement = (50 - 100) / 50 * 100 = -100%
        assert data["per_dataset"]["S9"]["2"]["improvement_pct"] == pytest.approx(-100.0)

        # CSV should contain the same rows.
        comparison_csv = out_dir / "comparison.csv"
        assert comparison_csv.exists()
        rows = comparison_csv.read_text().strip().split("\n")
        assert len(rows) == 3  # header + 2 data rows

        # Markdown report should be generated.
        assert (out_dir / "report.md").exists()

    def test_per_k_json_merge(self, tmp_path: Path) -> None:
        v85_k2 = tmp_path / "v85_k2.json"
        v85_k3 = tmp_path / "v85_k3.json"
        baseline_path = tmp_path / "baseline.json"
        out_dir = tmp_path / "output"

        _write_json(v85_k2, V85_JSON)
        # Only S11 has k=3 in v85.
        _write_json(
            v85_k3,
            {
                "per_dataset": {
                    "S11": {
                        "3": {
                            "mpjpe_at_k": 90.0,
                            "mean_mm": 90.0,
                            "std_mm": 4.0,
                            "n_subsets": 4,
                            "temporal_jerk": 9.0,
                        }
                    }
                }
            },
        )
        _write_json(baseline_path, BASELINE_JSON)

        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_v85_sparse_view.py",
                "--v85_json",
                str(tmp_path / "v85_k*.json"),
                "--baseline_json",
                str(baseline_path),
                "--out_dir",
                str(out_dir),
                "--no_plot",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        data = json.loads((out_dir / "comparison.json").read_text())
        assert "S9" in data["per_dataset"]
        assert "2" in data["per_dataset"]["S9"]
        assert "3" not in data["per_dataset"]["S9"]
        assert "2" in data["per_dataset"]["S11"]
        assert "3" in data["per_dataset"]["S11"]
        assert data["per_dataset"]["S11"]["3"]["v85_mpjpe_mm"] == 90.0
        assert data["per_dataset"]["S11"]["3"]["baseline_mpjpe_mm"] == 25.0

    def test_missing_baseline_k(self, tmp_path: Path) -> None:
        v85_path = tmp_path / "v85.json"
        baseline_path = tmp_path / "baseline.json"
        out_dir = tmp_path / "output"

        _write_json(v85_path, V85_JSON)
        # Baseline only has k=3.
        _write_json(
            baseline_path,
            {
                "per_dataset": {
                    "S9": {
                        "3": {
                            "mpjpe_at_k": 30.0,
                            "mean_mm": 30.0,
                            "std_mm": 2.0,
                            "n_subsets": 4,
                            "temporal_jerk": 3.0,
                        }
                    }
                }
            },
        )

        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_v85_sparse_view.py",
                "--v85_json",
                str(v85_path),
                "--baseline_json",
                str(baseline_path),
                "--out_dir",
                str(out_dir),
                "--no_plot",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        data = json.loads((out_dir / "comparison.json").read_text())
        for row in data["rows"]:
            assert row["baseline_mpjpe_mm"] is None
            assert row["delta_mm"] is None
            assert row["improvement_pct"] is None

    def test_csv_only_input(self, tmp_path: Path) -> None:
        v85_csv = tmp_path / "v85.csv"
        baseline_csv = tmp_path / "baseline.csv"
        out_dir = tmp_path / "output"

        v85_csv.write_text(
            "dataset,k,mpjpe_at_k,mean_mm,std_mm,n_subsets,temporal_jerk\n"
            "S9,2,100.0,100.0,5.0,6,10.0\n"
            "S11,2,120.0,120.0,6.0,6,12.0\n"
        )
        baseline_csv.write_text(
            "dataset,k,mpjpe_at_k,mean_mm,std_mm,n_subsets,temporal_jerk\n"
            "S9,2,50.0,50.0,3.0,6,5.0\n"
            "S11,2,60.0,60.0,4.0,6,6.0\n"
        )

        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_v85_sparse_view.py",
                "--v85_csv",
                str(v85_csv),
                "--baseline_csv",
                str(baseline_csv),
                "--out_dir",
                str(out_dir),
                "--no_plot",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        data = json.loads((out_dir / "comparison.json").read_text())
        assert data["per_dataset"]["S9"]["2"]["v85_mpjpe_mm"] == 100.0
        assert data["per_dataset"]["S9"]["2"]["baseline_mpjpe_mm"] == 50.0

    def test_per_frame_flag(self, tmp_path: Path) -> None:
        v85_path = tmp_path / "v85.json"
        baseline_path = tmp_path / "baseline.json"
        out_dir = tmp_path / "output"

        v85_with_frames = {
            "per_dataset": {
                "S9": {
                    "2": {
                        "mpjpe_at_k": 100.0,
                        "mean_mm": 100.0,
                        "std_mm": 5.0,
                        "n_subsets": 6,
                        "temporal_jerk": 10.0,
                        "per_frame": [90.0, 100.0, 110.0],
                    }
                }
            }
        }

        _write_json(v85_path, v85_with_frames)
        _write_json(baseline_path, BASELINE_JSON)

        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_v85_sparse_view.py",
                "--v85_json",
                str(v85_path),
                "--baseline_json",
                str(baseline_path),
                "--out_dir",
                str(out_dir),
                "--no_plot",
                "--per_frame",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        per_frame_json = out_dir / "per_frame.json"
        assert per_frame_json.exists()
        per_frame = json.loads(per_frame_json.read_text())
        assert "S9" in per_frame
        assert "2" in per_frame["S9"]
        assert per_frame["S9"]["2"]["mean_mm"] == 100.0
        assert per_frame["S9"]["2"]["min_mm"] == 90.0
        assert per_frame["S9"]["2"]["max_mm"] == 110.0
        assert per_frame["S9"]["2"]["n_frames"] == 3

        per_frame_csv = out_dir / "per_frame.csv"
        assert per_frame_csv.exists()
        rows = per_frame_csv.read_text().strip().split("\n")
        assert len(rows) == 2  # header + 1 data row
        assert "S9" in rows[1]

    def test_per_camera_analysis_flag(self, tmp_path: Path) -> None:
        v85_path = tmp_path / "v85.json"
        baseline_path = tmp_path / "baseline.json"
        out_dir = tmp_path / "output"

        # Camera 0 in both subsets, camera 1 only in subset (0,1).
        v85_with_subsets = {
            "per_dataset": {
                "S9": {
                    "2": {
                        "mpjpe_at_k": 75.0,
                        "mean_mm": 75.0,
                        "std_mm": 5.0,
                        "n_subsets": 2,
                        "temporal_jerk": 5.0,
                        "subsets": [[0, 1], [0, 2]],
                        "per_subset": [
                            {"mpjpe": 70.0},
                            {"mpjpe": 80.0},
                        ],
                    }
                }
            }
        }

        _write_json(v85_path, v85_with_subsets)
        _write_json(baseline_path, BASELINE_JSON)

        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_v85_sparse_view.py",
                "--v85_json",
                str(v85_path),
                "--baseline_json",
                str(baseline_path),
                "--out_dir",
                str(out_dir),
                "--no_plot",
                "--per_camera_analysis",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        per_camera_json = out_dir / "per_camera.json"
        assert per_camera_json.exists()
        per_camera = json.loads(per_camera_json.read_text())
        assert "S9" in per_camera
        assert "2" in per_camera["S9"]
        cameras = per_camera["S9"]["2"]["cameras"]
        assert cameras["0"]["count"] == 2
        assert cameras["0"]["mean_mpjpe_mm"] == 75.0
        assert cameras["1"]["count"] == 1
        assert cameras["1"]["mean_mpjpe_mm"] == 70.0
        assert cameras["2"]["count"] == 1
        assert cameras["2"]["mean_mpjpe_mm"] == 80.0

        per_camera_csv = out_dir / "per_camera.csv"
        assert per_camera_csv.exists()
        rows = per_camera_csv.read_text().strip().split("\n")
        assert len(rows) == 4  # header + 3 camera rows

    def test_per_frame_and_per_camera_analysis_together(self, tmp_path: Path) -> None:
        v85_path = tmp_path / "v85.json"
        baseline_path = tmp_path / "baseline.json"
        out_dir = tmp_path / "output"

        v85_mixed = {
            "per_dataset": {
                "S9": {
                    "2": {
                        "mpjpe_at_k": 100.0,
                        "mean_mm": 100.0,
                        "std_mm": 5.0,
                        "n_subsets": 2,
                        "temporal_jerk": 10.0,
                        "per_frame": [90.0, 110.0],
                        "subsets": [[0, 1], [0, 2]],
                        "per_subset": [
                            {"mpjpe": 90.0},
                            {"mpjpe": 110.0},
                        ],
                    }
                }
            }
        }

        _write_json(v85_path, v85_mixed)
        _write_json(baseline_path, BASELINE_JSON)

        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_v85_sparse_view.py",
                "--v85_json",
                str(v85_path),
                "--baseline_json",
                str(baseline_path),
                "--out_dir",
                str(out_dir),
                "--no_plot",
                "--per_frame",
                "--per_camera_analysis",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert (out_dir / "per_frame.json").exists()
        assert (out_dir / "per_camera.json").exists()
        report = (out_dir / "report.md").read_text()
        assert "## Per-Frame Analysis" in report
        assert "## Per-Camera Analysis" in report
