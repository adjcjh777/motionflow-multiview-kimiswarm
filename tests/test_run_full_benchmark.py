"""Dry-run tests for scripts/run_full_benchmark.py."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


def _write_dummy_npz(path: Path, n_frames: int = 10, n_views: int = 4, n_joints: int = 17):
    npz_path = Path(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        points_2d=np.zeros((n_frames, n_views, n_joints, 2), dtype=np.float32),
        confidences=np.ones((n_frames, n_views, n_joints), dtype=np.float32),
        joints_3d=np.zeros((n_frames, n_joints, 3), dtype=np.float32),
        camera_K=np.tile(np.eye(3, dtype=np.float32)[None, None], (1, n_views, 1, 1)),
        camera_R=np.tile(np.eye(3, dtype=np.float32)[None, None], (1, n_views, 1, 1)),
        camera_t=np.zeros((1, n_views, 3), dtype=np.float32),
    )


def test_run_full_benchmark_dry_run(tmp_path):
    """Dry-run should produce a valid JSON report without touching a GPU."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    npz_path = data_dir / "dummy.npz"
    _write_dummy_npz(npz_path)

    split_path = tmp_path / "splits" / "dummy.yaml"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(yaml.dump({"test": [str(npz_path)]}))

    manifest = {
        "model_config": {
            "model": "crossview_residual_pp",
            "checkpoint": str(tmp_path / "nonexistent.pth"),
            "clip_len": 13,
            "d": 64,
            "n_st_layers": 2,
            "residual_hidden": 128,
            "batch_size": 2,
            "source_n_views": 4,
        },
        "datasets": [
            {"name": "h36m_test", "path": str(split_path), "split": "test"},
            {"name": "mpiinf3dhp_test", "path": str(npz_path)},
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.dump(manifest))

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/run_full_benchmark.py",
        "--manifest",
        str(manifest_path),
        "--out",
        str(out_dir),
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    report_path = out_dir / "benchmark_results.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())

    assert report["dry_run"] is True
    assert "model_config" in report
    assert len(report["datasets"]) == 2
    assert "summary" in report
    assert "overall" in report["summary"]

    for ds in report["datasets"]:
        assert "metrics" in ds
        assert "mpjpe" in ds["metrics"]
        assert "pa_mpjpe" in ds["metrics"]
        assert "sequences" in ds
        assert len(ds["sequences"]) >= 1


def test_expand_dataset(tmp_path):
    from scripts.run_full_benchmark import _expand_dataset

    npz_path = tmp_path / "seq.npz"
    _write_dummy_npz(npz_path)
    yaml_path = tmp_path / "split.yaml"
    yaml_path.write_text(yaml.dump({"test": [str(npz_path)], "val": []}))

    single = _expand_dataset("single", str(npz_path), "test")
    assert len(single) == 1
    assert single[0]["file"] == str(npz_path)

    multi = _expand_dataset("multi", str(yaml_path), "test")
    assert len(multi) == 1
    assert multi[0]["file"] == str(npz_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
