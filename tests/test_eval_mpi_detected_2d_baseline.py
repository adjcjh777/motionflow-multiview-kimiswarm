"""Tests for scripts/eval_mpi_detected_2d_baseline.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_canonical_npz(path: Path, n_frames: int = 5, n_views: int = 14, n_joints: int = 28) -> None:
    """Create a minimal canonical multiview .npz for DLT baseline testing."""
    np.savez(
        path,
        points_2d=np.random.rand(n_frames, n_views, n_joints, 2) * 512,
        confidences=np.ones((n_frames, n_views, n_joints), dtype=np.float32),
        joints_3d=np.random.rand(n_frames, n_joints, 3),
        camera_K=np.tile(np.eye(3)[None, ...], (n_views, 1, 1)),
        camera_R=np.tile(np.eye(3)[None, ...], (n_views, 1, 1)),
        camera_t=np.zeros((n_views, 3)),
    )


def _write_config(path: Path, train_paths: list[Path], val_paths: list[Path]) -> None:
    """Write a YAML config with train/val paths (absolute or relative)."""
    lines = ["name: MPI Detected-2D Baseline Test\n"]
    lines.append("train_paths:\n")
    for p in train_paths:
        lines.append(f"  - {p}\n")
    lines.append("val_paths:\n")
    for p in val_paths:
        lines.append(f"  - {p}\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_eval_mpi_detected_2d_baseline_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test: script runs and produces expected JSON output."""
    train_file = tmp_path / "train.npz"
    val_file = tmp_path / "val.npz"
    _make_canonical_npz(train_file)
    _make_canonical_npz(val_file)

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, [train_file], [val_file])

    output_json = tmp_path / "result.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_mpi_detected_2d_baseline.py"),
        "--config",
        str(config_path),
        "--output",
        str(output_json),
        "--device",
        "cpu",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["unit"] == "mm"
    assert "confidence_weighted" in data
    assert "mean_mpjpe_mm" in data["confidence_weighted"]
    assert "weighted_mean_mpjpe_mm" in data["confidence_weighted"]
    assert data["confidence_weighted"]["splits"]["train"]["simple_mean_mm"] > 0
    assert data["confidence_weighted"]["splits"]["val"]["simple_mean_mm"] > 0


def test_eval_mpi_detected_2d_baseline_alternative_config_keys(tmp_path: Path) -> None:
    """Script should also accept YAML configs using ``train:`` / ``val:`` keys."""
    train_file = tmp_path / "train.npz"
    val_file = tmp_path / "val.npz"
    _make_canonical_npz(train_file)
    _make_canonical_npz(val_file)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"name: MPI Detected-2D Baseline Alternative Keys\n"
        f"train:\n"
        f"  - {train_file}\n"
        f"val:\n"
        f"  - {val_file}\n",
        encoding="utf-8",
    )

    output_json = tmp_path / "result.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_mpi_detected_2d_baseline.py"),
        "--config",
        str(config_path),
        "--output",
        str(output_json),
        "--device",
        "cpu",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert "confidence_weighted" in data
    assert data["confidence_weighted"]["splits"]["train"]["simple_mean_mm"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
