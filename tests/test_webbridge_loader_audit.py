"""Audit tests for the canonical WebBridge loader.

These tests verify that the loader fails loudly when it would otherwise
produce circular 3D labels, and that safer defaults are used for datasets
that have clean optimized ground-truth available.
"""

import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from motionflow_mv.data.webbridge_loader import convert_human36m


def _make_minimal_h36m_archive(tmp_path: Path) -> Path:
    """Create a minimal fake H36M archive and camera params file."""
    data_root = tmp_path / "h36m_hf"
    data_root.mkdir(parents=True, exist_ok=True)

    # Minimal camera params for subject 1.
    camera_names = ["54138969", "55011271", "58860488", "60457274"]
    intrinsics = {
        name: {
            "calibration_matrix": [
                [1000.0, 0.0, 512.0],
                [0.0, 1000.0, 512.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        for name in camera_names
    }
    extrinsics = {
        "S1": {
            name: {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [[0.0], [0.0], [5000.0]],
            }
            for name in camera_names
        }
    }
    cam_path = data_root / "camera_params.json"
    with open(cam_path, "w") as f:
        json.dump({"intrinsics": intrinsics, "extrinsics": extrinsics}, f)

    # Minimal pkl archive: one subject, one action, one frame.
    sources = ["s_01_act_02_cam_01", "s_01_act_02_cam_02",
               "s_01_act_02_cam_03", "s_01_act_02_cam_04"]
    n_joints = 17
    data = {
        "train": {
            "source": sources,
            "joint_2d": np.random.randn(len(sources), n_joints, 2).astype(np.float64),
            "confidence": np.ones((len(sources), n_joints, 1), dtype=np.float64),
            "camera_name": [camera_names[i] for i in range(len(sources))],
        }
    }
    archive_path = data_root / "h36m_sh_conf_cam_source_final.pkl.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        with z.open("h36m_sh_conf_cam_source_final.pkl", "w") as f:
            f.write(pickle.dumps(data))

    return data_root


def test_convert_human36m_fails_without_true_gt(tmp_path: Path) -> None:
    """Without true 3D GT, conversion must raise rather than emit circular labels."""
    data_root = _make_minimal_h36m_archive(tmp_path)
    out_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match="No true 3D GT found"):
        convert_human36m(
            data_root=data_root,
            subject=1,
            actions=[2],
            split="train",
            out_dir=out_dir,
            true_gt_dir=tmp_path / "empty",
            allow_circular_fallback=False,
        )


def test_convert_human36m_allows_fallback_when_enabled(tmp_path: Path) -> None:
    """The legacy circular fallback is only allowed when explicitly requested."""
    data_root = _make_minimal_h36m_archive(tmp_path)
    out_dir = tmp_path / "out"

    out_path = convert_human36m(
        data_root=data_root,
        subject=1,
        actions=[2],
        split="train",
        out_dir=out_dir,
        true_gt_dir=tmp_path / "empty",
        allow_circular_fallback=True,
    )

    assert out_path.exists()
    data = np.load(out_path)
    assert "points_2d" in data
    assert "joints_3d" in data
