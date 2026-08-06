"""Tests for the mixed-dataset balanced sampler and smoke training."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.mixed_dataset import (
    DATASET_IDS,
    DATASET_REGISTRY,
    DatasetBalancedSampler,
    MixedDataset,
    build_mixed_dataloaders,
)


def _make_canonical_npz(path, n_views, n_joints, n_frames=50):
    """Create a tiny canonical mixed-dataset .npz file for tests."""
    points_2d = np.random.randn(n_frames, n_views, n_joints, 2).astype(np.float32)
    confidences = np.random.rand(n_frames, n_views, n_joints).astype(np.float32)
    joints_3d = np.random.randn(n_frames, n_joints, 3).astype(np.float32)
    camera_K = np.eye(3, dtype=np.float32)[None, ...].repeat(n_views, axis=0)
    camera_R = np.eye(3, dtype=np.float32)[None, ...].repeat(n_views, axis=0)
    camera_t = np.zeros((n_views, 3), dtype=np.float32)
    np.savez(
        path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=camera_K,
        camera_R=camera_R,
        camera_t=camera_t,
    )


@pytest.fixture
def mixed_smoke_paths(tmp_path):
    mpi_path = tmp_path / "mpi_smoke.npz"
    h36m_path = tmp_path / "h36m_smoke.npz"
    _make_canonical_npz(mpi_path, n_views=14, n_joints=28)
    _make_canonical_npz(h36m_path, n_views=4, n_joints=17)
    return {"mpi": str(mpi_path), "h36m": str(h36m_path)}


def test_dataset_balanced_sampler_balanced():
    sampler = DatasetBalancedSampler([2, 6], samples_per_dataset=4, seed=42)
    indices = list(sampler)
    assert len(indices) == 8
    # First dataset indices should be in [0, 1], second in [2, 7].
    first_half = [i for i in indices if i < 2]
    second_half = [i for i in indices if i >= 2]
    assert len(first_half) == 4
    assert len(second_half) == 4


def test_dataset_balanced_sampler_no_replacement():
    sampler = DatasetBalancedSampler([3, 3], replacement=False, seed=42)
    indices = set(sampler)
    assert len(indices) == 6


def test_build_mixed_dataloaders_balance_flag(mixed_smoke_paths):
    train_loader, val_loader = build_mixed_dataloaders(
        train_paths={"mpi": [mixed_smoke_paths["mpi"]], "h36m": [mixed_smoke_paths["h36m"]]},
        val_path=mixed_smoke_paths["mpi"],
        val_dataset="mpi",
        clip_len=9,
        batch_size=2,
        train_samples=8,
        balance_datasets=True,
        balance_samples_per_dataset=4,
        balance_seed=42,
    )

    ids = []
    for _, _, _, _, _, batch_ids in train_loader:
        ids.extend(batch_ids.tolist())
    # 4 samples from each dataset -> expect 4 MPI (id=0) and 4 H36M (id=2).
    assert ids.count(DATASET_IDS["mpi"]) == 4
    assert ids.count(DATASET_IDS["h36m"]) == 4

    # Val loader should still work unchanged.
    for xb, yb, K, R, t, ids in val_loader:
        assert xb.shape[0] > 0
        break


def test_mixed_dataset_shape_and_id(mixed_smoke_paths):
    ds = MixedDataset(mixed_smoke_paths["mpi"], "mpi", clip_len=9, n_samples=2)
    x, y, K, R, t, did = ds[0]
    assert x.shape == (9, DATASET_REGISTRY["mpi"]["n_views"], DATASET_REGISTRY["mpi"]["n_joints"], 3)
    assert y.shape == (9, DATASET_REGISTRY["mpi"]["n_joints"], 3)
    assert did == DATASET_IDS["mpi"]


def test_train_mixed_dataset_principal_point_smoke(mixed_smoke_paths):
    """End-to-end smoke run of the mixed-dataset PP trainer."""
    cmd = [
        sys.executable,
        "experiments/train_mixed_dataset_principal_point.py",
        "--mpi_train", mixed_smoke_paths["mpi"],
        "--h36m_train", mixed_smoke_paths["h36m"],
        "--val", mixed_smoke_paths["mpi"],
        "--val_dataset", "mpi",
        "--balance_datasets",
        "--smoke",
        "--output", str(Path(mixed_smoke_paths["mpi"]).parent / "mixed_pp_smoke.pth"),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "Smoke mode" in result.stdout
    assert "Best val MPJPE" in result.stdout
