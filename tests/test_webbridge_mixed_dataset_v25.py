"""Tests for v25 view-mask support in the WebBridge mixed loader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from motionflow_mv.data.webbridge_mixed_dataset import (
    WebBridgeCanonical17Dataset,
    WebBridgeMixedDataset,
    build_webbridge_mixed_dataloaders,
    webbridge_mixed_collate_fn,
    webbridge_mixed_collate_fn_with_mask,
)


@pytest.fixture
def tiny_h36m_npz() -> str:
    """Create a tiny H36M-shaped canonical ``.npz`` (4 views, 17 joints)."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        T, V, J = 30, 4, 17
        np.savez(
            f,
            points_2d=np.random.randn(T, V, J, 2).astype(np.float32),
            confidences=np.random.rand(T, V, J).astype(np.float32),
            joints_3d=np.random.randn(T, J, 3).astype(np.float32),
            camera_K=np.stack([np.eye(3) for _ in range(V)], axis=0).astype(np.float32),
            camera_R=np.stack([np.eye(3) for _ in range(V)], axis=0).astype(np.float32),
            camera_t=np.zeros((V, 3), dtype=np.float32),
        )
        return f.name


@pytest.fixture
def tiny_mpi_npz() -> str:
    """Create a tiny MPI-INF-3DHP canonical ``.npz`` (14 views, 28 joints)."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        T, V, J = 30, 14, 28
        np.savez(
            f,
            points_2d=np.random.randn(T, V, J, 2).astype(np.float32),
            confidences=np.random.rand(T, V, J).astype(np.float32),
            joints_3d=np.random.randn(T, J, 3).astype(np.float32),
            camera_K=np.stack([np.eye(3) for _ in range(V)], axis=0).astype(np.float32),
            camera_R=np.stack([np.eye(3) for _ in range(V)], axis=0).astype(np.float32),
            camera_t=np.zeros((V, 3), dtype=np.float32),
        )
        return f.name


def test_default_return_shape(tiny_h36m_npz: str) -> None:
    ds = WebBridgeCanonical17Dataset(tiny_h36m_npz, "h36m", clip_len=9)
    sample = ds[0]
    assert len(sample) == 6
    x, y, K, R, t, dataset_id = sample
    assert x.shape == (9, 14, 17, 3)
    assert y.shape == (9, 17, 3)
    assert K.shape == (14, 3, 3)
    assert dataset_id == 0


def test_view_mask_shape(tiny_h36m_npz: str) -> None:
    ds = WebBridgeCanonical17Dataset(
        tiny_h36m_npz, "h36m", clip_len=9, return_view_mask=True
    )
    sample = ds[0]
    assert len(sample) == 7
    x, y, K, R, t, dataset_id, view_mask = sample
    assert view_mask.shape == (14,)
    assert view_mask.dtype == torch.bool
    assert view_mask[:4].all()
    assert not view_mask[4:].any()


def test_mpi_canonical_17_joints(tiny_mpi_npz: str) -> None:
    ds = WebBridgeCanonical17Dataset(
        tiny_mpi_npz, "mpi", clip_len=9, return_view_mask=True
    )
    x, y, K, R, t, dataset_id, view_mask = ds[0]
    assert x.shape == (9, 14, 17, 3)
    assert y.shape == (9, 17, 3)
    assert view_mask.sum().item() == 14


def test_collate_fn_with_mask(tiny_h36m_npz: str) -> None:
    ds = WebBridgeCanonical17Dataset(
        tiny_h36m_npz, "h36m", clip_len=9, return_view_mask=True
    )
    batch = [ds[i] for i in range(3)]
    x, y, K, R, t, dataset_ids, view_mask = webbridge_mixed_collate_fn_with_mask(batch)
    assert x.shape == (3, 9, 14, 17, 3)
    assert view_mask.shape == (3, 14)
    assert view_mask[:, :4].all()
    assert not view_mask[:, 4:].any()


def test_build_dataloaders_with_mask(tiny_h36m_npz: str) -> None:
    train_loader, val_loader = build_webbridge_mixed_dataloaders(
        train_paths=[tiny_h36m_npz],
        train_names=["h36m"],
        val_paths=[tiny_h36m_npz],
        val_names=["h36m"],
        clip_len=9,
        batch_size=2,
        train_samples=4,
        return_view_mask=True,
    )
    x, y, K, R, t, dataset_ids, view_mask = next(iter(train_loader))
    assert x.shape == (2, 9, 14, 17, 3)
    assert view_mask.shape == (2, 14)
    assert view_mask[:, :4].all()

    x, y, K, R, t, dataset_ids, view_mask = next(iter(val_loader))
    assert x.shape[0] <= 2  # val may be smaller than batch_size


def test_backward_compatible_without_mask(tiny_h36m_npz: str) -> None:
    """Existing callers using the old collate fn must not break."""
    ds = WebBridgeCanonical17Dataset(tiny_h36m_npz, "h36m", clip_len=9)
    batch = [ds[i] for i in range(3)]
    x, y, K, R, t, dataset_ids = webbridge_mixed_collate_fn(batch)
    assert x.shape == (3, 9, 14, 17, 3)


def test_mixed_dataset_propagates_mask(tiny_h36m_npz: str, tiny_mpi_npz: str) -> None:
    ds = WebBridgeMixedDataset(
        [tiny_h36m_npz, tiny_mpi_npz],
        ["h36m", "mpi"],
        clip_len=9,
        return_view_mask=True,
    )
    sample = ds[0]
    _, _, _, _, _, _, view_mask = sample
    assert view_mask[:4].all()
    assert not view_mask[4:].any()
