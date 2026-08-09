import pathlib
import tempfile

import numpy as np
import pytest
import torch

from motionflow_mv.data.webbridge_mixed_dataset import WebBridgeCanonical17Dataset


@pytest.mark.parametrize("return_view_mask", [False, True])
def test_webbridge_3dpw_canonical17_dataset(return_view_mask: bool) -> None:
    """Verify 3DPW canonical .npz loads and maps to 17 joints / 14 padded views."""
    npz_path = pathlib.Path("data/webbridge/3dpw/converted/train/courtyard_arguing_00_pseudo.npz")
    if not npz_path.exists():
        pytest.skip("3DPW canonical .npz not available on this machine")

    dataset = WebBridgeCanonical17Dataset(
        str(npz_path),
        dataset_name="3dpw",
        clip_len=9,
        n_samples=2,
        return_view_mask=return_view_mask,
    )
    assert len(dataset) == 2
    sample = dataset[0]
    if return_view_mask:
        x, y, K, R, t, dataset_id, view_mask = sample
        assert view_mask.shape == (14,)
        assert view_mask.sum() > 0
    else:
        x, y, K, R, t, dataset_id = sample

    assert x.shape == (9, 14, 17, 3)
    assert y.shape == (9, 17, 3)
    assert K.shape == (14, 3, 3)
    assert R.shape == (14, 3, 3)
    assert t.shape == (14, 3)
    assert dataset_id == 5


def _make_3dpw_actual_npz(tmp_path: pathlib.Path, T: int = 20) -> pathlib.Path:
    """Create a tiny 3DPW actual-mode .npz with 24 source joints and 1 moving camera."""
    npz_path = tmp_path / "tiny_3dpw_actual.npz"

    J24 = 24
    V = 1

    points_2d = np.random.randn(T, V, J24, 2).astype(np.float64)
    confidences = np.ones((T, V, J24), dtype=np.float32)
    joints_3d = np.random.randn(T, J24, 3).astype(np.float64)

    camera_K = np.eye(3, dtype=np.float64)[None, ...].repeat(V, axis=0)
    camera_R = np.eye(3, dtype=np.float64)[None, ...].repeat(V, axis=0)
    camera_t = np.zeros((V, 3), dtype=np.float64)

    # Per-frame moving cameras: shape (T, V, ...)
    camera_K_frames = np.eye(3, dtype=np.float64)[None, None, ...].repeat(T, axis=0).repeat(V, axis=1)
    camera_R_frames = np.eye(3, dtype=np.float64)[None, None, ...].repeat(T, axis=0).repeat(V, axis=1)
    camera_t_frames = np.zeros((T, V, 3), dtype=np.float64)

    np.savez_compressed(
        npz_path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=camera_K,
        camera_R=camera_R,
        camera_t=camera_t,
        camera_K_frames=camera_K_frames,
        camera_R_frames=camera_R_frames,
        camera_t_frames=camera_t_frames,
    )
    return npz_path


@pytest.mark.parametrize("return_view_mask", [False, True])
def test_webbridge_3dpw_actual_mode(return_view_mask: bool, tmp_path: pathlib.Path) -> None:
    """Verify 3DPW actual-mode per-frame cameras are loaded and padded correctly."""
    npz_path = _make_3dpw_actual_npz(tmp_path)

    dataset = WebBridgeCanonical17Dataset(
        str(npz_path),
        dataset_name="3dpw",
        clip_len=9,
        n_samples=2,
        return_view_mask=return_view_mask,
    )
    assert len(dataset) == 2
    sample = dataset[0]
    if return_view_mask:
        x, y, K, R, t, dataset_id, view_mask = sample
        assert view_mask.shape == (14,)
        assert view_mask[:1].sum() == 1
        assert view_mask[1:].sum() == 0
    else:
        x, y, K, R, t, dataset_id = sample

    assert x.shape == (9, 14, 17, 3)
    assert y.shape == (9, 17, 3)
    assert K.shape == (9, 14, 3, 3)
    assert R.shape == (9, 14, 3, 3)
    assert t.shape == (9, 14, 3)
    assert dataset_id == 5
