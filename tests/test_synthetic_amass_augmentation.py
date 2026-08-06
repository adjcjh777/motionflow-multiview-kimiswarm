"""Smoke tests for synthetic SMPL/AMASS augmentation.

These tests do not train anything; they only verify that the generator and
dataset APIs produce tensors with the expected shape and that the legacy API
remains backward compatible.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.data.synthetic_3d_dataset import (
    AugmentConfig,
    CameraRigSampler,
    SyntheticMultiViewDataset,
    augment_2d_keypoints,
    generate_dataset,
    generate_sequence,
    generate_synthetic_dataset,
    project_points,
)

SMPL_NEUTRAL_PATH = Path("data/smpl/SMPL_NEUTRAL.pkl")


def _smpl_available() -> bool:
    """Return True if smplx and the neutral SMPL model are available."""
    try:
        import smplx  # noqa: F401
    except ImportError:
        return False
    return SMPL_NEUTRAL_PATH.exists()


@pytest.fixture
def smpl_neutral_path() -> Path:
    return SMPL_NEUTRAL_PATH


def test_legacy_generate_sequence():
    """The original simple generator still returns the expected shapes."""
    inputs, baselines, gt, cameras = generate_sequence(
        n_frames=10, n_views=5, j=17, noise_std=1.0
    )
    assert inputs.shape == (10, 5, 17, 3)
    assert baselines.shape == (10, 17, 3)
    assert gt.shape == (10, 17, 3)
    assert len(cameras) == 5
    assert all(isinstance(c, Camera) for c in cameras)


def test_legacy_generate_dataset():
    """The original batch generator returns the expected shapes."""
    X, B, Y = generate_dataset(n_seq=3, n_frames=8, n_views=4, j=17)
    assert X.shape == (3, 8, 4, 17, 3)
    assert B.shape == (3, 8, 17, 3)
    assert Y.shape == (3, 8, 17, 3)


def test_camera_rig_sampler():
    """Each camera mode returns the requested number of cameras."""
    rng = np.random.default_rng(7)
    for mode in ["h36m", "mpiinf3dhp", "legacy", "random"]:
        sampler = CameraRigSampler(mode)
        cameras = sampler.sample(n_views=4, rng=rng)
        assert len(cameras) == 4
        for cam in cameras:
            assert cam.K.shape == (3, 3)
            assert cam.R.shape == (3, 3)
            assert cam.t.shape == (3,)


def test_augment_2d_keypoints():
    """2D augmentation does not crash and preserves shape."""
    rng = np.random.default_rng(8)
    V, J = 4, 17
    points = rng.normal(0, 100, size=(V, J, 2)).astype(np.float32)
    conf = np.ones((V, J), dtype=np.float32)
    config = AugmentConfig(
        noise_std=2.0,
        occlusion_rate=0.1,
        outlier_rate=0.05,
        outlier_scale=50.0,
    )
    p_aug, c_aug = augment_2d_keypoints(points, conf, rng, config)
    assert p_aug.shape == points.shape
    assert c_aug.shape == conf.shape


def test_project_points():
    """Projection of a simple 3D point set yields finite 2D coordinates."""
    rng = np.random.default_rng(9)
    sampler = CameraRigSampler("h36m")
    cameras = sampler.sample(n_views=2, rng=rng)
    points_3d = rng.normal(0, 500, size=(17, 3)).astype(np.float64)
    points_2d = project_points(points_3d, cameras[0])
    assert points_2d.shape == (17, 2)
    assert np.all(np.isfinite(points_2d))


@pytest.mark.skipif(not _smpl_available(), reason="smplx or SMPL model not available")
def test_smpl_sequence_generator():
    """The new SMPL sequence generator produces a valid canonical .npz."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "synthetic_test.npz"
        path = generate_synthetic_dataset(
            output=str(out),
            smpl_model_path=str(SMPL_NEUTRAL_PATH),
            n_sequences=2,
            n_frames=10,
            n_views=4,
            camera_mode="h36m",
            augment_config=AugmentConfig(noise_std=1.0),
            seed=42,
        )
        data = np.load(path)
        assert data["joints_3d"].shape[0] == 2 * 10
        assert data["points_2d"].shape == (2 * 10, 4, 17, 2)
        assert data["confidences"].shape == (2 * 10, 4, 17)
        assert data["camera_K"].shape == (2 * 10, 4, 3, 3)
        assert data["camera_R"].shape == (2 * 10, 4, 3, 3)
        assert data["camera_t"].shape == (2 * 10, 4, 3)


@pytest.mark.skipif(not _smpl_available(), reason="smplx or SMPL model not available")
def test_synthetic_multiview_dataset():
    """The PyTorch Dataset wrapper yields tensors compatible with collate_fn."""
    dataset = SyntheticMultiViewDataset(
        smpl_model_path=str(SMPL_NEUTRAL_PATH),
        n_sequences=2,
        n_frames=8,
        n_views=4,
        camera_mode="mpiinf3dhp",
        seed=123,
        device=torch.device("cpu"),
    )
    assert len(dataset) == 2
    x, y, K, R, t = dataset[0]
    assert x.shape == (8, 4, 17, 3)
    assert y.shape == (8, 17, 3)
    assert K.shape == (4, 3, 3)
    assert R.shape == (4, 3, 3)
    assert t.shape == (4, 3)
