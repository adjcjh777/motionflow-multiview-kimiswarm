"""Unit tests for scripts/generate_mpi_detected_2d.py.

These tests run with the fallback detector only; they do not require MediaPipe
or OpenCV to be installed.
"""

import numpy as np
import pytest

from scripts.generate_mpi_detected_2d import (
    FallbackDetector,
    _build_detector,
    _generate_detected_sequence,
    _resolve_view_image_paths,
)


def _make_canonical_npz(path: str, n_frames: int = 5, n_views: int = 14, n_joints: int = 28) -> None:
    np.savez(
        path,
        points_2d=np.random.rand(n_frames, n_views, n_joints, 2) * 512,
        confidences=np.ones((n_frames, n_views, n_joints), dtype=np.float32),
        joints_3d=np.random.rand(n_frames, n_joints, 3),
        camera_K=np.tile(np.eye(3)[None, ...], (n_views, 1, 1)),
        camera_R=np.tile(np.eye(3)[None, ...], (n_views, 1, 1)),
        camera_t=np.zeros((n_views, 3)),
    )


def test_build_detector_auto_falls_back():
    detector = _build_detector("auto", fallback_noise=1.0)
    assert isinstance(detector, FallbackDetector)
    assert detector.noise_std == 1.0


def test_build_detector_explicit_fallback():
    detector = _build_detector("fallback", fallback_noise=2.5)
    assert isinstance(detector, FallbackDetector)
    assert detector.noise_std == 2.5


def test_build_detector_unknown_raises():
    with pytest.raises(ValueError, match="Unknown detector"):
        _build_detector("not_a_detector")


def test_fallback_detector_shape_and_noise():
    detector = FallbackDetector(noise_std=0.0)
    gt = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float64)
    out, conf = detector([], gt)
    assert out.shape == gt.shape
    assert conf.shape == (2, 3)
    np.testing.assert_allclose(out, gt)

    detector_noisy = FallbackDetector(noise_std=5.0)
    out_noisy, conf_noisy = detector_noisy([], gt)
    assert out_noisy.shape == gt.shape
    assert not np.allclose(out_noisy, gt)
    np.testing.assert_allclose(conf_noisy, np.ones((2, 3)) * 0.9)


def test_resolve_view_image_paths_returns_none_without_image_dir(tmp_path):
    paths = _resolve_view_image_paths(tmp_path, "s_01_seq_01_v14_multiview_m.npz", 0, 14)
    assert all(p is None for p in paths)
    assert len(paths) == 14


def test_generate_detected_sequence_output_structure(tmp_path):
    input_npz = tmp_path / "input.npz"
    output_npz = tmp_path / "output.npz"
    _make_canonical_npz(str(input_npz))

    detector = FallbackDetector(noise_std=0.0)
    _generate_detected_sequence(
        input_npz,
        output_npz,
        detector,
        fallback_noise=0.0,
        image_dir=None,
    )

    assert output_npz.exists()
    data = dict(np.load(output_npz))
    assert set(data.keys()) == {
        "points_2d",
        "confidences",
        "joints_3d",
        "camera_K",
        "camera_R",
        "camera_t",
    }
    assert data["points_2d"].shape == (5, 14, 28, 2)
    assert data["confidences"].shape == (5, 14, 28)
    assert data["joints_3d"].shape == (5, 28, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
