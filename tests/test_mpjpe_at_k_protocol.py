"""Smoke tests for the MPJPE@k evaluation protocol."""

import numpy as np
import pytest

from motionflow_mv.eval.mpjpe_at_k_protocol import (
    compute_mpjpe_at_k,
    generate_view_subsets,
    temporal_jerk,
)


class TestTemporalJerk:
    def test_short_sequence_returns_zero(self) -> None:
        poses = np.zeros((3, 17, 3))
        assert temporal_jerk(poses) == 0.0

    def test_constant_sequence_zero_jerk(self) -> None:
        poses = np.ones((10, 17, 3))
        assert temporal_jerk(poses) == 0.0

    def test_linear_motion_zero_jerk(self) -> None:
        t = np.arange(10)[:, None, None]
        poses = t * np.ones((10, 17, 3))
        # Third derivative of linear motion is zero.
        assert temporal_jerk(poses) == pytest.approx(0.0, abs=1e-6)


class TestGenerateViewSubsets:
    def test_enumerates_all_subsets_when_none_requested(self) -> None:
        subsets = generate_view_subsets(4, [2])
        assert len(subsets[2]) == 6  # C(4,2)
        assert all(len(s) == 2 for s in subsets[2])

    def test_samples_requested_number(self) -> None:
        subsets = generate_view_subsets(10, [3], num_subsets_per_k=5, seed=42)
        assert len(subsets[3]) == 5
        assert all(len(s) == 3 for s in subsets[3])

    def test_multiple_k_values(self) -> None:
        subsets = generate_view_subsets(5, [1, 2, 5])
        assert set(subsets.keys()) == {1, 2, 5}
        assert len(subsets[1]) == 5
        assert len(subsets[2]) == 10
        assert len(subsets[5]) == 1

    def test_invalid_k_is_skipped(self) -> None:
        subsets = generate_view_subsets(3, [0, 4])
        assert not subsets


class TestComputeMpjpAtK:
    def test_perfect_prediction_zero_error(self) -> None:
        gt = np.random.rand(20, 17, 3)
        pred = gt.copy()
        result = compute_mpjpe_at_k(pred, gt, k=3, align="none")
        assert result["mpjpe"] == pytest.approx(0.0, abs=1e-6)
        assert result["pa_mpjpe"] == pytest.approx(0.0, abs=1e-6)
        assert result["k"] == 3

    def test_nonzero_error_for_offset_prediction(self) -> None:
        gt = np.zeros((20, 17, 3))
        pred = gt + 0.01  # 10 mm per-coordinate offset
        result = compute_mpjpe_at_k(pred, gt, align="none")
        # Euclidean distance of a (0.01,0.01,0.01) m offset is sqrt(3)*10 mm.
        assert result["mpjpe"] == pytest.approx(10.0 * np.sqrt(3), abs=1e-3)

    def test_root_alignment_reduces_translation_error(self) -> None:
        gt = np.random.rand(20, 17, 3)
        pred = gt + 0.05  # large 50 mm translation
        result_root = compute_mpjpe_at_k(pred, gt, align="root")
        result_none = compute_mpjpe_at_k(pred, gt, align="none")
        assert result_root["mpjpe"] < result_none["mpjpe"]

    def test_pa_alignment_zero_for_rigid_offset(self) -> None:
        gt = np.random.rand(20, 17, 3)
        pred = gt + 0.02  # pure translation is rigid
        result = compute_mpjpe_at_k(pred, gt, align="pa")
        assert result["pa_mpjpe"] == pytest.approx(0.0, abs=1e-3)

    def test_unknown_align_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_mpjpe_at_k(np.zeros((5, 17, 3)), np.zeros((5, 17, 3)), align="bad")
