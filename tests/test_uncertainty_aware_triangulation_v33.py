"""Unit tests for the v33 uncertainty-aware triangulation head."""

import pytest
import torch

from motionflow_mv.fusion.uncertainty_aware_triangulation_v33 import (
    UncertaintyAwareTriangulationV33,
)


def _make_toy_inputs(
    batch: int = 2,
    temporal: int = 3,
    views: int = 4,
    joints: int = 17,
    feature_dim: int = 64,
):
    device = torch.device("cpu")
    points_2d = torch.rand(batch, temporal, views, joints, 2, device=device)
    confidences = torch.rand(batch, temporal, views, joints, device=device)
    features = torch.randn(batch, temporal, views, joints, feature_dim, device=device)
    proj_matrices = torch.randn(batch, temporal, views, 3, 4, device=device)
    pred_3d_init = torch.randn(batch, temporal, joints, 3, device=device)
    return points_2d, confidences, features, proj_matrices, pred_3d_init


def test_uncertainty_aware_triangulation_v33_forward_shape():
    """Forward pass should return refined 3D pose and a scalar uncertainty loss."""
    points_2d, confidences, features, proj_matrices, pred_3d_init = _make_toy_inputs()
    head = UncertaintyAwareTriangulationV33(d=64)

    pred_3d_ref, uncertainty_loss = head(
        points_2d=points_2d,
        confidences=confidences,
        features=features,
        proj_matrices=proj_matrices,
        pred_3d_init=pred_3d_init,
    )

    B, T, V, J, _ = points_2d.shape
    assert pred_3d_ref.shape == (B, T, J, 3)
    assert uncertainty_loss.shape == ()
    assert torch.isfinite(uncertainty_loss)


def test_uncertainty_aware_triangulation_v33_identity_at_init():
    """At init, residual_scale=0, so output should equal the initial estimate."""
    points_2d, confidences, features, proj_matrices, pred_3d_init = _make_toy_inputs()
    head = UncertaintyAwareTriangulationV33(d=64)

    pred_3d_ref, _ = head(
        points_2d=points_2d,
        confidences=confidences,
        features=features,
        proj_matrices=proj_matrices,
        pred_3d_init=pred_3d_init,
    )

    torch.testing.assert_close(pred_3d_ref, pred_3d_init, atol=1e-6, rtol=1e-6)


def test_uncertainty_aware_triangulation_v33_view_mask_ignores_views():
    """Masked-out views should not crash and should produce finite loss."""
    points_2d, confidences, features, proj_matrices, pred_3d_init = _make_toy_inputs()
    head = UncertaintyAwareTriangulationV33(d=64)
    view_mask = torch.ones(2, 3, 4)
    view_mask[:, :, 0] = 0.0  # Drop first view.

    pred_3d_ref, uncertainty_loss = head(
        points_2d=points_2d,
        confidences=confidences,
        features=features,
        proj_matrices=proj_matrices,
        pred_3d_init=pred_3d_init,
        view_mask=view_mask,
    )

    assert pred_3d_ref.shape == (2, 3, 17, 3)
    assert torch.isfinite(uncertainty_loss)


def test_uncertainty_aware_triangulation_v33_gradients_flow():
    """The head should be differentiable w.r.t. features."""
    points_2d, confidences, features, proj_matrices, pred_3d_init = _make_toy_inputs()
    features = features.requires_grad_(True)
    head = UncertaintyAwareTriangulationV33(d=64)

    pred_3d_ref, uncertainty_loss = head(
        points_2d=points_2d,
        confidences=confidences,
        features=features,
        proj_matrices=proj_matrices,
        pred_3d_init=pred_3d_init,
    )

    total_loss = uncertainty_loss + pred_3d_ref.mean()
    total_loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
