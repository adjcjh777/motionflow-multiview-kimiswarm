"""CPU smoke tests for the standalone visibility-gated fusion v2 module."""

import pytest
import torch

from motionflow_mv.fusion.visibility_gated_fusion_v2 import (
    VisibilityGatedFusionV2,
    VisibilityGatedCrossviewResidualPrincipalPointV2,
)


@pytest.fixture
def cameras():
    """Simple circular rig of pinhole cameras."""
    import numpy as np
    from motionflow_mv.calibration.camera import Camera

    cams = []
    for i in range(4):
        theta = 2 * np.pi * i / 4
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cams.append(Camera(K=K, R=R, t=t))
    return cams


def test_visibility_gate_head_shape_and_range():
    """VisibilityGateHead should return the correct shape and bounded values."""
    N, V, J, d = 6, 4, 17, 32
    gate = VisibilityGatedFusionV2(d=d, n_views=V)
    feat = torch.randn(N, V, J, d)
    confidences = torch.rand(N, V, J)

    vis = gate(feat, confidences)
    assert vis.shape == (N, V, J)
    assert torch.all((vis >= 0) & (vis <= 1))


def test_visibility_gate_with_context_and_uncertainty():
    """VisibilityGateHead supports context and uncertainty modes."""
    N, V, J, d = 2, 4, 17, 32
    gate = VisibilityGatedFusionV2(
        d=d,
        n_views=V,
        use_context=True,
        use_uncertainty=True,
    )
    feat = torch.randn(N, V, J, d)
    confidences = torch.rand(N, V, J)

    vis, logits = gate(feat, confidences, return_logits=True)
    assert vis.shape == (N, V, J)
    assert logits.shape == (N, V, J)


@pytest.mark.parametrize("use_context", [False, True])
def test_visibility_gate_gradient_flow(use_context):
    """Visibility gate should be differentiable."""
    N, V, J, d = 2, 4, 17, 32
    gate = VisibilityGatedFusionV2(d=d, n_views=V, use_context=use_context)
    feat = torch.randn(N, V, J, d, requires_grad=True)
    confidences = torch.rand(N, V, J)

    vis = gate(feat, confidences)
    loss = vis.sum()
    loss.backward()

    assert feat.grad is not None
    assert not torch.all(feat.grad == 0)


def test_visibility_gate_fallback_guard():
    """Fallback guard should activate when too few views are visible."""
    N, V, J, d = 2, 4, 17, 32
    gate = VisibilityGatedFusionV2(
        d=d,
        n_views=V,
        min_visible_views=2,
        visibility_threshold=0.5,
    )
    # Make features very negative so sigmoid -> 0 for all views.
    feat = torch.full((N, V, J, d), -10.0)
    confidences = torch.ones(N, V, J)

    vis = gate(feat, confidences)
    # Since all visibilities are below threshold and count < 2, fallback should
    # restore all views.
    per_joint = vis.sum(dim=1)  # (N, J)
    assert torch.all(per_joint > 0)


def test_crossview_v2_model_forward(cameras):
    """Cross-view v2 model should produce correct output shapes."""
    B, T, V, J = 2, 5, 4, 17
    model = VisibilityGatedCrossviewResidualPrincipalPointV2(j=J, d=64, n_views=V)
    x = torch.rand(B, T, V, J, 3)

    pred, weights, visibility = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert visibility.shape == (B, T, V, J)


def test_crossview_v2_model_single_frame(cameras):
    """Cross-view v2 model should handle single-frame (4D) inputs."""
    B, V, J = 2, 4, 17
    model = VisibilityGatedCrossviewResidualPrincipalPointV2(j=J, d=64, n_views=V)
    x = torch.rand(B, V, J, 3)

    pred, weights, visibility = model(x, cameras=cameras)
    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)
    assert visibility.shape == (B, V, J)


def test_crossview_v2_model_backward(cameras):
    """Cross-view v2 model should allow gradient propagation."""
    B, T, V, J = 1, 3, 4, 17
    model = VisibilityGatedCrossviewResidualPrincipalPointV2(j=J, d=32, n_views=V)
    x = torch.rand(B, T, V, J, 3, requires_grad=True)

    pred, weights, visibility = model(x, cameras=cameras)
    loss = pred.mean()
    loss.backward()

    assert x.grad is not None
    assert any(p.grad is not None for p in model.parameters())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
