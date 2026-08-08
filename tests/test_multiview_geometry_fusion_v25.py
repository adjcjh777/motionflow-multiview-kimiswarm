"""Unit tests for motionflow_mv.fusion.multiview_geometry_fusion_v25."""

import pytest
import torch

from motionflow_mv.fusion.multiview_geometry_fusion_v25 import (
    MultiViewGeometryFusionV25,
)


def _random_cameras(batch: tuple = (2, 3, 4), device: str = "cpu"):
    """Return a deterministic set of calibrated cameras and 2D observations."""
    B, T, V = batch
    K = torch.eye(3, device=device).view(1, 1, 1, 3, 3).expand(B, T, V, 3, 3).clone()
    K[..., 0, 0] = 800.0
    K[..., 1, 1] = 800.0
    K[..., 0, 2] = 320.0
    K[..., 1, 2] = 240.0

    # Identity rotations keep the geometry well-defined and simple.
    R = torch.eye(3, device=device).view(1, 1, 1, 3, 3).expand(B, T, V, 3, 3).clone()
    # Cameras placed in front of the subject.
    t = torch.zeros(B, T, V, 3, device=device)
    t[..., 2] = 5.0
    t[..., :2] = torch.randn(B, T, V, 2, device=device) * 0.5

    return K, R, t


def _random_2d(B: int, T: int, V: int, J: int, device: str = "cpu"):
    """Random 2D keypoints with per-joint confidence as third channel."""
    points = torch.randn(B, T, V, J, 2, device=device) * 100 + 320
    conf = torch.rand(B, T, V, J, device=device)
    return torch.cat([points, conf[..., None]], dim=-1)


def test_forward_shape():
    B, T, V, J, d = 2, 3, 4, 17, 128
    points_2d = _random_2d(B, T, V, J)
    K, R, t = _random_cameras((B, T, V))
    pred_3d_init = torch.randn(B, T, J, 3)

    module = MultiViewGeometryFusionV25(d=d, n_heads=4, n_views=V, n_geometry_layers=2)
    out = module(points_2d, K, R, t, pred_3d_init=pred_3d_init)
    assert out.shape == (B, T, J, 3)


def test_forward_with_view_mask():
    B, T, V, J = 2, 3, 4, 17
    points_2d = _random_2d(B, T, V, J)
    K, R, t = _random_cameras((B, T, V))
    pred_3d_init = torch.randn(B, T, J, 3)
    view_mask = torch.tensor([True, True, False, True]).view(1, 1, V).expand(B, T, V)

    module = MultiViewGeometryFusionV25(d=64, n_heads=2, n_views=V)
    out = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert out.shape == (B, T, J, 3)


def test_identity_at_init():
    """With zero effective weights, the module should act as a no-op."""
    B, T, V, J = 2, 3, 4, 17
    points_2d = _random_2d(B, T, V, J)
    K, R, t = _random_cameras((B, T, V))
    pred_3d_init = torch.randn(B, T, J, 3)

    module = MultiViewGeometryFusionV25(d=64, n_heads=2, n_views=V)
    # Zero the residual gate; the module now returns the input estimate.
    module.depth_tri_head.residual_scale.data.zero_()
    # Ensure attention residual is also zeroed (already initialised to zero).
    if module.use_geometry_attention:
        for layer in module.geom_attn_layers:
            for p in layer.out_proj.parameters():
                p.data.zero_()

    out = module(points_2d, K, R, t, pred_3d_init=pred_3d_init)
    assert torch.allclose(out, pred_3d_init, atol=1e-5)


def test_gradient_flow():
    B, T, V, J = 2, 3, 4, 17
    points_2d = _random_2d(B, T, V, J)
    points_2d.requires_grad_(True)
    K, R, t = _random_cameras((B, T, V))
    K = K.clone().requires_grad_(True)
    R = R.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    pred_3d_init = torch.randn(B, T, J, 3, requires_grad=True)

    module = MultiViewGeometryFusionV25(d=64, n_heads=2, n_views=V)
    # Open the residual gate so gradients can propagate through the depth head.
    module.depth_tri_head.residual_scale.data.fill_(1.0)

    out = module(points_2d, K, R, t, pred_3d_init=pred_3d_init)
    loss = out.sum()
    loss.backward()

    for tensor in (points_2d, K, R, t, pred_3d_init):
        assert tensor.grad is not None, f"{tensor.shape} has no gradient"
        assert torch.isfinite(tensor.grad).all()


def test_output_for_different_skeleton_sizes():
    B, T, V, d = 2, 3, 4, 64
    for J in (17, 28):
        points_2d = _random_2d(B, T, V, J)
        K, R, t = _random_cameras((B, T, V))
        pred = torch.randn(B, T, J, 3)
        module = MultiViewGeometryFusionV25(d=d, n_heads=2, n_views=V)
        out = module(points_2d, K, R, t, pred_3d_init=pred)
        assert out.shape == (B, T, J, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
