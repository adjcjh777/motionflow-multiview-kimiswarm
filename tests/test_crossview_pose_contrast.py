"""Smoke tests for the cross-view contrastive pose representation extension."""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from motionflow_mv.fusion.ray_attention_temporal_crossview_model import _make_cameras
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_crossview_contrast_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCrossViewContrast,
)
from motionflow_mv.losses.crossview_pose_contrast import CrossViewJointContrastiveLoss


def test_loss_shape_and_grad():
    N, V, J, d = 2, 4, 17, 32
    feat = torch.randn(N, V, J, d, requires_grad=True)
    loss_fn = CrossViewJointContrastiveLoss(d=d, projection_dim=16)
    loss = loss_fn(feat)
    assert loss.shape == ()
    assert loss.item() >= 0.0
    loss.backward()
    assert feat.grad is not None


def test_model_forward_and_contrastive_loss():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCrossViewContrast(
        j=J,
        d=32,
        n_views=V,
        n_st_layers=2,
        contrastive_dim=16,
        contrastive_loss_weight=0.1,
    )
    model.eval()

    pred, weights, c_loss = model.forward_with_contrastive_loss(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert c_loss.shape == ()
    assert c_loss.item() >= 0.0

    # Verify that the model can still be trained end-to-end.
    model.train()
    pred, weights, c_loss = model.forward_with_contrastive_loss(x, cameras=cameras)
    total = (pred ** 2).mean() + c_loss
    total.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_compute_contrastive_loss_matches_hook_path():
    """The dedicated feature path and the hook-based path should agree."""
    B, T, V, J = 1, 3, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCrossViewContrast(
        j=J,
        d=32,
        n_views=V,
        n_st_layers=1,
        contrastive_dim=16,
        contrastive_loss_weight=1.0,
    )
    model.eval()

    with torch.no_grad():
        c_loss_explicit = model.compute_contrastive_loss(x, cameras=cameras)
        _, _, c_loss_hook = model.forward_with_contrastive_loss(x, cameras=cameras)

    # Both paths use the same feature extractor; the scalar loss values should be
    # identical up to numerical jitter because the model is in eval mode.
    assert torch.allclose(c_loss_explicit, c_loss_hook, atol=1e-5, rtol=1e-4)


def test_single_frame_batch_one_keeps_contrastive_sample_axis():
    B, V, J = 1, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCrossViewContrast(
        j=J,
        d=32,
        n_views=V,
        n_st_layers=0,
        contrastive_dim=16,
    ).eval()

    with torch.no_grad():
        feat = model._prepare_contrastive_features(x, cameras=cameras)
        loss = model.compute_contrastive_loss(x, cameras=cameras)

    assert feat.shape == (B, V, J, 32)
    assert loss.shape == ()


if __name__ == "__main__":
    test_loss_shape_and_grad()
    test_model_forward_and_contrastive_loss()
    test_compute_contrastive_loss_matches_hook_path()
    print("cross-view contrastive tests passed")
