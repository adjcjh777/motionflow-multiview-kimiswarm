"""Single-frame output shapes for wrappers that delegate to PP models."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionflow_mv.fusion.ray_attention_temporal_crossview_model import _make_cameras
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_visibility_model import (
    RayAttentionFusionModelBayesianTriV2Visibility,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bundle_adjustment_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBundleAdjustment,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_kinematic_chain_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointKinematicChain,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_multiperson_assoc_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointMultiPersonAssoc,
)


def _input(batch=1, views=2, joints=17):
    x = torch.rand(batch, views, joints, 3)
    return x, _make_cameras(views)


def _common(views=2, joints=17):
    return dict(
        j=joints,
        d=16,
        n_views=views,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
    )


def test_bundle_adjustment_preserves_single_frame_parent_shapes():
    B, V, J = 1, 2, 17
    x, cameras = _input(B, V, J)
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBundleAdjustment(
        **_common(V, J),
        focal_max_scale=0.1,
        return_pp_delta=True,
        return_raw=True,
        dba_iters=0,
    ).eval()

    with torch.no_grad():
        pred, weights, pp_delta, focal_scale, raw = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)
    assert pp_delta.shape == (B, V, 2)
    assert focal_scale.shape == (B, V)
    assert raw.shape == (B, J, 3)


def test_bayesian_visibility_preserves_single_frame_parent_shapes():
    B, V, J = 1, 2, 17
    x, cameras = _input(B, V, J)
    model = RayAttentionFusionModelBayesianTriV2Visibility(
        **_common(V, J),
        focal_max_scale=0.1,
        return_pp_delta=True,
        return_covariance=True,
        return_raw=True,
        covariance_hidden=8,
        visibility_hidden=8,
        gn_iters=0,
    ).eval()

    with torch.no_grad():
        pred, weights, pp_delta, focal_scale, covariance, raw, visibility, epi_loss = model(
            x, cameras=cameras
        )

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)
    assert pp_delta.shape == (B, V, 2)
    assert focal_scale.shape == (B, V)
    assert covariance.shape == (B, V, J, 2, 2)
    assert raw.shape == (B, J, 3)
    assert visibility.shape == (B, V, J)
    assert epi_loss.shape == ()


def test_kinematic_chain_preserves_single_frame_parent_shapes():
    B, V, J = 1, 2, 17
    x, cameras = _input(B, V, J)
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointKinematicChain(
        **_common(V, J),
        focal_max_scale=0.1,
        return_pp_delta=True,
        kc_hidden_dim=8,
        kc_num_layers=0,
    ).eval()

    with torch.no_grad():
        pred, weights, pp_delta, focal_scale = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)
    assert pp_delta.shape == (B, V, 2)
    assert focal_scale.shape == (B, V)


def test_multiperson_wrapper_delegates_single_person_frame_unchanged():
    B, V, J = 1, 2, 17
    x, cameras = _input(B, V, J)
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointMultiPersonAssoc(
        **_common(V, J),
        n_persons=1,
        assoc_num_layers=0,
        focal_max_scale=0.1,
        return_pp_delta=True,
    ).eval()

    with torch.no_grad():
        pred, weights, pp_delta, focal_scale = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)
    assert pp_delta.shape == (B, V, 2)
    assert focal_scale.shape == (B, V)
