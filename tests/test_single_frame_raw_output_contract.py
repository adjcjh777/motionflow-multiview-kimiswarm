"""Single-frame raw 3D outputs follow the public no-time-axis contract."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model import (
    RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint,
)
from motionflow_mv.fusion.action_aware_principal_point_model import (
    ActionAwareRayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_camera_conditioned_model import (
    RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_camera_centric_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCameraCentric,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_canonical_skeleton_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCanonicalSkeleton,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_completion_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCompletion,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_epipolar_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_splat_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplat,
)
from motionflow_mv.fusion.semantic_action_conditional_fusion_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSemanticActionConditional,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_visibility_transformer_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibilityTransformer,
)


MODEL_FACTORIES = [
    lambda common: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(**common),
    lambda common: ActionAwareRayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        **common
    ),
    lambda common: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSemanticActionConditional(
        **common
    ),
    lambda common: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCameraCentric(
        **common, camera_centric_hidden=8
    ),
    lambda common: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCanonicalSkeleton(
        **common, graph_num_layers=0
    ),
    lambda common: RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint(
        **common, n_view_layers=0, n_temporal_layers=0, n_joint_graph_layers=0
    ),
    lambda common: RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned(**common),
    lambda common: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2(
        **common
    ),
    lambda common: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar(**common),
    lambda common: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplat(**common),
    lambda common: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibilityTransformer(
        **common, visibility_n_layers=1, visibility_n_heads=4
    ),
    lambda common: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCompletion(**common),
]


def _inputs(batch=1, views=2, joints=17):
    K = torch.eye(3).view(1, 1, 3, 3).repeat(batch, views, 1, 1)
    R = torch.eye(3).view(1, 1, 3, 3).repeat(batch, views, 1, 1)
    centers = torch.zeros(batch, views, 3)
    centers[:, 1, 0] = 1.0
    t = -centers
    points = torch.randn(batch, joints, 3) * 0.1
    points[..., 2] += 5.0
    camera = torch.einsum("bvij,bkj->bvki", R, points) + t[:, :, None, :]
    image = torch.einsum("bvij,bvkj->bvki", K, camera)
    xy = image[..., :2] / image[..., 2:3]
    x = torch.cat([xy, torch.ones(batch, views, joints, 1)], dim=-1)
    return x, K, R, t


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_single_frame_raw_output_has_no_time_axis(factory):
    common = dict(
        j=17,
        d=16,
        n_views=2,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        max_temporal_len=4,
        residual_hidden=16,
        principal_point_hidden=8,
        return_pp_delta=True,
        return_raw=True,
    )
    model = factory(common).eval()
    x, K, R, t = _inputs()

    with torch.no_grad():
        output = model(x, K=K, R=R, t=t)
    pred, weights, raw_3d = output[0], output[1], output[-1]

    assert pred.shape == (1, 17, 3)
    assert weights.shape == (1, 2, 17)
    assert raw_3d.shape == (1, 17, 3)


def test_base_raw_only_single_frame_has_no_time_axis():
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=17,
        d=16,
        n_views=2,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        return_raw=True,
    ).eval()
    x, K, R, t = _inputs()

    with torch.no_grad():
        pred, weights, raw_3d = model(x, K=K, R=R, t=t)

    assert pred.shape == (1, 17, 3)
    assert weights.shape == (1, 2, 17)
    assert raw_3d.shape == (1, 17, 3)


def test_sequence_and_pp_tuple_keep_their_existing_axes_and_order():
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=17,
        d=16,
        n_views=2,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        focal_max_scale=0.1,
        return_pp_delta=True,
        return_raw=True,
    ).eval()
    x, K, R, t = _inputs()

    with torch.no_grad():
        single = model(x, K=K, R=R, t=t)
        pred, weights, pp_delta, focal_scale, raw_3d = model(
            x[:, None].expand(-1, 2, -1, -1, -1), K=K, R=R, t=t
        )

    assert single[0].shape == (1, 17, 3)
    assert single[1].shape == (1, 2, 17)
    assert single[2].shape == (1, 2, 2)
    assert single[3].shape == (1, 2)
    assert single[4].shape == (1, 17, 3)
    assert pred.shape == (1, 2, 17, 3)
    assert weights.shape == (1, 2, 2, 17)
    assert pp_delta.shape == (2, 2, 2)
    assert focal_scale.shape == (2, 2)
    assert raw_3d.shape == (1, 2, 17, 3)
