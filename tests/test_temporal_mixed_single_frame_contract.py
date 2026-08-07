"""Single-frame shapes for mixed-dataset temporal fusion models."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionflow_mv.fusion.ray_attention_temporal_mixed_residual_principal_point_model import (
    RayAttentionFusionModelTemporalMixedResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_mixed_residual_v1 import (
    RayAttentionFusionModelTemporalMixedResidual,
)
from motionflow_mv.fusion.ray_attention_temporal_model_mixed_v1 import (
    RayAttentionFusionModelTemporalMixed,
)


def _inputs(batch=1, views=14, joints=28):
    x = torch.rand(batch, views, joints, 3)
    K = torch.eye(3).view(1, 1, 3, 3).repeat(batch, views, 1, 1)
    R = torch.eye(3).view(1, 1, 3, 3).repeat(batch, views, 1, 1)
    t = torch.zeros(batch, views, 3)
    t[:, :, 0] = -torch.arange(views)
    dataset_ids = torch.zeros(batch, dtype=torch.long)
    return x, K, R, t, dataset_ids


@pytest.mark.parametrize(
    "model",
    [
        RayAttentionFusionModelTemporalMixed(d=8, n_temporal_layers=0, max_temporal_len=2),
        RayAttentionFusionModelTemporalMixedResidual(
            d=8, n_temporal_layers=0, max_temporal_len=2, residual_hidden=8
        ),
    ],
)
def test_mixed_model_single_frame_outputs_have_no_time_axis(model):
    x, K, R, t, dataset_ids = _inputs()

    with torch.no_grad():
        pred, mask = model.eval()(x, K, R, t, dataset_ids)

    assert pred.shape == (1, 28, 3)
    assert mask.shape == (1, 28)


def test_mixed_pp_single_frame_preserves_view_axis():
    x, K, R, t, dataset_ids = _inputs()
    model = RayAttentionFusionModelTemporalMixedResidualPrincipalPoint(
        d=8,
        n_temporal_layers=0,
        max_temporal_len=2,
        residual_hidden=8,
        principal_point_hidden=8,
        focal_max_scale=0.1,
        return_pp_delta=True,
    ).eval()

    with torch.no_grad():
        pred, mask, pp_delta, focal_scale = model(x, K, R, t, dataset_ids)

    assert pred.shape == (1, 28, 3)
    assert mask.shape == (1, 28)
    assert pp_delta.shape == (1, 14, 2)
    assert focal_scale.shape == (1, 14)
