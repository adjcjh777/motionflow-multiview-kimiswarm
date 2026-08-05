"""CPU-only forward sanity tests for the GraphJointRelation PP model.

This script does **not** start training; it only checks that the model can be
instantiated, run a forward pass, and produce gradients.  GPU training is queued
and must be run separately once the RTX 4090 is free.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_graph_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_model import _make_cameras


def test_graph_pp_forward_28j():
    B, T, V, J = 2, 3, 4, 28
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph(
        j=J,
        d=64,
        n_views=V,
        n_st_layers=1,
        graph_num_layers=1,
    )
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_graph_pp_forward_17j():
    B, T, V, J = 1, 2, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph(
        j=J,
        d=64,
        n_views=V,
        n_st_layers=1,
        graph_num_layers=1,
    )
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)


def test_graph_pp_single_frame():
    B, V, J = 2, 4, 28
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph(
        j=J,
        d=64,
        n_views=V,
        n_st_layers=1,
        graph_num_layers=1,
    )
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)


def test_graph_pp_shared_vs_separate_layers():
    """Ensure shared-weight and separate-layer variants build correctly."""
    B, T, V, J = 1, 2, 4, 28
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    for share in (True, False):
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph(
            j=J,
            d=64,
            n_views=V,
            n_st_layers=1,
            graph_num_layers=2,
            graph_share_weights=share,
        )
        pred, _ = model(x, cameras=cameras)
        assert pred.shape == (B, T, J, 3)


if __name__ == "__main__":
    test_graph_pp_forward_28j()
    test_graph_pp_forward_17j()
    test_graph_pp_single_frame()
    test_graph_pp_shared_vs_separate_layers()
    print("GraphJointRelation PP tests passed")
