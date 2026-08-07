"""Batch routing for the multi-person association graph."""

import torch

from motionflow_mv.fusion.multiperson_association_graph import MultiPersonAssociationGraph
from motionflow_mv.fusion.ray_attention_temporal_crossview_model import _make_cameras
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_multiperson_assoc_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointMultiPersonAssoc,
)


def test_batched_graph_matches_independent_second_sample():
    torch.manual_seed(0)
    graph = MultiPersonAssociationGraph(
        d=4,
        n_views=1,
        n_persons=1,
        num_layers=1,
    ).eval()
    x = torch.randn(2, 1, 1, 2, 4)

    with torch.no_grad():
        batched = graph(x, parents=[-1, 0], symmetry_pairs=[])
        independent = graph(x[1:], parents=[-1, 0], symmetry_pairs=[])

    torch.testing.assert_close(batched[1], independent[0])


def test_zero_layer_model_matches_independent_people_and_restores_aux_axes():
    torch.manual_seed(1)
    B, T, V, P, J = 1, 2, 2, 2, 17
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointMultiPersonAssoc(
        j=J,
        d=16,
        n_views=V,
        n_persons=P,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        focal_max_scale=0.1,
        return_pp_delta=True,
        assoc_num_layers=0,
    ).eval()
    x = torch.rand(B, T, V, P, J, 3)
    cameras = _make_cameras(V)

    with torch.no_grad():
        pred, weights, pp_delta, focal_scale = model(x, cameras=cameras)
        independent = [model(x[:, :, :, p], cameras=cameras) for p in range(P)]

    assert pp_delta.shape == (B, T, V, P, 2)
    assert focal_scale.shape == (B, T, V, P)
    for p, single in enumerate(independent):
        torch.testing.assert_close(pred[:, :, p], single[0], atol=1e-4, rtol=2e-3)
        torch.testing.assert_close(weights[:, :, :, p], single[1])
        torch.testing.assert_close(pp_delta[:, :, :, p], single[2].view(B, T, V, 2))
        torch.testing.assert_close(focal_scale[:, :, :, p], single[3].view(B, T, V))

    model.return_pp_delta = False
    model.return_visibility = True
    with torch.no_grad():
        _, _, visibility = model(x, cameras=cameras)
    assert visibility.shape == (B, T, V, P, J)
