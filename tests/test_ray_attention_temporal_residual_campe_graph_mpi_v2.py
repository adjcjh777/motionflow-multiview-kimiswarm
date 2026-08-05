"""Smoke tests for RayAttentionFusionModelTemporalResidualCamPEGraphMPIV2."""

import torch

from motionflow_mv.fusion.ray_attention_temporal_residual_campe_graph_mpi_v2_model import (
    RayAttentionFusionModelTemporalResidualCamPEGraphMPIV2,
    _make_cameras,
)


def test_forward_backward_28_joints():
    B, T, V, J = 2, 5, 4, 28
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalResidualCamPEGraphMPIV2(j=J, d=64, n_views=V)
    pred, w = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_single_frame_28_joints():
    B, V, J = 2, 4, 28
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)
    model = RayAttentionFusionModelTemporalResidualCamPEGraphMPIV2(j=J, d=64, n_views=V)
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, J, 3)
    assert w.shape == (B, V, J)


def test_skeleton_losses():
    B, T, V, J = 2, 5, 4, 28
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalResidualCamPEGraphMPIV2(j=J, d=64, n_views=V)
    pred, _ = model(x, cameras=cameras)
    y = torch.rand_like(pred)

    bl = model.bone_length_loss(pred, y)
    sym = model.symmetry_loss(pred)
    assert bl.item() >= 0
    assert sym.item() >= 0


if __name__ == "__main__":
    test_forward_backward_28_joints()
    print("test_forward_backward_28_joints passed")
    test_single_frame_28_joints()
    print("test_single_frame_28_joints passed")
    test_skeleton_losses()
    print("test_skeleton_losses passed")
    print("All MPI Graph v2 smoke tests passed")
