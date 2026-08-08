"""Tests for v29 Self-Evolving Hierarchical Multi-View Fusion."""

import torch
import pytest

from motionflow_mv.fusion.self_evolving_hierarchical_multiview_v29 import (
    HierarchicalViewEncoderV29,
    PhysicalSpaceTemporalLossV29,
    TestTimeSelfEvolutionV29,
)


def test_hierarchical_view_encoder_shape_and_identity_init():
    B, T, V, J, d = 2, 3, 4, 17, 64
    tokens = torch.rand(B, T, V, J, d)
    encoder = HierarchicalViewEncoderV29(d=d, n_heads=4, n_views=V)
    out = encoder(tokens)
    assert out.shape == (B, T, V, J, d)
    # Identity at init: output should be close to zero because projections are zeroed.
    assert out.abs().mean().item() < 1e-4


def test_hierarchical_view_encoder_with_view_mask():
    B, T, V, J, d = 2, 3, 4, 17, 64
    tokens = torch.rand(B, T, V, J, d)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    view_mask[:, :, 3] = False
    encoder = HierarchicalViewEncoderV29(d=d, n_heads=4, n_views=V)
    out = encoder(tokens, view_mask=view_mask)
    assert out.shape == (B, T, V, J, d)


def test_test_time_self_evolution_v29():
    B, T, V, J = 2, 3, 4, 17
    pred_3d = torch.rand(B, T, J, 3) * 2.0 - 1.0
    points_2d = torch.rand(B, T, V, J, 2) * 500.0 + 200.0
    K = torch.eye(3).view(1, 1, 1, 3, 3).expand(B, T, V, -1, -1).clone().float()
    K[:, :, :, 0, 2] = 320.0
    K[:, :, :, 1, 2] = 240.0
    # Circular camera rig pointing at origin.
    import math
    Rs = []
    ts = []
    for i in range(V):
        theta = 2 * math.pi * i / V
        c = torch.tensor([3.0 * math.cos(theta), 3.0 * math.sin(theta), 1.0], dtype=torch.float32)
        forward = -c / c.norm()
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        right = torch.cross(forward, up)
        right /= right.norm()
        up = torch.cross(right, forward)
        R = torch.stack([right, up, -forward], dim=0)
        Rs.append(R)
        ts.append(-R @ c)
    R = torch.stack(Rs, dim=0).view(1, 1, V, 3, 3).expand(B, T, -1, -1, -1).float()
    t = torch.stack(ts, dim=0).view(1, 1, V, 3).expand(B, T, -1, -1).float()
    tte = TestTimeSelfEvolutionV29(
        n_iters=2,
        use_physical_space_alignment=True,
        j=J,
    )
    refined = tte(pred_3d, points_2d, K, R, t)
    assert refined.shape == (B, T, J, 3)


def test_physical_space_temporal_loss_v29():
    B, T, J = 2, 4, 17
    X = torch.rand(B, T, J, 3)
    parents = [-1, 0, 1, 2, 0, 3, 4, 0, 7, 8, 8, 8, 10, 11, 10, 13, 14]
    foot_indices = [5, 6, 15, 16]
    loss_fn = PhysicalSpaceTemporalLossV29(
        floor_loss_weight=0.01,
        bone_temporal_weight=0.01,
        com_jitter_weight=0.001,
        foot_joint_indices=foot_indices,
        parents=parents,
    )
    loss, terms = loss_fn(X)
    assert loss.numel() == 1
    assert "floor" in terms
    assert "bone_temporal" in terms
    assert "com_jitter" in terms
