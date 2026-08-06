"""Sanity tests for the hierarchical attention + entropy regularisation model."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_hierarchical_attention_entropy_reg_model import (
    RayAttentionFusionModelHierarchicalAttentionEntropyReg,
)


def _make_cameras(n_views: int = 4):
    import numpy as np
    from motionflow_mv.calibration.camera import Camera

    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
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
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def test_hierarchical_attention_entropy_reg_forward_shape_and_grad():
    B, T, V, J = 2, 7, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelHierarchicalAttentionEntropyReg(
        j=J, d=32, n_views=V, n_view_groups=2, attention_entropy_weight=0.01,
        return_pp_delta=True,
    )
    pred, weights, pp_delta, entropy_loss = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)
    assert entropy_loss.shape == ()

    loss = pred.mean() + entropy_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_hierarchical_attention_entropy_reg_single_frame():
    B, V, J = 2, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)

    model = RayAttentionFusionModelHierarchicalAttentionEntropyReg(
        j=J, d=32, n_views=V, n_view_groups=2, attention_entropy_weight=0.01,
        return_pp_delta=True,
    )
    pred, weights, pp_delta, entropy_loss = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)
    assert pp_delta.shape == (B, V, 2)
    assert entropy_loss.shape == ()


def test_entropy_regularization_decreases_with_concentrated_weights():
    model = RayAttentionFusionModelHierarchicalAttentionEntropyReg(
        j=17, d=32, n_views=4, attention_entropy_weight=1.0
    )
    # One-hot-like weights should have lower entropy than uniform weights.
    concentrated = torch.zeros(2, 4, 17)
    concentrated[:, 0, :] = 1.0
    uniform = torch.ones(2, 4, 17)

    ent_conc = model._entropy_regularization(concentrated)
    ent_uniform = model._entropy_regularization(uniform)
    assert ent_conc < ent_uniform


if __name__ == "__main__":
    test_hierarchical_attention_entropy_reg_forward_shape_and_grad()
    test_hierarchical_attention_entropy_reg_single_frame()
    test_entropy_regularization_decreases_with_concentrated_weights()
    print("hierarchical attention entropy-reg tests passed")
