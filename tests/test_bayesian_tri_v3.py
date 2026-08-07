"""CPU smoke tests for Bayesian triangulation v3."""

import numpy as np
import torch

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.prototypes.bayesian_tri_v3_model import (
    RayAttentionFusionModelBayesianTriV3,
)


def _make_cameras(n_views: int = 4):
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


def test_bayesian_tri_v3_forward_backward():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelBayesianTriV3(
        j=J,
        d=64,
        n_views=V,
        gn_iters=2,
        epipolar_loss_weight=0.05,
        return_pp_delta=True,
    )
    pred, weights, pp_delta, L3d, epi_loss = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)
    assert L3d.shape == (B, T, J, 3, 3)
    assert epi_loss.shape == ()
    loss = pred.mean() + 0.0 * epi_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_bayesian_tri_v3_joint_precision_is_spd():
    B, T, V, J = 1, 3, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelBayesianTriV3(
        j=J, d=32, n_views=V, gn_iters=1, epipolar_loss_weight=0.0
    )
    with torch.no_grad():
        _, _, L3d, _ = model(x, cameras=cameras)
    # Build information matrix from Cholesky factor and check positive definiteness.
    Lambda = torch.matmul(L3d, L3d.transpose(-2, -1))
    eig = torch.linalg.eigvalsh(Lambda)
    assert (eig > 0).all()


def test_bayesian_tri_v3_squeeze_output():
    V, J = 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(2, V, J, 3)
    model = RayAttentionFusionModelBayesianTriV3(
        j=J, d=32, n_views=V, gn_iters=1, epipolar_loss_weight=0.0
    )
    pred, weights, L3d, epi_loss = model(x, cameras=cameras)
    assert pred.shape == (2, J, 3)
    assert weights.shape == (2, V, J)
    assert L3d.shape == (2, J, 3, 3)
    assert epi_loss.shape == ()


if __name__ == "__main__":
    test_bayesian_tri_v3_forward_backward()
    test_bayesian_tri_v3_joint_precision_is_spd()
    test_bayesian_tri_v3_squeeze_output()
    print("Bayesian tri v3 tests passed")
