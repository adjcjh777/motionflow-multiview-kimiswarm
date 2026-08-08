#!/usr/bin/env python
"""CPU smoke test for the v26 camera calibration refinement sub-module.

Generates a toy multi-view scene, perturbs the cameras, and checks that the
refinement module can reduce the reprojection error when its residual gate is
opened. No GPU or network access is required.
"""

import os
import sys

# Safe on the Anaconda/MKL stack used in this repo.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from motionflow_mv.calibration.camera_refinement_v26 import (
    CameraRefinementV26,
    _reprojection_loss,
)
from motionflow_mv.calibration.perturb import perturb_cameras


def _make_scene(B=2, T=3, V=4, J=17):
    torch.manual_seed(0)
    X = torch.randn(B, T, J, 3) * 0.3
    X[..., 2] = X[..., 2].abs() + 2.5

    K = torch.zeros(B, T, V, 3, 3)
    K[..., 0, 0] = 800.0
    K[..., 1, 1] = 800.0
    K[..., 0, 2] = 320.0
    K[..., 1, 2] = 240.0
    K[..., 2, 2] = 1.0

    angles = torch.linspace(-0.6, 0.6, V)
    R = torch.eye(3).view(1, 1, 1, 3, 3).expand(B, T, V, 3, 3).clone()
    t = torch.zeros(B, T, V, 3)
    for i, a in enumerate(angles):
        ca, sa = a.cos(), a.sin()
        cam_pos = torch.tensor([sa * 4.0, 0.0, ca * 4.0])
        forward = -cam_pos / cam_pos.norm()
        world_up = torch.tensor([0.0, 1.0, 0.0])
        right = torch.cross(forward, world_up, dim=-1)
        right = right / right.norm()
        up = torch.cross(right, forward, dim=-1)
        R[:, :, i] = torch.stack([right, up, forward], dim=0)
        t[:, :, i] = -torch.matmul(R[:, :, i], cam_pos)

    # Project with only tiny noise so the error is dominated by camera miscalibration.
    X_cam = torch.matmul(R.unsqueeze(3), X.unsqueeze(2).unsqueeze(-1)).squeeze(-1) + t.unsqueeze(3)
    proj = torch.matmul(K.unsqueeze(3), X_cam.unsqueeze(-1)).squeeze(-1)
    points_2d = proj[..., :2] / proj[..., 2:3]
    points_2d = points_2d + torch.randn_like(points_2d) * 0.1
    return X, points_2d, K, R, t


def main():
    device = torch.device("cpu")
    X, points_2d, K_gt, R_gt, t_gt = _make_scene()
    weights = torch.ones(*points_2d.shape[:4])

    # Perturb the ground-truth cameras.
    K_pert, R_pert, t_pert = perturb_cameras(
        K_gt,
        R_gt,
        t_gt,
        rot_std=2.5,
        trans_std=0.20,
        focal_std=0.05,
        pp_std=10.0,
    )

    initial_loss = _reprojection_loss(
        X, points_2d, K_pert, R_pert, t_pert, weights
    ).item()

    module = CameraRefinementV26(n_steps=5, lr=0.1).to(device)
    # Open the gate to let the geometric correction flow through.
    module.residual_scale.data.fill_(3.0)

    K_ref, R_ref, t_ref = module(points_2d, X, K_pert, R_pert, t_pert, weights)

    refined_loss = _reprojection_loss(
        X, points_2d, K_ref.detach(), R_ref.detach(), t_ref.detach(), weights
    ).item()

    print("=== Camera Refinement v26 CPU smoke test ===")
    print(f"Initial reprojection loss: {initial_loss:.6f}")
    print(f"Refined reprojection loss: {refined_loss:.6f}")
    print(f"Output shapes: K={tuple(K_ref.shape)}, R={tuple(R_ref.shape)}, t={tuple(t_ref.shape)}")

    assert K_ref.shape == K_gt.shape
    assert R_ref.shape == R_gt.shape
    assert t_ref.shape == t_gt.shape
    assert refined_loss < initial_loss
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
