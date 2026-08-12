#!/usr/bin/env python3
"""CPU-only smoke test for visibility-aware adaptive fusion.

Generates a synthetic 4-view, 17-joint sequence, corrupts a subset of views,
and trains the visibility-gated PP model for a few iterations.  Reports whether
the learned visibility mask correctly identifies the corrupted views and
whether using the mask lowers 3D error.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_visibility_model import (  # noqa: E501
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility,
)


def make_circular_rig(n_views=4, radius=5.0, focal=200.0):
    """Return a simple circular camera rig (K, R, t)."""
    K_list = []
    for _ in range(n_views):
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = focal
        K_list.append(K)

    R_list, t_list = [], []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([radius * np.cos(theta), radius * np.sin(theta), 1.0], dtype=np.float64)
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        R_list.append(R)
        t_list.append(t)

    return (
        torch.from_numpy(np.stack(K_list, 0)).float(),
        torch.from_numpy(np.stack(R_list, 0)).float(),
        torch.from_numpy(np.stack(t_list, 0)).float(),
    )


def project(points_3d, K, R, t):
    """Project 3D world points to 2D for a circular rig.

    Args:
        points_3d: (B, T, J, 3) tensor.
        K: (V, 3, 3) intrinsics.
        R: (V, 3, 3) rotations.
        t: (V, 3) translation.

    Returns:
        (B, T, V, J, 2) projected 2D points.
    """
    B, T, J, _ = points_3d.shape
    V = K.shape[0]
    X = points_3d[:, :, None, :, :, None]  # (B, T, 1, J, 3, 1)
    R = R.to(points_3d.device).view(1, 1, V, 1, 3, 3)
    t_ = t.to(points_3d.device).view(1, 1, V, 1, 3, 1)
    Xcam = torch.matmul(R, X).squeeze(-1) + t_.squeeze(-1)  # (B, T, V, J, 3)
    x = Xcam[..., 0] / Xcam[..., 2]
    y = Xcam[..., 1] / Xcam[..., 2]
    fx = K[:, 0, 0].view(1, 1, V, 1).to(points_3d.device)
    fy = K[:, 1, 1].view(1, 1, V, 1).to(points_3d.device)
    cx = K[:, 0, 2].view(1, 1, V, 1).to(points_3d.device)
    cy = K[:, 1, 2].view(1, 1, V, 1).to(points_3d.device)
    u = fx * x + cx
    v = fy * y + cy
    return torch.stack([u, v], dim=-1)


def generate_batch(K, R, t, B=8, T=9, J=17, n_occluded=1, device="cpu"):
    """Generate one synthetic batch with random corrupted occlusions."""
    V = len(K)
    points_3d = torch.randn(B, T, J, 3, device=device) * 0.5
    points_2d = project(points_3d, K, R, t)
    confidence = torch.ones(B, T, V, J, device=device)
    visibility_target = torch.ones(B, T, V, J, device=device)

    for b in range(B):
        for t_i in range(T):
            for j_i in range(J):
                occluded = np.random.choice(V, size=n_occluded, replace=False)
                confidence[b, t_i, occluded, j_i] = 0.5
                points_2d[b, t_i, occluded, j_i, :] = (
                    torch.randn_like(points_2d[b, t_i, occluded, j_i, :]) * 5.0
                )
                visibility_target[b, t_i, occluded, j_i] = 0.0

    x = torch.cat([points_2d, confidence[..., None]], dim=-1)
    return x, points_3d, visibility_target


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    V, J, T, B = 4, 17, 9, 8
    K, R, t = make_circular_rig(V)
    device = "cpu"

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility(
        j=J,
        d=32,
        n_views=V,
        n_heads=2,
        n_joint_layers=1,
        n_st_layers=1,
        max_temporal_len=32,
        residual_hidden=64,
        principal_point_hidden=32,
        principal_point_max_offset=5.0,
        focal_max_scale=0.0,
        return_pp_delta=False,
        visibility_hidden=32,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def eval_error(n_occluded=2, n_batches=10):
        model.eval()
        total_err = 0.0
        total_vis_acc = 0.0
        n = 0
        with torch.no_grad():
            for _ in range(n_batches):
                x, target, vis_target = generate_batch(
                    K, R, t, B=B, T=T, J=J, n_occluded=n_occluded, device=device
                )
                pred, _weights, visibility = model(x, K=K, R=R, t=t)
                err = (pred - target).norm(dim=-1).mean().item()
                vis_acc = ((visibility > 0.5).float() == vis_target).float().mean().item()
                total_err += err
                total_vis_acc += vis_acc
                n += 1
        return total_err / n, total_vis_acc / n

    init_err, init_vis_acc = eval_error()
    print(f"Before training: err={init_err * 1000:.2f}mm, vis_acc={init_vis_acc:.3f}")

    model.train()
    for step in range(1, 121):
        x, target, vis_target = generate_batch(
            K, R, t, B=B, T=T, J=J, n_occluded=1, device=device
        )
        pred, _weights, visibility = model(x, K=K, R=R, t=t)
        loss_3d = F.mse_loss(pred, target)
        loss_vis = F.binary_cross_entropy(visibility, vis_target)
        loss = loss_3d + 0.1 * loss_vis

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            print(
                f"step {step:3d}: 3d_loss={loss_3d.item():.4f} "
                f"vis_loss={loss_vis.item():.4f}"
            )

    final_err, final_vis_acc = eval_error()
    print(f"After training: err={final_err * 1000:.2f}mm, vis_acc={final_vis_acc:.3f}")

    assert final_vis_acc > 0.75, f"Visibility mask accuracy too low: {final_vis_acc}"
    assert final_err < init_err, "Training did not reduce 3D error"
    print("smoke test passed")


if __name__ == "__main__":
    main()
