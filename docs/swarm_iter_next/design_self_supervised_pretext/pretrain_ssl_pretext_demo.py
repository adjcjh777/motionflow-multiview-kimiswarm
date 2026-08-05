"""Minimal prototype of the masked-view SSL pretext for temporal ray-attention fusion.

This script loads a canonical `.npz` clip, masks a subset of views, and runs one
optimization step of the SSL objective (visible + masked reprojection + temporal
smoothness + bone-length consistency).  It is meant as a proof-of-concept only;
use `experiments/pretrain_ray_attention_ssl.py` for full training.

Usage
-----
    conda run -n mf python docs/swarm_iter_next/design_self_supervised_pretext/pretrain_ssl_pretext_demo.py \
        --npz data/webbridge/h36m/s_01_acts_02_multiview.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# Make project imports available when run from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from motionflow_mv.fusion.ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual
from motionflow_mv.losses.reprojection import reprojection_loss
from experiments.train_utils import temporal_bone_length_consistency_loss, H36M_17_PARENTS


def load_clip(npz_path: str, clip_len: int = 13, start: int = 0):
    """Load a single temporal clip from a canonical .npz file."""
    data = np.load(npz_path)
    points_2d = torch.from_numpy(data["points_2d"][start : start + clip_len]).float()
    confidences = torch.from_numpy(data["confidences"][start : start + clip_len]).float()
    K = torch.from_numpy(data["camera_K"]).float()
    R = torch.from_numpy(data["camera_R"]).float()
    t = torch.from_numpy(data["camera_t"]).float()

    # Normalize H36M from millimetres to meters.
    if np.abs(t.numpy()).max() > 10.0:
        t = t / 1000.0

    x = torch.cat([points_2d, confidences.unsqueeze(-1)], dim=-1)  # (T, V, J, 3)
    x = x.unsqueeze(0)  # (B=1, T, V, J, 3)
    K = K.unsqueeze(0)  # (B=1, V, 3, 3)
    R = R.unsqueeze(0)
    t = t.unsqueeze(0)
    return x, K, R, t


def mask_views(x: torch.Tensor, mask_ratio: float = 0.25):
    """Randomly mask whole views and return a boolean mask.

    Args:
        x: (B, T, V, J, 3) input tensor with (x, y, confidence).
        mask_ratio: Probability of masking a view at a given frame.

    Returns:
        x_masked: same shape as x with masked confidence set to 0.
        mask: (B, T, V) boolean tensor (True = masked).
    """
    B, T, V, J, _ = x.shape
    mask = torch.rand(B, T, V, 1, device=x.device) < mask_ratio  # (B, T, V, 1)
    x_masked = x.clone()
    x_masked[..., 2] = x[..., 2] * (~mask).float()  # broadcast over J
    return x_masked, mask.squeeze(-1)  # (B, T, V)


def ssl_loss(pred_3d, x, K, R, t, mask, lambda_vis=1.0, lambda_mask=1.0,
             lambda_smooth=0.1, lambda_bone=0.1):
    """Compute masked-view SSL loss."""
    points_2d = x[..., :2]          # (B, T, V, J, 2)
    conf = x[..., 2]                  # (B, T, V, J)

    visible_mask = ~mask              # (B, T, V)

    # Visible-view reprojection.
    loss_vis = reprojection_loss(
        pred_3d, points_2d, K, R, t, confidences=conf,
        mask=visible_mask.unsqueeze(-1).expand_as(conf)
    )

    # Masked-view reprojection (the SSL signal).
    loss_mask = reprojection_loss(
        pred_3d, points_2d, K, R, t, confidences=conf,
        mask=mask.unsqueeze(-1).expand_as(conf)
    )

    # Temporal smoothness: second-order finite difference.
    if pred_3d.size(1) > 2:
        accel = pred_3d[:, 2:] - 2 * pred_3d[:, 1:-1] + pred_3d[:, :-2]
        loss_smooth = accel.pow(2).mean()
    else:
        loss_smooth = torch.tensor(0.0, device=pred_3d.device)

    # Bone-length consistency.
    B, T, J, _ = pred_3d.shape
    flat = pred_3d.view(B * T, J, 3)
    loss_bone = temporal_bone_length_consistency_loss(
        flat, parents=H36M_17_PARENTS, weight=1.0
    )

    return (
        lambda_vis * loss_vis
        + lambda_mask * loss_mask
        + lambda_smooth * loss_smooth
        + lambda_bone * loss_bone
    ), {
        "loss_vis": loss_vis.item(),
        "loss_mask": loss_mask.item(),
        "loss_smooth": loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else loss_smooth,
        "loss_bone": loss_bone.item(),
    }


def main():
    parser = argparse.ArgumentParser(description="SSL pretext demo")
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to canonical .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, K, R, t = load_clip(args.npz, clip_len=args.clip_len)
    x, K, R, t = x.to(device), K.to(device), R.to(device), t.to(device)
    x_orig = x.clone()

    B, T, V, J, _ = x.shape
    model = RayAttentionFusionModelTemporalResidual(
        j=J, d=args.d, n_views=V, n_temporal_layers=2, residual_hidden=64
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Loaded clip: B={B}, T={T}, V={V}, J={J}")
    print("Running SSL pretext demo (no 3D labels used)...")

    for epoch in range(args.epochs):
        x_masked, mask = mask_views(x_orig, mask_ratio=0.25)
        pred_3d, _ = model(x_masked, K=K, R=R, t=t)
        loss, parts = ssl_loss(pred_3d, x_orig, K, R, t, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1}: total={loss.item():.4f}  "
            f"vis={parts['loss_vis']:.4f} mask={parts['loss_mask']:.4f} "
            f"smooth={parts['loss_smooth']:.4f} bone={parts['loss_bone']:.4f}"
        )

    print("Demo complete.")


if __name__ == "__main__":
    main()
