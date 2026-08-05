"""Train visibility-gated cross-view PP model on MPI-INF-3DHP.

Largely reuses the principal-point correction training script and adds a BCE
auxiliary loss on the learned per-view/per-joint visibility mask.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.perturb import perturb_cameras_with_delta
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_visibility_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility,
)
from motionflow_mv.losses import reprojection_loss
from train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp import (
    RandomClipDataset,
    TemporalClipDataset,
    augment_clip,
    collate_fn,
    evaluate,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train visibility-gated cross-view temporal residual + principal-point correction on MPI-INF-3DHP"
    )
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--reproj_weight", type=float, default=0.0)
    parser.add_argument("--pp_loss_weight", type=float, default=0.0)
    parser.add_argument("--focal_loss_weight", type=float, default=None)
    parser.add_argument("--focal_max_scale", type=float, default=0.0)
    parser.add_argument("--cam_aug_rot", type=float, default=0.5)
    parser.add_argument("--cam_aug_trans", type=float, default=0.005)
    parser.add_argument("--cam_aug_focal", type=float, default=0.01)
    parser.add_argument("--cam_aug_pp", type=float, default=2.0)
    parser.add_argument("--cam_aug_schedule", type=str, default="flat", choices=["flat", "extrinsic_curriculum"])
    parser.add_argument("--cam_aug_ramp_epochs", type=int, default=10)
    parser.add_argument("--view_dropout_rate", type=float, default=0.0)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--visibility_loss_weight", type=float, default=0.1, help="Weight for BCE visibility loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/ray_attention_temporal_crossview_residual_principal_point_visibility_mpiinf3dhp.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_datasets = []
    for tp in args.train:
        train_datasets.append(RandomClipDataset(tp, args.clip_len, n_samples=args.train_samples))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(args.val, args.clip_len, stride=args.val_stride)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}, n_st_layers={args.n_st_layers}, "
          f"residual_hidden={args.residual_hidden}")

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility(
        j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=args.principal_point_max_offset,
        focal_max_scale=args.focal_max_scale,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        if args.cam_aug_schedule == "extrinsic_curriculum":
            ramp = min(1.0, epoch / max(1, args.cam_aug_ramp_epochs))
            schedule_rot = args.cam_aug_rot * ramp
            schedule_trans = args.cam_aug_trans * ramp
        else:
            schedule_rot = args.cam_aug_rot
            schedule_trans = args.cam_aug_trans
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb, view_dropout_rate=args.view_dropout_rate, min_views=args.min_views)
            K, R, t, true_pp_delta, true_focal_scale = perturb_cameras_with_delta(
                K, R, t,
                rot_std=schedule_rot,
                trans_std=schedule_trans,
                focal_std=args.cam_aug_focal,
                pp_std=args.cam_aug_pp,
            )
            optimizer.zero_grad()
            pred, weights, visibility = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)

            # Visibility BCE loss: target is 1 for views with positive detector confidence.
            visible_target = (xb[..., 2] > 0).float()
            vis_loss = F.binary_cross_entropy(visibility, visible_target)
            loss = loss + args.visibility_loss_weight * vis_loss

            if args.pp_loss_weight > 0.0:
                # visibility model does not return pp_delta; this branch is not used.
                pass
            if args.reproj_weight > 0.0:
                points_2d = xb[..., :2]
                conf = xb[..., 2]
                loss_reproj = reprojection_loss(pred, points_2d, K, R, t, confidences=conf)
                loss = loss + args.reproj_weight * loss_reproj
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_err = evaluate(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
