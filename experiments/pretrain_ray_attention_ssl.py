"""Self-supervised masked-view pre-training for the temporal ray-attention fusion model.

The model is trained without any 3D ground truth.  Random views/time steps are masked
out in the input; the loss is the reprojection error on both visible and masked slots,
plus temporal smoothness and bone-length consistency regularizers.

Optionally adds a cross-view contrastive objective (``--lambda_contrast > 0``) that
encourages per-joint representations to be view-invariant, which can improve the
transferability of the pretrained weights to the downstream supervised task.

Usage
-----
    python experiments/pretrain_ray_attention_ssl.py \
        --train data/webbridge/h36m/s_01_acts_02_06_multiview.npz \
                data/webbridge/h36m/s_05_acts_02_06_multiview.npz \
        --val data/webbridge/h36m/s_09_acts_02_06_multiview.npz \
        --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 --epochs 50 \
        --mask_ratio 0.25 --lambda_contrast 0.1 \
        --output outputs/ray_attention_ssl_h36m.pth
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.ssl_dataset import make_ssl_dataloaders
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_ssl_view_contrast_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSSLViewContrast,
)
from motionflow_mv.losses import reprojection_loss
from train_utils import temporal_bone_length_consistency_loss


# Simple skeleton presets for bone-length consistency loss.
H36M_17_PARENTS = [
    -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14, 9,
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask_views(x: torch.Tensor, ratio: float = 0.25, mode: str = "mixed"):
    """Mask random views/time steps by zeroing confidences.

    Args:
        x: (B, T, V, J, 3) with the last channel as confidence.
        ratio: fraction of slots to mask.
        mode: "view", "time", or "mixed".

    Returns:
        x_masked, mask (B, T, V, J) bool where True means the slot is masked.
    """
    B, T, V, J, _ = x.shape
    masked = torch.zeros(B, T, V, J, dtype=torch.bool, device=x.device)

    if mode in ("view", "mixed"):
        n_view_mask = max(1, int(V * ratio / 2)) if mode == "mixed" else max(1, int(V * ratio))
        for b in range(B):
            for t in range(T):
                # Ensure at least one view remains unmasked.
                k = min(n_view_mask, V - 1)
                if k > 0:
                    idx = torch.randperm(V, device=x.device)[:k]
                    masked[b, t, idx, :] = True

    if mode in ("time", "mixed"):
        n_time_mask = max(1, int(T * ratio / 2)) if mode == "mixed" else max(1, int(T * ratio))
        for b in range(B):
            k = min(n_time_mask, T - 1)
            if k > 0:
                idx = torch.randperm(T, device=x.device)[:k]
                masked[b, idx, :, :] = True

    x_masked = x.clone()
    x_masked[..., 2] = x_masked[..., 2] * (~masked).float()
    return x_masked, masked


def augment_clip(x, noise_std: float = 0.5, dropout_rate: float = 0.05,
                 outlier_rate: float = 0.01, outlier_scale: float = 100.0):
    """Lightweight per-clip augmentation."""
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask
    if outlier_rate > 0:
        outlier_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) < outlier_rate
        outlier = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 2, device=x.device) - 0.5) * 2 * outlier_scale
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    return x


def temporal_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Second-order temporal smoothness (acceleration) over a clip."""
    if pred.shape[1] < 3:
        return torch.tensor(0.0, device=pred.device)
    acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    return acc.norm(dim=-1).mean()


@torch.no_grad()
def evaluate(model, loader, device, lambda_mask, lambda_smooth, lambda_bone, parents):
    model.eval()
    total_loss = 0.0
    total_vis = 0.0
    total_mask = 0.0
    total_count = 0
    for xb, K, R, t in loader:
        xb = xb.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        xb_masked, masked = mask_views(xb.clone(), ratio=0.0, mode="view")
        outputs = model(xb_masked, K=K, R=R, t=t)
        pred = outputs[0]
        points_2d = xb[..., :2]
        conf = xb[..., 2]
        loss_vis = reprojection_loss(pred, points_2d, K, R, t, confidences=conf, mask=~masked)
        loss = loss_vis
        total_vis += loss_vis.item() * xb.size(0)
        if lambda_mask > 0:
            loss_m = reprojection_loss(pred, points_2d, K, R, t, confidences=conf, mask=masked)
            total_mask += loss_m.item() * xb.size(0)
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
    return total_loss / total_count, total_vis / total_count, total_mask / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser(description="Self-supervised masked-view pre-training")
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Training .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=0.0,
                        help="Set to 0 to disable PP correction during SSL")
    parser.add_argument("--focal_max_scale", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--mask_mode", type=str, default="mixed", choices=["view", "time", "mixed"])
    parser.add_argument("--lambda_vis", type=float, default=1.0)
    parser.add_argument("--lambda_mask", type=float, default=1.0)
    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument("--lambda_bone", type=float, default=0.1)
    parser.add_argument("--lambda_contrast", type=float, default=0.0,
                        help="Weight for the cross-view contrastive loss during SSL pretraining")
    parser.add_argument("--contrastive_dim", type=int, default=64,
                        help="Projection dimension for the cross-view contrastive head")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07,
                        help="Temperature for the cross-view contrastive loss")
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--dropout_rate", type=float, default=0.05)
    parser.add_argument("--outlier_rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/ray_attention_ssl.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = make_ssl_dataloaders(
        args.train, args.val, args.clip_len, args.batch_size, args.train_samples, args.val_stride
    )

    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}")

    if args.lambda_contrast > 0.0:
        print(f"Using SSL view-contrastive model (lambda_contrast={args.lambda_contrast})")
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSSLViewContrast(
            j=j,
            d=args.d,
            n_views=n_views,
            n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=False,
            contrastive_dim=args.contrastive_dim,
            contrastive_temperature=args.contrastive_temperature,
            contrastive_loss_weight=1.0,
        ).to(device)
    else:
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
            j=j,
            d=args.d,
            n_views=n_views,
            n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=False,
        ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    parents = None
    if j == 17:
        parents = H36M_17_PARENTS
    elif args.lambda_bone > 0:
        print("Warning: unknown skeleton layout; disabling bone-length loss")
        args.lambda_bone = 0.0

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, K, R, t in train_loader:
            xb = xb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb, noise_std=args.noise_std, dropout_rate=args.dropout_rate,
                              outlier_rate=args.outlier_rate)
            xb_masked, masked = mask_views(xb, ratio=args.mask_ratio, mode=args.mask_mode)

            if args.lambda_contrast > 0.0:
                pred, weights, c_loss = model(xb_masked, K=K, R=R, t=t)
            else:
                pred, _ = model(xb_masked, K=K, R=R, t=t)
                c_loss = None
            points_2d = xb[..., :2]
            conf = xb[..., 2]

            loss_vis = reprojection_loss(pred, points_2d, K, R, t, confidences=conf, mask=~masked)
            loss_mask = reprojection_loss(pred, points_2d, K, R, t, confidences=conf, mask=masked)
            loss = args.lambda_vis * loss_vis + args.lambda_mask * loss_mask

            if c_loss is not None:
                loss = loss + args.lambda_contrast * c_loss

            if args.lambda_smooth > 0:
                loss = loss + args.lambda_smooth * temporal_smoothness_loss(pred)

            if args.lambda_bone > 0 and parents is not None:
                # Collapse batch and time for bone-length loss.
                pred_flat = pred.reshape(-1, j, 3)
                loss = loss + args.lambda_bone * temporal_bone_length_consistency_loss(
                    pred_flat, parents=parents, weight=1.0
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        val_loss, val_vis, val_mask = evaluate(
            model, val_loader, device, args.lambda_mask, args.lambda_smooth, args.lambda_bone, parents
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                  f"val_vis={val_vis:.6f} val_mask={val_mask:.6f} (saved)")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                  f"val_vis={val_vis:.6f} val_mask={val_mask:.6f}")

    print(f"Best val loss: {best_val:.6f} -> {output_path}")


if __name__ == "__main__":
    main()
