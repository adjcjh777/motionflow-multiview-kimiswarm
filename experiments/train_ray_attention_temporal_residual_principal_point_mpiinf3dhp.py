"""Train RayAttentionFusionModelTemporalResidualPrincipalPoint on MPI-INF-3DHP.

This is a thin wrapper around ``train_ray_attention_temporal_residual_mpiinf3dhp.py``
that swaps the model for ``RayAttentionFusionModelTemporalResidualPrincipalPoint``,
which adds a bounded, learned principal-point correction layer before the
differentiable triangulation step.  The correction is trained jointly with the
rest of the model; to make the layer learn, supply non-zero camera-perturbation
magnitudes for the principal point (e.g. ``--cam_aug_pp 5``).
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

from motionflow_mv.calibration.perturb import perturb_cameras_with_delta
from motionflow_mv.fusion.ray_attention_temporal_residual_principal_point_model import RayAttentionFusionModelTemporalResidualPrincipalPoint
from motionflow_mv.fusion.graph_joint_relation import MPI_INF_3DHP_28_PARENTS
from motionflow_mv.losses import bone_length_loss, reprojection_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield clips (T, V, J, 3) from a long canonical .npz sequence."""

    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()  # (T, V, J, 2)
        self.confidences = torch.from_numpy(data["confidences"]).float()  # (T, V, J)
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()  # (T, J, 3)
        self.K = torch.from_numpy(data["camera_K"]).float()  # (V, 3, 3)
        self.R = torch.from_numpy(data["camera_R"]).float()  # (V, 3, 3)
        self.t = torch.from_numpy(data["camera_t"]).float()  # (V, 3)

        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - self.clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end],
             self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )  # (T, V, J, 3)
        y = self.joints_3d[start:end]  # (T, J, 3)
        return x, y, self.K, self.R, self.t


class RandomClipDataset(torch.utils.data.Dataset):
    """Sample random clips from a sequence; useful for train set augmentation."""

    def __init__(self, npz_path: str, clip_len: int, n_samples: int = 2000):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.n_samples = n_samples
        self.total_frames = self.points_2d.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = random.randint(0, max(0, self.total_frames - self.clip_len))
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end],
             self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def augment_clip(x, noise_std: float = 0.5, dropout_rate: float = 0.1,
                 outlier_rate: float = 0.02, outlier_scale: float = 100.0):
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


def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, *_ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser(description="Train temporal ray-attention fusion with residual refinement and principal-point correction on MPI-INF-3DHP")
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000, help="Random clips per train sequence")
    parser.add_argument("--val_stride", type=int, default=1, help="Stride for validation clips (higher = faster)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 = main thread)")
    parser.add_argument("--reproj_weight", type=float, default=0.0, help="Weight for reprojection auxiliary loss")
    parser.add_argument("--bone_weight", type=float, default=0.0, help="Weight for bone-length auxiliary loss")
    parser.add_argument("--pp_loss_weight", type=float, default=0.0, help="Weight for principal-point offset supervision loss")
    parser.add_argument("--focal_max_scale", type=float, default=0.0, help="Maximum predicted focal-length scale (relative); 0 disables focal correction")
    parser.add_argument("--use_reproj_gate", action="store_true", help="Use reprojection-error gate in the residual head")
    parser.add_argument("--cam_aug_rot", type=float, default=0.5, help="Camera rotation augmentation std in degrees")
    parser.add_argument("--cam_aug_trans", type=float, default=0.005, help="Camera translation augmentation std in meters")
    parser.add_argument("--cam_aug_focal", type=float, default=0.01, help="Camera focal length augmentation std (relative)")
    parser.add_argument("--cam_aug_pp", type=float, default=2.0, help="Camera principal point augmentation std in pixels")
    parser.add_argument("--start_epoch", type=int, default=1, help="Epoch to resume from (1-based)")
    parser.add_argument("--end_epoch", type=int, default=None, help="Epoch to stop at (1-based, inclusive)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/ray_attention_temporal_residual_principal_point_mpiinf3dhp.pth")
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
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=args.num_workers > 0,
    )

    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}, residual_hidden={args.residual_hidden}, "
          f"principal_point_hidden={args.principal_point_hidden}, principal_point_max_offset={args.principal_point_max_offset}, "
          f"focal_max_scale={args.focal_max_scale}")

    model = RayAttentionFusionModelTemporalResidualPrincipalPoint(
        j=j, d=args.d, n_views=n_views, n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
        use_reproj_gate=args.use_reproj_gate,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=args.principal_point_max_offset,
        focal_max_scale=args.focal_max_scale,
        return_pp_delta=args.pp_loss_weight > 0.0 or args.focal_max_scale > 0.0,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
        print(f"Resumed from {args.resume}")

    start_epoch = args.start_epoch
    end_epoch = args.end_epoch if args.end_epoch is not None else args.epochs
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb)
            K, R, t, true_pp_delta, true_focal_scale = perturb_cameras_with_delta(
                K, R, t,
                rot_std=args.cam_aug_rot,
                trans_std=args.cam_aug_trans,
                focal_std=args.cam_aug_focal,
                pp_std=args.cam_aug_pp,
            )
            optimizer.zero_grad()
            outputs = model(xb, K=K, R=R, t=t)
            pred = outputs[0]
            loss = criterion(pred, yb)
            if args.pp_loss_weight > 0.0:
                pred_pp_delta = outputs[2]  # (B*T, V, 2)
                B, T = yb.shape[:2]
                true_pp_delta = true_pp_delta.to(device).unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, -1, 2)
                # The correction layer *adds* predicted delta to the perturbed principal point,
                # so the target is the negative of the applied offset.
                loss = loss + args.pp_loss_weight * criterion(pred_pp_delta, -true_pp_delta)
                if args.focal_max_scale > 0.0:
                    pred_focal_scale = outputs[3]  # (B*T, V)
                    # True focal scale has shape (B, V, 1); broadcast across the
                    # temporal dimension and reshape to (B*T, V). The correction layer
                    # should predict the inverse of the perturbation so that the
                    # corrected focal length matches the original calibration.
                    true_focal_scale = true_focal_scale.to(device).squeeze(-1).unsqueeze(1).expand(B, T, -1)
                    target_focal_scale = 1.0 / true_focal_scale.reshape(B * T, -1)
                    loss = loss + args.pp_loss_weight * criterion(pred_focal_scale, target_focal_scale)
            if args.reproj_weight > 0.0:
                points_2d = xb[..., :2]
                conf = xb[..., 2]
                loss_reproj = reprojection_loss(pred, points_2d, K, R, t, confidences=conf)
                loss = loss + args.reproj_weight * loss_reproj
            if args.bone_weight > 0.0:
                loss_bone = bone_length_loss(pred, yb, MPI_INF_3DHP_28_PARENTS)
                loss = loss + args.bone_weight * loss_bone
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
