"""Train MultiTaskShapePoseModel on MPI-INF-3DHP clips.

This script extends ``train_ray_attention_temporal_residual_mpiinf3dhp.py`` by
replacing the model with ``MultiTaskShapePoseModel``.  The 3D pose regression
task is unchanged; an additional SMPL shape/pose head is trained in parallel
when ``smplx`` is available and ``--smpl_model_path`` points to a valid model.

Usage
-----
    conda run -n mf python experiments/train_multitask_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --clip_len 13 --epochs 30 --smpl_model_path data/smpl/SMPL_NEUTRAL.pkl

If ``smplx`` is not installed the SMPL auxiliary losses are silently skipped
and the script falls back to training the 3D pose task only.
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

from motionflow_mv.calibration.perturb import perturb_cameras
from motionflow_mv.fusion.multi_task_shape_pose import (
    MultiTaskShapePoseModel,
    HAS_SMPLX as _HAS_SMPLX,
)
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
        self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)

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
            pred, _ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser(description="Train multi-task shape/pose model on MPI-INF-3DHP")
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000, help="Random clips per train sequence")
    parser.add_argument("--val_stride", type=int, default=1, help="Stride for validation clips (higher = faster)")
    parser.add_argument("--reproj_weight", type=float, default=0.0, help="Weight for reprojection auxiliary loss")
    parser.add_argument("--bone_weight", type=float, default=0.0, help="Weight for bone-length auxiliary loss")
    parser.add_argument("--use_reproj_gate", action="store_true", help="Use reprojection-error gate in the residual head")
    parser.add_argument("--cam_aug_rot", type=float, default=0.5, help="Camera rotation augmentation std in degrees")
    parser.add_argument("--cam_aug_trans", type=float, default=0.005, help="Camera translation augmentation std in meters")
    parser.add_argument("--cam_aug_focal", type=float, default=0.01, help="Camera focal length augmentation std (relative)")
    parser.add_argument("--cam_aug_pp", type=float, default=2.0, help="Camera principal point augmentation std in pixels")
    parser.add_argument("--start_epoch", type=int, default=1, help="Epoch to resume from (1-based)")
    parser.add_argument("--end_epoch", type=int, default=None, help="Epoch to stop at (1-based, inclusive)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/multitask_shape_pose_mpiinf3dhp.pth")

    # Multi-task SMPL head options.
    parser.add_argument("--smpl_model_path", type=str, default="data/smpl/SMPL_NEUTRAL.pkl",
                        help="Path to SMPL_NEUTRAL.pkl (required for SMPL auxiliary losses)")
    parser.add_argument("--shape_loss_weight", type=float, default=0.1,
                        help="Weight for SMPL 3D joint loss")
    parser.add_argument("--pose_loss_weight", type=float, default=0.0,
                        help="Weight for SMPL body-pose parameter L2 regularizer")
    parser.add_argument("--shape_prior_weight", type=float, default=1e-4,
                        help="Weight for shape prior ||betas||^2")
    parser.add_argument("--freeze_smpl_head_epochs", type=int, default=0,
                        help="Freeze SMPL head for the first N epochs")
    args = parser.parse_args()

    if not _HAS_SMPLX:
        print("[warning] smplx not installed; training 3D pose task only.")

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
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}, residual_hidden={args.residual_hidden}")

    model = MultiTaskShapePoseModel(
        j=j, d=args.d, n_views=n_views, n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
        use_reproj_gate=args.use_reproj_gate,
        smpl_model_path=args.smpl_model_path,
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
        # Optionally freeze the SMPL head during warm-up.
        smpl_frozen = False
        if args.freeze_smpl_head_epochs > 0 and epoch <= args.freeze_smpl_head_epochs:
            for p in model.shape_pose_head.parameters():
                p.requires_grad = False
            smpl_frozen = True
        else:
            for p in model.shape_pose_head.parameters():
                p.requires_grad = True

        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb)
            K, R, t = perturb_cameras(
                K, R, t,
                rot_std=args.cam_aug_rot,
                trans_std=args.cam_aug_trans,
                focal_std=args.cam_aug_focal,
                pp_std=args.cam_aug_pp,
            )

            optimizer.zero_grad()
            pred_3d, weights, smpl_out = model(xb, K=K, R=R, t=t, return_smpl=True)

            # Primary 3D pose task.
            loss = criterion(pred_3d, yb)

            # Optional reprojection / bone-length losses on the main 3D output.
            if args.reproj_weight > 0.0:
                points_2d = xb[..., :2]
                conf = xb[..., 2]
                loss_reproj = reprojection_loss(pred_3d, points_2d, K, R, t, confidences=conf)
                loss = loss + args.reproj_weight * loss_reproj
            if args.bone_weight > 0.0:
                loss_bone = bone_length_loss(pred_3d, yb, MPI_INF_3DHP_28_PARENTS)
                loss = loss + args.bone_weight * loss_bone

            # Multi-task SMPL head losses.
            if smpl_out is not None and not smpl_frozen:
                if "pred_joints_17" in smpl_out:
                    # Reshape (B*T, J, 3) -> (B, T, J, 3) to match yb.
                    smpl_joints = smpl_out["pred_joints_17"].view(xb.shape[0], xb.shape[1], j, 3)
                    loss = loss + args.shape_loss_weight * criterion(smpl_joints, yb)
                if args.pose_loss_weight > 0.0 and "body_pose" in smpl_out:
                    loss = loss + args.pose_loss_weight * smpl_out["body_pose"].pow(2).mean()
                if args.shape_prior_weight > 0.0 and "betas" in smpl_out:
                    loss = loss + args.shape_prior_weight * smpl_out["betas"].pow(2).mean()

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
