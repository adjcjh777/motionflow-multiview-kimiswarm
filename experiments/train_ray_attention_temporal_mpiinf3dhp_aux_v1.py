"""Train RayAttentionFusionModelTemporal on MPI-INF-3DHP with auxiliary losses.

Auxiliaries added in this copy:
- velocity consistency loss: L1/L2 alignment of per-joint finite-difference
  velocities between prediction and ground truth.
- bone-length loss: supervised L1 on per-bone lengths w.r.t. the MPI-INF-3DHP
  28-joint skeleton.

Usage
-----
    conda run -n mf python experiments/train_ray_attention_temporal_mpiinf3dhp_aux_v1.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
               data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --clip_len 13 --epochs 5 --velocity_weight 0.1 --bone_weight 0.1

The script is a minimal copy of experiments/train_ray_attention_temporal_mpiinf3dhp.py
with the auxiliary-loss plumbing kept local to this file.
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

from motionflow_mv.fusion.ray_attention_temporal_model import RayAttentionFusionModelTemporal
from experiments.train_utils import bone_length_loss


# ---------------------------------------------------------------------------
# MPI-INF-3DHP 28-joint skeleton (0-based, -1 denotes root)
# Source: widely used MPI-INF-3DHP joint order from reference implementations
#         (Arka-h/mpi-inf-3dhp, XinArkh/VNect).
# ---------------------------------------------------------------------------
MPI_INF_3DHP_28_PARENTS = [
    2,  # 0  spine3        -> 2  spine2
    0,  # 1  spine4        -> 0  spine3
    3,  # 2  spine2        -> 3  spine
    4,  # 3  spine         -> 4  pelvis
    -1, # 4  pelvis        (root)
    1,  # 5  neck          -> 1  spine4
    5,  # 6  head          -> 5  neck
    6,  # 7  head_top      -> 6  head
    5,  # 8  left_clavicle -> 5  neck
    8,  # 9  left_shoulder -> 8  left_clavicle
    9,  # 10 left_elbow    -> 9  left_shoulder
    10, # 11 left_wrist    -> 10 left_elbow
    11, # 12 left_hand     -> 11 left_wrist
    5,  # 13 right_clavicle-> 5  neck
    13, # 14 right_shoulder-> 13 right_clavicle
    14, # 15 right_elbow   -> 14 right_shoulder
    15, # 16 right_wrist   -> 15 right_elbow
    16, # 17 right_hand    -> 16 right_wrist
    4,  # 18 left_hip      -> 4  pelvis
    18, # 19 left_knee     -> 18 left_hip
    19, # 20 left_ankle    -> 19 left_knee
    20, # 21 left_foot     -> 20 left_ankle
    21, # 22 left_toe      -> 21 left_foot
    4,  # 23 right_hip     -> 4  pelvis
    23, # 24 right_knee    -> 23 right_hip
    24, # 25 right_ankle   -> 24 right_knee
    25, # 26 right_foot    -> 25 right_ankle
    26, # 27 right_toe     -> 26 right_foot
]


def velocity_consistency_loss(pred: torch.Tensor, target: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Supervised velocity consistency loss over a clip of poses.

    Args:
        pred:   (B, T, J, 3) predicted 3D pose sequence.
        target: (B, T, J, 3) ground-truth 3D pose sequence.
        weight: scalar multiplier.

    Returns:
        Scalar loss.
    """
    if weight == 0.0 or pred.size(1) < 2:
        return torch.tensor(0.0, device=pred.device)
    pred_vel = pred[:, 1:] - pred[:, :-1]
    target_vel = target[:, 1:] - target[:, :-1]
    return weight * F.l1_loss(pred_vel, target_vel)


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
        # Number of valid starting positions.
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
    parser = argparse.ArgumentParser(description="Train temporal ray-attention fusion on MPI-INF-3DHP with auxiliary losses")
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000, help="Random clips per train sequence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/ray_attention_temporal_mpiinf3dhp_aux.pth")
    parser.add_argument("--velocity_weight", type=float, default=0.1, help="Weight for velocity consistency loss")
    parser.add_argument("--bone_weight", type=float, default=0.1, help="Weight for supervised bone-length loss")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build datasets.
    train_datasets = []
    for tp in args.train:
        train_datasets.append(RandomClipDataset(tp, args.clip_len, n_samples=args.train_samples))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(args.val, args.clip_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # Infer dimensions from first file.
    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}")

    model = RayAttentionFusionModelTemporal(
        j=j, d=args.d, n_views=n_views, n_temporal_layers=args.n_temporal_layers,
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
        train_aux = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb)
            optimizer.zero_grad()
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)

            # Auxiliary losses.
            if args.velocity_weight > 0.0:
                loss = loss + velocity_consistency_loss(pred, yb, weight=args.velocity_weight)
            if args.bone_weight > 0.0:
                b, t_len, j_n, _ = pred.shape
                pred_flat = pred.reshape(b * t_len, j_n, 3)
                target_flat = yb.reshape(b * t_len, j_n, 3)
                loss = loss + bone_length_loss(
                    pred_flat, target_flat, parents=MPI_INF_3DHP_28_PARENTS, weight=args.bone_weight
                )

            loss.backward()
            optimizer.step()
            train_loss += criterion(pred, yb).item() * xb.size(0)
            train_aux += (loss.item() - criterion(pred, yb).item()) * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_aux /= len(train_loader.dataset)

        val_err = evaluate(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, aux_loss={train_aux:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, aux_loss={train_aux:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
