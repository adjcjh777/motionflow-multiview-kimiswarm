"""Train ray-aware attention fusion v3 on Human3.6M with a robustness curriculum.

Summary
-------
This script extends the v3 H36M trainer with a *scheduled corruption curriculum*:
corruption strength (2D Gaussian noise, view dropout, and sparse 2D outliers)
starts low and increases over epochs.  The idea is to let the model first learn
clean geometric triangulation, then progressively adapt to heavier real-world
noise.  Validation is reported on both a clean and a fixed-corruption hold-out
set so robustness can be tracked alongside standard reconstruction error.

Default data is the 62 k frame H36M multi-view NPZ produced by
`prepare_h36m_multiview.py` for subject 01, actions 02-16.

Usage
    /d/anaconda3/envs/jz_py310/python.exe experiments/train_ray_attention_v4_h36m.py \
        --dataset data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz \
        --epochs 50 --lr 1e-3 --d 64 --batch_size 32
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_v4_model import RayAttentionFusionModelV4


class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict, idx: np.ndarray):
        self.x = torch.from_numpy(data["points_2d"][idx]).float()
        self.conf = torch.from_numpy(data["confidences"][idx]).float()
        self.y = torch.from_numpy(data["joints_3d"][idx]).float()
        # Cameras may be a single rig (V, ...) broadcast to all samples.
        if data["camera_K"].shape[0] == self.x.shape[0]:
            self.K = torch.from_numpy(data["camera_K"][idx]).float()
            self.R = torch.from_numpy(data["camera_R"][idx]).float()
            self.t = torch.from_numpy(data["camera_t"][idx]).float()
        else:
            self.K = torch.from_numpy(data["camera_K"]).float().unsqueeze(0).repeat(len(idx), 1, 1, 1)
            self.R = torch.from_numpy(data["camera_R"]).float().unsqueeze(0).repeat(len(idx), 1, 1, 1)
            self.t = torch.from_numpy(data["camera_t"]).float().unsqueeze(0).repeat(len(idx), 1, 1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.cat([self.x[idx], self.conf[idx, ..., None]], dim=-1)
        return x, self.y[idx], self.K[idx], self.R[idx], self.t[idx]


def collate_fn(batch):
    xb = torch.stack([b[0] for b in batch], dim=0)
    yb = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return xb, yb, K, R, t


def augment_batch(x, noise_std=0.5, dropout_rate=0.1, outlier_rate=0.02, outlier_scale=100.0):
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask
    if outlier_rate > 0:
        outlier_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], device=x.device) < outlier_rate
        outlier = (torch.rand(x.shape[0], x.shape[1], x.shape[2], 2, device=x.device) - 0.5) * 2 * outlier_scale
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    return x


class RobustnessCurriculum:
    """Linear corruption schedule with an optional clean warmup phase.

    For the first ``warmup_epochs`` the corruption is held at its minimum.
    After that it linearly ramps from ``min`` to ``max`` over the remaining
    epochs.  All values are returned in pixels/rates suitable for
    ``augment_batch``.
    """

    def __init__(
        self,
        total_epochs: int,
        warmup_epochs: int = 5,
        noise_std_min: float = 0.0,
        noise_std_max: float = 5.0,
        dropout_rate_min: float = 0.0,
        dropout_rate_max: float = 0.3,
        outlier_rate_min: float = 0.0,
        outlier_rate_max: float = 0.05,
        outlier_scale_min: float = 50.0,
        outlier_scale_max: float = 100.0,
    ):
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        self.total_epochs = total_epochs
        self.warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.dropout_rate_min = dropout_rate_min
        self.dropout_rate_max = dropout_rate_max
        self.outlier_rate_min = outlier_rate_min
        self.outlier_rate_max = outlier_rate_max
        self.outlier_scale_min = outlier_scale_min
        self.outlier_scale_max = outlier_scale_max

    def _lerp(self, t: float, min_val: float, max_val: float) -> float:
        return min_val + t * (max_val - min_val)

    def get_params(self, epoch: int) -> dict:
        # epoch is 1-based.
        progress_epochs = max(0, self.total_epochs - self.warmup_epochs)
        if progress_epochs > 0 and epoch > self.warmup_epochs:
            t = min(1.0, (epoch - self.warmup_epochs) / progress_epochs)
        else:
            t = 0.0
        return {
            "noise_std": self._lerp(t, self.noise_std_min, self.noise_std_max),
            "dropout_rate": self._lerp(t, self.dropout_rate_min, self.dropout_rate_max),
            "outlier_rate": self._lerp(t, self.outlier_rate_min, self.outlier_rate_max),
            "outlier_scale": self._lerp(t, self.outlier_scale_min, self.outlier_scale_max),
        }


def run_epoch(model, loader, criterion, device, augment_fn=None):
    """Run one epoch and return (avg_loss, avg_mpjpe)."""
    total_loss = 0.0
    total_mpjpe = 0.0
    total_samples = 0
    for xb, yb, K, R, t in loader:
        xb, yb = xb.to(device), yb.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        if augment_fn is not None:
            xb = augment_fn(xb)
        pred, _ = model(xb, K=K, R=R, t=t)
        loss = criterion(pred, yb)
        total_loss += loss.item() * xb.size(0)
        total_mpjpe += (pred - yb).norm(dim=-1).mean().item() * xb.size(0)
        total_samples += xb.size(0)
    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_mpjpe / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # Curriculum parameters.
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of initial epochs with minimum corruption.")
    parser.add_argument("--noise_std_min", type=float, default=0.0)
    parser.add_argument("--noise_std_max", type=float, default=5.0)
    parser.add_argument("--dropout_rate_min", type=float, default=0.0)
    parser.add_argument("--dropout_rate_max", type=float, default=0.3)
    parser.add_argument("--outlier_rate_min", type=float, default=0.0)
    parser.add_argument("--outlier_rate_max", type=float, default=0.05)
    parser.add_argument("--outlier_scale_min", type=float, default=50.0)
    parser.add_argument("--outlier_scale_max", type=float, default=100.0)
    # Fixed validation corruption (used to report robustness metric).
    parser.add_argument("--val_noise_std", type=float, default=5.0)
    parser.add_argument("--val_dropout_rate", type=float, default=0.2)
    parser.add_argument("--val_outlier_rate", type=float, default=0.05)
    parser.add_argument("--val_outlier_scale", type=float, default=100.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.dataset)
    n = data["joints_3d"].shape[0]
    n_val = int(n * args.val_ratio)
    perm = np.random.permutation(n)
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    train_dataset = CameraDataset(data, train_idx)
    val_dataset = CameraDataset(data, val_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, K_shape={data['camera_K'].shape}")
    model = RayAttentionFusionModelV4(j=j, d=args.d, n_views=n_views).to(device)
    print(f"fusion_mlp weight shape: {model.fusion_mlp[0].weight.shape}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    curriculum = RobustnessCurriculum(
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        dropout_rate_min=args.dropout_rate_min,
        dropout_rate_max=args.dropout_rate_max,
        outlier_rate_min=args.outlier_rate_min,
        outlier_rate_max=args.outlier_rate_max,
        outlier_scale_min=args.outlier_scale_min,
        outlier_scale_max=args.outlier_scale_max,
    )

    best_val_mpjpe = float("inf")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    dataset_stem = Path(args.dataset).stem

    for epoch in range(1, args.epochs + 1):
        params = curriculum.get_params(epoch)
        augment_fn = lambda x: augment_batch(
            x,
            noise_std=params["noise_std"],
            dropout_rate=params["dropout_rate"],
            outlier_rate=params["outlier_rate"],
            outlier_scale=params["outlier_scale"],
        )

        model.train()
        train_loss, train_mpjpe = run_epoch(model, train_loader, criterion, device, augment_fn=augment_fn)

        model.eval()
        with torch.no_grad():
            clean_val_loss, clean_val_mpjpe = run_epoch(model, val_loader, criterion, device, augment_fn=None)
            corrupt_val_loss, corrupt_val_mpjpe = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                augment_fn=lambda x: augment_batch(
                    x,
                    noise_std=args.val_noise_std,
                    dropout_rate=args.val_dropout_rate,
                    outlier_rate=args.val_outlier_rate,
                    outlier_scale=args.val_outlier_scale,
                ),
            )

        if clean_val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = clean_val_mpjpe
            torch.save(model.state_dict(), output_dir / f"ray_attention_v4_{dataset_stem}.pth")

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_MPJPE={train_mpjpe:.4f}m, "
                f"clean_val_loss={clean_val_loss:.4f}, clean_val_MPJPE={clean_val_mpjpe:.4f}m, "
                f"corrupt_val_loss={corrupt_val_loss:.4f}, corrupt_val_MPJPE={corrupt_val_mpjpe:.4f}m "
                f"(noise={params['noise_std']:.2f}, drop={params['dropout_rate']:.2f}, "
                f"outlier={params['outlier_rate']:.3f})"
            )

    print(f"Best clean_val_MPJPE={best_val_mpjpe:.4f}m, checkpoint: {output_dir / f'ray_attention_v4_{dataset_stem}.pth'}")


if __name__ == "__main__":
    main()
