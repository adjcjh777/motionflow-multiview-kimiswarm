"""Synthetic-to-real transfer training for ray-aware attention fusion v3.

Pre-trains RayAttentionFusionModelV3 on the domain-matched synthetic dataset,
then fine-tunes on real Human3.6M data.

Example (quick smoke test):
    /d/anaconda3/envs/jz_py310/python.exe experiments/train_ray_attention_v3_transfer.py \
        --synthetic_dataset outputs/synthetic_multiview_dataset.npz \
        --real_dataset data/h36m_hf/s_01_acts_02_03_..._16_multiview.npz \
        --synth_epochs 2 --real_epochs 2 --batch_size 32

Full run:
    --synth_epochs 50 --real_epochs 50 --batch_size 32 --lr 1e-3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_v3_model import RayAttentionFusionModelV3


class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict, idx: np.ndarray):
        self.x = torch.from_numpy(data["points_2d"][idx]).float()
        self.conf = torch.from_numpy(data["confidences"][idx]).float()
        self.y = torch.from_numpy(data["joints_3d"][idx]).float()
        # Cameras may be a single rig (V, ...) broadcast to all samples, or a
        # per-frame rig (T, V, ...).  In the latter case the leading dimension
        # matches the full dataset and we index it with ``idx``.
        n_frames = data["points_2d"].shape[0]
        if data["camera_K"].shape[0] == n_frames:
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


def augment_batch(x, noise_std=1.0, dropout_rate=0.1, outlier_rate=0.02, outlier_scale=100.0):
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


def run_epoch(model, loader, optimizer, criterion, device, augment=False, augment_kwargs=None):
    model.train() if optimizer is not None else model.eval()
    total_loss = 0.0
    total_mpjpe = 0.0
    for xb, yb, K, R, t in loader:
        xb, yb = xb.to(device), yb.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        if optimizer is not None:
            xb = augment_batch(xb, **augment_kwargs)
            optimizer.zero_grad()
        pred, _ = model(xb, K=K, R=R, t=t)
        loss = criterion(pred, yb)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_mpjpe += (pred - yb).norm(dim=-1).mean().item() * xb.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_mpjpe / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dataset", type=str, default="outputs/synthetic_multiview_dataset.npz")
    parser.add_argument("--real_dataset", type=str, default="data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--synth_epochs", type=int, default=30)
    parser.add_argument("--real_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--outlier_rate", type=float, default=0.02)
    parser.add_argument("--outlier_scale", type=float, default=100.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    synth_stem = Path(args.synthetic_dataset).stem
    real_stem = Path(args.real_dataset).stem

    augment_kwargs = {
        "noise_std": args.noise_std,
        "dropout_rate": args.dropout_rate,
        "outlier_rate": args.outlier_rate,
        "outlier_scale": args.outlier_scale,
    }

    # Pre-train on synthetic data.
    print(f"\n=== Loading synthetic dataset: {args.synthetic_dataset} ===")
    synth_data = np.load(args.synthetic_dataset)
    n_synth = synth_data["joints_3d"].shape[0]
    n_val_synth = int(n_synth * args.val_ratio)
    perm = np.random.permutation(n_synth)
    train_idx_synth = perm[n_val_synth:]
    val_idx_synth = perm[:n_val_synth]

    synth_train_loader = torch.utils.data.DataLoader(
        CameraDataset(synth_data, train_idx_synth),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    synth_val_loader = torch.utils.data.DataLoader(
        CameraDataset(synth_data, val_idx_synth),
        batch_size=args.batch_size, collate_fn=collate_fn,
    )

    n_views = synth_data["camera_K"].shape[-3]
    j = synth_data["points_2d"].shape[-2]
    print(f"Synthetic: n_views={n_views}, j={j}, frames={n_synth}")

    model = RayAttentionFusionModelV3(j=j, d=args.d, n_views=n_views).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    best_val = float("inf")
    for epoch in range(1, args.synth_epochs + 1):
        train_loss, train_mpjpe = run_epoch(
            model, synth_train_loader, optimizer, criterion, device, augment=True, augment_kwargs=augment_kwargs
        )
        val_loss, val_mpjpe = run_epoch(
            model, synth_val_loader, None, criterion, device, augment=False, augment_kwargs=None
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / f"ray_attention_v3_{synth_stem}.pth")
        if epoch % 5 == 0 or epoch == 1:
            print(f"[Synth] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MPJPE={val_mpjpe:.4f}mm")

    print(f"Best synthetic val_loss={best_val:.4f}, checkpoint: {output_dir / f'ray_attention_v3_{synth_stem}.pth'}")

    # Fine-tune on real H36M data.
    print(f"\n=== Loading real dataset: {args.real_dataset} ===")
    real_data = np.load(args.real_dataset)
    n_real = real_data["joints_3d"].shape[0]
    n_val_real = int(n_real * args.val_ratio)
    perm = np.random.permutation(n_real)
    train_idx_real = perm[n_val_real:]
    val_idx_real = perm[:n_val_real]

    real_train_loader = torch.utils.data.DataLoader(
        CameraDataset(real_data, train_idx_real),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    real_val_loader = torch.utils.data.DataLoader(
        CameraDataset(real_data, val_idx_real),
        batch_size=args.batch_size, collate_fn=collate_fn,
    )

    n_views_real = real_data["camera_K"].shape[-3]
    j_real = real_data["points_2d"].shape[-2]
    print(f"Real: n_views={n_views_real}, j={j_real}, frames={n_real}")
    assert n_views_real == n_views and j_real == j, "Synthetic and real datasets must share V and J"

    # Load best synthetic checkpoint before fine-tuning.
    synth_ckpt = output_dir / f"ray_attention_v3_{synth_stem}.pth"
    if synth_ckpt.exists():
        model.load_state_dict(torch.load(synth_ckpt, map_location=device))
        print(f"Loaded synthetic checkpoint: {synth_ckpt}")
    else:
        print("Synthetic checkpoint not found; fine-tuning from random init.")

    # Use a fresh optimizer for fine-tuning with a slightly lower LR.
    optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.5)
    best_val = float("inf")
    best_mpjpe = float("inf")
    final_ckpt = output_dir / f"ray_attention_v3_transfer_{synth_stem}_{real_stem}.pth"

    for epoch in range(1, args.real_epochs + 1):
        train_loss, train_mpjpe = run_epoch(
            model, real_train_loader, optimizer, criterion, device, augment=True, augment_kwargs=augment_kwargs
        )
        val_loss, val_mpjpe = run_epoch(
            model, real_val_loader, None, criterion, device, augment=False, augment_kwargs=None
        )
        if val_loss < best_val:
            best_val = val_loss
            best_mpjpe = val_mpjpe
            torch.save(model.state_dict(), final_ckpt)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[Real] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MPJPE={val_mpjpe:.4f}mm")

    print(f"Best real val_loss={best_val:.4f}, best val_MPJPE={best_mpjpe:.4f}mm, checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
