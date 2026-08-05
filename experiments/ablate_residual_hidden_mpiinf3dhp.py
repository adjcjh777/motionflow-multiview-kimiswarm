"""Ablation study: residual_hidden size for RayAttentionFusionModelTemporalResidual.

Trains residual models with residual_hidden in {32, 64, 128, 256} and reports
MPJPE vs. parameter count.  Designed as a short smoke-run (≤10 epochs) on
the MPI-INF-3DHP smoke .npz files by default so it finishes in <30 min on a
single RTX 4090.

Usage
-----
    conda run -n mf python experiments/ablate_residual_hidden_mpiinf3dhp.py \
        --epochs 5 --train_samples 1000 --batch_size 8 --smoke

The script writes per-experiment checkpoints to
``outputs/ablate_residual_hidden/`` and a markdown report to
``docs/swarm_iter6/ablate_residual_hidden_report.md``.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual


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
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()

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
        )
        y = self.joints_3d[start:end]
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


def train_single(residual_hidden: int, args, train_paths, val_path):
    set_seed(args.seed + residual_hidden)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_datasets = [RandomClipDataset(tp, args.clip_len, n_samples=args.train_samples) for tp in train_paths]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(val_path, args.clip_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    sample = np.load(train_paths[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]

    model = RayAttentionFusionModelTemporalResidual(
        j=j,
        d=args.d,
        n_views=n_views,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=residual_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[residual_hidden={residual_hidden}] n_params={n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"residual_h{residual_hidden}.pth"

    best_val = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb)
            optimizer.zero_grad()
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_err = evaluate(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            best_epoch = epoch
            torch.save(model.state_dict(), output_path)

        history.append((epoch, train_loss, val_err * 1000.0))
        print(f"[residual_hidden={residual_hidden}] Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    return {
        "residual_hidden": residual_hidden,
        "n_params": n_params,
        "best_val_mpjpe_mm": best_val * 1000.0,
        "best_epoch": best_epoch,
        "checkpoint": str(output_path),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation: residual_hidden size for temporal residual model")
    parser.add_argument("--data_root", type=str, default="data/webbridge/mpi_inf_3dhp")
    parser.add_argument("--smoke", action="store_true", default=True, help="Use *_smoke.npz files for fast run")
    parser.add_argument("--full", dest="smoke", action="store_false", help="Use full .npz files instead of smoke files")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/ablate_residual_hidden")
    parser.add_argument("--report", type=str, default="docs/swarm_iter6/ablate_residual_hidden_report.md")
    parser.add_argument("--residual_hidden", type=int, nargs="+", default=[32, 64, 128, 256])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    suffix = "_smoke" if args.smoke else ""
    train_paths = [
        data_root / f"s_01_seq_01_v14_multiview_m{suffix}.npz",
        data_root / f"s_01_seq_02_v14_multiview_m{suffix}.npz",
    ]
    val_path = data_root / f"s_02_seq_01_v14_multiview_m{suffix}.npz"

    missing = [p for p in train_paths + [val_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing data files: {missing}")

    results = []
    start_time = time.time()
    for h in args.residual_hidden:
        result = train_single(h, args, train_paths, val_path)
        results.append(result)
    elapsed = time.time() - start_time

    # Write report.
    report_path = Path(args.report)
    report_path.parent.mkdir(exist_ok=True, parents=True)
    with open(report_path, "w") as f:
        f.write("# Ablation: residual_hidden size\n\n")
        f.write("Trains ``RayAttentionFusionModelTemporalResidual`` with different ``residual_hidden``\n")
        f.write("sizes and reports cross-subject val MPJPE (mm) and parameter count.\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Data: `{args.data_root}` (smoke={args.smoke})\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Random clips per train seq: {args.train_samples}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- d: {args.d}, n_temporal_layers: {args.n_temporal_layers}\n")
        f.write(f"- Total elapsed time: {elapsed/60:.1f} min\n\n")
        f.write("## Results\n\n")
        f.write("| residual_hidden | Parameters | Best val MPJPE (mm) | Best epoch | Checkpoint |\n")
        f.write("|----------------:|-----------:|--------------------:|-----------:|------------|\n")
        for r in results:
            f.write(
                f"| {r['residual_hidden']} | {r['n_params']:,} | "
                f"{r['best_val_mpjpe_mm']:.2f} | {r['best_epoch']} | "
                f"`{r['checkpoint']}` |\n"
            )
        f.write("\n## Per-epoch history\n\n")
        for r in results:
            f.write(f"### residual_hidden={r['residual_hidden']}\n\n")
            f.write("| Epoch | Train loss | Val MPJPE (mm) |\n")
            f.write("|------:|-----------:|---------------:|\n")
            for epoch, train_loss, val_mpjpe in r["history"]:
                f.write(f"| {epoch} | {train_loss:.6f} | {val_mpjpe:.2f} |\n")
            f.write("\n")

        # Try to identify sweet spot.
        best = min(results, key=lambda x: x["best_val_mpjpe_mm"])
        f.write("## Sweet spot\n\n")
        f.write(
            f"Lowest val MPJPE = {best['best_val_mpjpe_mm']:.2f} mm "
            f"with residual_hidden={best['residual_hidden']} ({best['n_params']:,} params).\n\n"
        )
        f.write("## Notes\n\n")
        f.write("This is a smoke-run ablation; absolute MPJPE differs from the fully trained checkpoint, ")
        f.write("but the relative ranking of ``residual_hidden`` sizes is informative for architecture design.\n")

    print(f"\nReport written to {report_path}")
    print("\n=== Summary ===")
    for r in results:
        print(f"residual_hidden={r['residual_hidden']:>3}: params={r['n_params']:>8,}, best MPJPE={r['best_val_mpjpe_mm']:.2f}mm @ epoch {r['best_epoch']}")
    print(f"Sweet spot: residual_hidden={best['residual_hidden']} ({best['n_params']:,} params)")


if __name__ == "__main__":
    main()
