"""Data-efficiency curve: fine-tune a pretrained checkpoint on label fractions.

This script takes a pretrained checkpoint and a list of MPI-INF-3DHP training
.npz files, fine-tunes on a configurable set of label fractions
(default: 5, 10, 25, 50, 100 %), and evaluates each resulting checkpoint on
the validation set.  The output is a JSON curve of MPJPE/PA-MPJPE vs. label
fraction.

Usage
-----
    python experiments/run_data_efficiency_curve.py \
        --pretrained outputs/ray_attention_temporal_crossview_residual_principal_point_full.pth \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
                data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --fractions 0.05 0.10 0.25 0.50 1.00 \
        --output_dir outputs/data_efficiency_curve \
        --epochs 20 --batch_size 8 --train_samples 4000

CPU smoke test (1 % subset, tiny model, 2 epochs):
    python experiments/run_data_efficiency_curve.py --smoke
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import mpjpe as mpjpe_metric
from motionflow_mv.eval.metrics import pa_mpjpe as pa_mpjpe_metric
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RandomClipDataset(torch.utils.data.Dataset):
    """Sample random clips from a canonical .npz sequence."""

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
            [
                self.points_2d[start:end],
                self.confidences[start:end].unsqueeze(-1),
            ],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield deterministic sliding clips from a canonical .npz sequence."""

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
        self.num_clips = max(1, (self.total_frames - self.clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [
                self.points_2d[start:end],
                self.confidences[start:end].unsqueeze(-1),
            ],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


class SubsetRandomClipDataset(torch.utils.data.Dataset):
    """Wrap a RandomClipDataset and expose only a deterministic fraction."""

    def __init__(self, base: RandomClipDataset, fraction: float, seed: int = 42):
        self.base = base
        self.fraction = fraction
        n = len(base)
        rng = np.random.default_rng(seed)
        self.indices = rng.choice(n, size=max(1, int(n * fraction)), replace=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def augment_clip(x, noise_std: float = 0.5, dropout_rate: float = 0.1):
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (
            torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)
            > dropout_rate
        ).float()
        x[..., 2] = x[..., 2] * mask
    return x


def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, *_ = model(xb, K=K, R=R, t=t)
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    return {"mpjpe": mpjpe_metric(preds, gts), "pa_mpjpe": pa_mpjpe_metric(preds, gts)}


def build_model(
    j: int,
    n_views: int,
    d: int,
    n_st_layers: int,
    residual_hidden: int,
    principal_point_hidden: int,
    principal_point_max_offset: float,
    focal_max_scale: float,
):
    return RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=j,
        d=d,
        n_views=n_views,
        n_st_layers=n_st_layers,
        residual_hidden=residual_hidden,
        principal_point_hidden=principal_point_hidden,
        principal_point_max_offset=principal_point_max_offset,
        focal_max_scale=focal_max_scale,
        return_pp_delta=True,
    )


def train_once(
    pretrained_path: Path,
    train_files,
    val_file,
    fraction: float,
    args,
    out_dir: Path,
    seed: int,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\n=== Fraction {fraction*100:.0f}% | device={device} | seed={seed} ===")

    train_datasets = []
    for tp in train_files:
        base = RandomClipDataset(tp, args.clip_len, n_samples=args.train_samples)
        subset = SubsetRandomClipDataset(base, fraction, seed=seed)
        train_datasets.append(subset)
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(val_file, args.clip_len, stride=args.val_stride)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    sample = np.load(train_files[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]

    model = build_model(
        j=j,
        n_views=n_views,
        d=args.d,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=args.principal_point_max_offset,
        focal_max_scale=args.focal_max_scale,
    ).to(device)

    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing keys when warm-starting: {missing[:5]}")
        if unexpected:
            print(f"Warning: unexpected keys when warm-starting: {unexpected[:5]}")
        print(f"Warm-started from {pretrained_path}")
    else:
        print("Training from scratch (no pretrained checkpoint provided)")

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
    print(f"Train clips: {len(train_dataset)}  Val clips: {len(val_dataset)}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_path = out_dir / f"data_efficiency_fraction{int(fraction*100):03d}_seed{seed}.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb, noise_std=args.noise_std, dropout_rate=args.dropout_rate)
            optimizer.zero_grad()
            pred, *_ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_report = evaluate(model, val_loader, device)
        mpjpe_mm = float(val_report["mpjpe"])
        pa_mm = float(val_report["pa_mpjpe"])
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}  val_MPJPE={mpjpe_mm:.2f}mm  PA={pa_mm:.2f}mm")

        if mpjpe_mm < best_val:
            best_val = mpjpe_mm
            torch.save(model.state_dict(), best_path)

    report = evaluate(model, val_loader, device)
    report_path = out_dir / f"data_efficiency_fraction{int(fraction*100):03d}_seed{seed}.json"
    summary = {
        "fraction": fraction,
        "seed": seed,
        "train_clips": len(train_dataset),
        "val_clips": len(val_dataset),
        "best_val_mpjpe_mm": best_val,
        "final_mpjpe_mm": float(report["mpjpe"]),
        "final_pa_mpjpe_mm": float(report["pa_mpjpe"]),
        "checkpoint": str(best_path),
    }
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {report_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Data-efficiency curve: fine-tune a pretrained checkpoint on label fractions."
    )
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--train", type=str, nargs="+", default=None, help="Training .npz files")
    parser.add_argument("--val", type=str, default=None, help="Validation .npz file")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.25, 0.50, 1.00],
        help="Label fractions to evaluate (default: 5/10/25/50/100%%)",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/data_efficiency_curve")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--focal_max_scale", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000, help="Random clips per train sequence before subsampling")
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1%% subset CPU smoke test with tiny hyperparameters",
    )
    args = parser.parse_args()

    if args.smoke:
        print("[SMOKE] Running 1%% subset CPU smoke test")
        args.cpu = True
        args.d = 32
        args.n_st_layers = 2
        args.residual_hidden = 64
        args.epochs = 2
        args.batch_size = 2
        args.train_samples = 100
        args.fractions = [0.01]
        args.output_dir = "outputs/data_efficiency_curve_smoke"
        if not args.train:
            args.train = [
                "data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz"
            ]
        if not args.val:
            args.val = "data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz"

    if not args.train or not args.val:
        parser.error("--train and --val are required unless --smoke is set")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrained_path = Path(args.pretrained) if args.pretrained else None
    curve = []
    for fraction in args.fractions:
        summary = train_once(
            pretrained_path,
            args.train,
            args.val,
            fraction,
            args,
            out_dir,
            seed=args.seed,
        )
        curve.append(summary)

    curve_path = out_dir / "data_efficiency_curve.json"
    with open(curve_path, "w") as f:
        json.dump({"fractions": args.fractions, "curve": curve}, f, indent=2)
    print(f"\nData-efficiency curve saved to {curve_path}")

    print("\nFraction | Train clips | MPJPE (mm) | PA-MPJPE (mm)")
    print("---------|-------------|------------|--------------")
    for summary in curve:
        print(
            f"{summary['fraction']*100:6.0f}% | "
            f"{summary['train_clips']:11d} | "
            f"{summary['final_mpjpe_mm']:10.2f} | "
            f"{summary['final_pa_mpjpe_mm']:13.2f}"
        )


if __name__ == "__main__":
    main()
