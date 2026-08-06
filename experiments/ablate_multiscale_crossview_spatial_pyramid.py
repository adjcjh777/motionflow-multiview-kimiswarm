"""CPU/GPU smoke test for the multi-scale cross-view spatial pyramid model.

Instantiates the new
``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSpatialPyramid``
model, runs a few forward/backward steps on a small MPI-INF-3DHP smoke .npz,
compares shapes and parameter count with the anchor, and writes a short report.

Usage
-----
    conda run -n mf python experiments/ablate_multiscale_crossview_spatial_pyramid.py --smoke --max_batches 5
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

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_spatial_pyramid_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSpatialPyramid,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SmokeClipDataset(torch.utils.data.Dataset):
    """Yield short clips from a canonical multi-view .npz file."""

    def __init__(self, npz_path: str, clip_len: int, n_samples: int = 200):
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


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def main():
    parser = argparse.ArgumentParser(description="Smoke test: multi-scale cross-view spatial pyramid anchor model")
    parser.add_argument("--data_root", type=str, default="data/webbridge/mpi_inf_3dhp")
    parser.add_argument("--smoke", action="store_true", default=True, help="Use *_smoke.npz files")
    parser.add_argument("--full", dest="smoke", action="store_false", help="Use full .npz files")
    parser.add_argument("--clip_len", type=int, default=9)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4], help="Spatial pyramid downsample factors")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=5, help="Limit batches per epoch for smoke")
    parser.add_argument("--train_samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=str, default="docs/swarm_iter_next/ablate_multiscale_crossview_spatial_pyramid_report.md")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    data_root = Path(args.data_root)
    suffix = "_smoke" if args.smoke else ""
    npz_path = data_root / f"s_01_seq_01_v14_multiview_m{suffix}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing data file: {npz_path}")

    sample = np.load(npz_path)
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    total_frames = sample["points_2d"].shape[0]

    dataset = SmokeClipDataset(str(npz_path), args.clip_len, n_samples=args.train_samples)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Build anchor and pyramid models for parameter comparison.
    anchor = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=j,
        d=args.d,
        n_views=n_views,
        n_st_layers=2,
        residual_hidden=64,
    ).to(device)
    pyramid_model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSpatialPyramid(
        j=j,
        d=args.d,
        n_views=n_views,
        n_st_layers=2,
        residual_hidden=64,
        spatial_pyramid_scales=tuple(args.scales),
    ).to(device)

    anchor_params = sum(p.numel() for p in anchor.parameters() if p.requires_grad)
    pyramid_params = sum(p.numel() for p in pyramid_model.parameters() if p.requires_grad)

    # Forward shape sanity check on one batch.
    xb, yb, K, R, t = next(iter(loader))
    xb = xb.to(device)
    yb = yb.to(device)
    K = K.to(device)
    R = R.to(device)
    t = t.to(device)
    with torch.no_grad():
        pred_anchor, _ = anchor(xb, K=K, R=R, t=t)
        pred_pyramid, _ = pyramid_model(xb, K=K, R=R, t=t)
    assert pred_anchor.shape == yb.shape, (pred_anchor.shape, yb.shape)
    assert pred_pyramid.shape == yb.shape, (pred_pyramid.shape, yb.shape)

    optimizer = optim.Adam(pyramid_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("=" * 60)
    print("Multi-Scale Cross-View Spatial Pyramid smoke")
    print("=" * 60)
    print(f"Data file: {npz_path}")
    print(f"Frames: {total_frames}, Views: {n_views}, Joints: {j}, Clip len: {args.clip_len}")
    print(f"Device: {device}")
    print(f"Anchor params: {anchor_params:,}")
    print(f"Pyramid params: {pyramid_params:,} (delta: {pyramid_params - anchor_params:,})")
    print(f"Spatial pyramid scales: {args.scales}")
    print(f"Forward output shapes: anchor {tuple(pred_anchor.shape)}, pyramid {tuple(pred_pyramid.shape)}")

    # Run a short training loop.
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        pyramid_model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (xb, yb, K, R, t) in enumerate(loader, start=1):
            if batch_idx > args.max_batches:
                break
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            optimizer.zero_grad()
            pred, _ = pyramid_model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append((epoch, avg_loss))
        print(f"Epoch {epoch}: train_loss={avg_loss:.6f} ({n_batches} batches)")

    elapsed = time.time() - start_time

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Ablation / Smoke: Multi-Scale Cross-View Spatial Pyramid\n\n")
        f.write("Smoke run of the new spatial-pyramid anchor model.\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Data: `{npz_path}` (smoke={args.smoke})\n")
        f.write(f"- Frames: {total_frames}, Views: {n_views}, Joints: {j}, Clip len: {args.clip_len}\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Model d={args.d}, scales={args.scales}\n")
        f.write(f"- Anchor trainable parameters: {anchor_params:,}\n")
        f.write(f"- Pyramid trainable parameters: {pyramid_params:,} (+{pyramid_params - anchor_params:,})\n")
        f.write(f"- Epochs: {args.epochs}, max batches per epoch: {args.max_batches}, batch size: {args.batch_size}\n\n")
        f.write("## Smoke training loss\n\n")
        f.write("| Epoch | Train loss |\n")
        f.write("|------:|-----------|\n")
        for epoch, loss in history:
            f.write(f"| {epoch} | {loss:.6f} |\n")
        f.write(f"\nTotal elapsed time: {elapsed:.1f} s\n")

    print(f"\nReport written to {report_path}")
    print(f"Elapsed time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
