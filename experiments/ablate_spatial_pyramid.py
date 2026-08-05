"""CPU smoke test for the spatial feature pyramid module.

Builds the tiny ``SpatialFeaturePyramidModel`` around the new
``SpatialFeaturePyramid`` block, runs a handful of training iterations on a
smoke ``.npz`` file using only the CPU, and reports the intermediate feature map
shapes produced by the pyramid.

Usage
-----
    conda run -n mf python experiments/ablate_spatial_pyramid.py --smoke --max_batches 5
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

from motionflow_mv.models.spatial_feature_pyramid import (
    SpatialFeaturePyramid,
    SpatialFeaturePyramidModel,
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
        return x, y


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    return x, y


def report_shapes(model: SpatialFeaturePyramidModel, x: torch.Tensor):
    """Run the pyramid once and print the shape of every branch."""
    with torch.no_grad():
        z = model.obs_embed(x)
        _, shapes = model.sfp(z, return_shapes=True)
    return shapes


def main():
    parser = argparse.ArgumentParser(description="CPU smoke: spatial feature pyramid")
    parser.add_argument("--data_root", type=str, default="data/webbridge/mpi_inf_3dhp")
    parser.add_argument("--smoke", action="store_true", default=True, help="Use *_smoke.npz files")
    parser.add_argument("--full", dest="smoke", action="store_false", help="Use full .npz files")
    parser.add_argument("--clip_len", type=int, default=5)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--num_scales", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=5, help="Limit batches per epoch for CPU smoke")
    parser.add_argument("--train_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--report", type=str, default="docs/swarm_iter_next/ablate_spatial_pyramid_report.md")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

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

    model = SpatialFeaturePyramidModel(
        j=j,
        d=args.d,
        n_views=n_views,
        num_scales=args.num_scales,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Report shapes on a single sample.
    with torch.no_grad():
        dummy_x = torch.cat(
            [
                torch.from_numpy(sample["points_2d"][: args.clip_len]).float().unsqueeze(0),
                torch.from_numpy(sample["confidences"][: args.clip_len]).float().unsqueeze(0).unsqueeze(-1),
            ],
            dim=-1,
        )
        # Ensure shape (1, T, V, J, 3)
        dummy_shapes = report_shapes(model, dummy_x)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("=" * 60)
    print("Spatial Feature Pyramid CPU smoke")
    print("=" * 60)
    print(f"Data file: {npz_path}")
    print(f"Frames: {total_frames}, Views: {n_views}, Joints: {j}, Clip len: {args.clip_len}")
    print(f"Model d={args.d}, num_scales={args.num_scales}, params={n_params:,}")
    print("\nPyramid feature map shapes (per scale) on a single clip:")
    print(f"  input: {tuple(dummy_x.shape)}")
    for i, shape in enumerate(dummy_shapes, start=1):
        print(f"  scale {i} conv output: {shape}  -> upsampled to (B*C_out, {shape[1]}, {j})")
    print()

    # Run a short training loop.
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            if batch_idx > args.max_batches:
                break
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
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
        f.write("# Ablation / Smoke: Spatial Feature Pyramid\n\n")
        f.write("CPU-only smoke run of the new ``SpatialFeaturePyramid`` module.\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Data: `{npz_path}` (smoke={args.smoke})\n")
        f.write(f"- Frames: {total_frames}, Views: {n_views}, Joints: {j}, Clip len: {args.clip_len}\n")
        f.write(f"- Model d={args.d}, num_scales={args.num_scales}\n")
        f.write(f"- Trainable parameters: {n_params:,}\n")
        f.write(f"- Epochs: {args.epochs}, max batches per epoch: {args.max_batches}, batch size: {args.batch_size}\n\n")
        f.write("## Pyramid feature-map shapes\n\n")
        f.write(f"Input clip shape: `{tuple(dummy_x.shape)}` (B, T, V, J, 3).\n\n")
        f.write("Per-scale conv output shapes (before upsampling) and final upsampled target:\n\n")
        f.write("| Scale | Shape (N, C_out, target_J) | Upsampled length |\n")
        f.write("|------:|----------------------------|-----------------|\n")
        for i, shape in enumerate(dummy_shapes, start=1):
            f.write(f"| {i} | `{shape}` | {j} |\n")
        f.write("\n## Smoke training loss\n\n")
        f.write("| Epoch | Train loss |\n")
        f.write("|------:|-----------|\n")
        for epoch, loss in history:
            f.write(f"| {epoch} | {loss:.6f} |\n")
        f.write(f"\nTotal elapsed time: {elapsed:.1f} s\n")

    print(f"\nReport written to {report_path}")
    print(f"Elapsed time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
