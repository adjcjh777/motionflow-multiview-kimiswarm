"""Train MultiScaleTemporalConvModel on MPI-INF-3DHP clips.

Usage
-----
    conda run -n mf python experiments/train_multiscale_temporal_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
                data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --clip_len 13 --epochs 10

This script reuses the data pipeline from the transformer baseline and swaps
in a multi-scale temporal convolution head.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# The baseline script sits in the same directory and already appends the project
# root to sys.path, which makes all project packages importable.
from train_ray_attention_temporal_mpiinf3dhp import (
    set_seed,
    TemporalClipDataset,
    RandomClipDataset,
    collate_fn,
    augment_clip,
    evaluate,
)

from motionflow_mv.fusion.multiscale_temporal_conv_model import MultiScaleTemporalConvModel


def main():
    parser = argparse.ArgumentParser(description="Train multi-scale temporal convolution fusion on MPI-INF-3DHP")
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--temporal_kernel_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--temporal_dilations", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000, help="Random clips per train sequence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/multiscale_temporal_conv_mpiinf3dhp.pth")
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

    model = MultiScaleTemporalConvModel(
        j=j,
        d=args.d,
        n_views=n_views,
        n_temporal_layers=args.n_temporal_layers,
        temporal_kernel_sizes=args.temporal_kernel_sizes,
        temporal_dilations=args.temporal_dilations,
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
            torch.save(model.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
