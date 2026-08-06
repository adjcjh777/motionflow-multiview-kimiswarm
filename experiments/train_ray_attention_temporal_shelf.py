"""Train RayAttentionFusionModelTemporal on Shelf/Campus temporal clips.

Usage
-----
    # Smoke test on Shelf (5 views) with tiny settings (~5 min on RTX 4090)
    conda run -n mf python experiments/train_ray_attention_temporal_shelf.py

    # Longer run
    conda run -n mf python experiments/train_ray_attention_temporal_shelf.py \
        --epochs 10 --d 128 --batch_size 4 --clip_len 27 --train_samples 4000

The script expects canonical WebBridge .npz files produced by
``experiments/convert_shelf_campus_temporal.py``.  By default it trains on the
Shelf train split and validates on the Shelf val split.  Use the ``--val``
argument to point at the Campus val split if your model supports 3 views.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.temporal_clip_dataset import (
    augment_clip,
    make_dataloaders,
    set_seed,
)
from motionflow_mv.fusion.ray_attention_temporal_model import (
    RayAttentionFusionModelTemporal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate(model, loader, device, criterion):
    """Compute per-epoch validation MPJPE (mean per-joint position error)."""
    model.eval()
    total_err = 0.0
    total_count = 0
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_loss += criterion(pred, yb).item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count, total_loss / total_count


def main():
    parser = argparse.ArgumentParser(
        description="Train temporal ray-attention fusion on Shelf/Campus"
    )
    parser.add_argument("--train", type=str, nargs="+",
                        default=[str(PROJECT_ROOT / "data/webbridge/shelf_campus/shelf_seq1_train_v5_multiview_m.npz")],
                        help="Train .npz files")
    parser.add_argument("--val", type=str,
                        default=str(PROJECT_ROOT / "data/webbridge/shelf_campus/shelf_seq1_val_v5_multiview_m.npz"),
                        help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_samples", type=int, default=500,
                        help="Random clips sampled per train sequence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="outputs/ray_attention_temporal_shelf.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = make_dataloaders(
        args.train,
        args.val,
        args.clip_len,
        args.batch_size,
        train_samples=args.train_samples,
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

        val_err, val_loss = evaluate(model, val_loader, device, criterion)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"val_MPJPE={val_err*1000:.2f}mm, val_loss={val_loss:.6f} (saved)"
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"val_MPJPE={val_err*1000:.2f}mm, val_loss={val_loss:.6f}"
            )

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
