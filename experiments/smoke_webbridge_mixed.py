"""CPU smoke test for the WebBridge mixed-dataset loader + PP anchor.

Runs one epoch of the principal-point residual model on a tiny mixed batch
containing H36M, MPI-INF-3DHP, and AIST++ canonical ``.npz`` files.  The goal is
a clean run, not good metrics.
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

from motionflow_mv.data.webbridge_mixed_dataset import (
    WebBridgeMixedDataset,
    webbridge_mixed_collate_fn,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="WebBridge mixed-dataset CPU smoke")
    parser.add_argument("--train_paths", type=str, nargs="+", required=True)
    parser.add_argument("--train_names", type=str, nargs="+", required=True)
    parser.add_argument("--val_paths", type=str, nargs="+", required=True)
    parser.add_argument("--val_names", type=str, nargs="+", required=True)
    parser.add_argument("--clip_len", type=int, default=9)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--n_st_layers", type=int, default=1)
    parser.add_argument("--residual_hidden", type=int, default=64)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_samples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    train_dataset = WebBridgeMixedDataset(
        args.train_paths,
        args.train_names,
        args.clip_len,
        n_samples=args.train_samples,
    )
    val_dataset = WebBridgeMixedDataset(
        args.val_paths,
        args.val_names,
        args.clip_len,
        n_samples=None,
        stride=5,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=webbridge_mixed_collate_fn,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=webbridge_mixed_collate_fn,
        num_workers=0,
    )

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=17,
        d=args.d,
        n_views=14,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=20.0,
        focal_max_scale=0.0,
        return_pp_delta=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(
        f"Smoke model: n_views=14, j=17, d={args.d}, "
        f"n_st_layers={args.n_st_layers}, residual_hidden={args.residual_hidden}"
    )
    print(f"Train clips: {len(train_dataset)}  Val clips: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb, K, R, t, dataset_ids in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)

            optimizer.zero_grad()
            pred, *_ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)
            print(
                f"  batch loss={loss.item():.6f}  shapes "
                f"x={tuple(xb.shape)} y={tuple(yb.shape)} "
                f"datasets={dataset_ids.tolist()}"
            )

        avg_train_loss = total_loss / total_samples if total_samples else 0.0

        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for xb, yb, K, R, t, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                K, R, t = K.to(device), R.to(device), t.to(device)
                pred, *_ = model(xb, K=K, R=R, t=t)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_samples += xb.size(0)
        avg_val_loss = val_loss / val_samples if val_samples else 0.0

        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.6f} "
            f"val_loss={avg_val_loss:.6f}"
        )

    print("Smoke completed successfully.")


if __name__ == "__main__":
    main()
