"""Train a mixed-dataset temporal residual model on MPI + H36M + AIST++.

This script is the unified, reusable counterpart to the earlier
``train_ray_attention_temporal_mixed_residual_v1.py`` demo.  It moves the
mixed-dataset loading logic into ``motionflow_mv.data.mixed_dataset`` and
demonstrates end-to-end training on the existing
``RayAttentionFusionModelTemporalMixedResidual`` model.

Example
-------
    python experiments/train_mixed_dataset.py \\
        --mpi_train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \\
        --aist_train data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch01_multiview.npz \\
        --h36m_train data/h36m_hf/s_01_acts_..._multiview_m.npz \\
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \\
        --val_dataset mpi \\
        --clip_len 13 --epochs 2 --d 32 --train_samples 500
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.mixed_dataset import (
    DATASET_REGISTRY,
    build_mixed_dataloaders,
)
from motionflow_mv.data.temporal_clip_dataset import set_seed
from motionflow_mv.fusion.ray_attention_temporal_mixed_residual_v1 import (
    RayAttentionFusionModelTemporalMixedResidual,
)


def augment_clip(
    x: torch.Tensor,
    noise_std: float = 0.5,
    dropout_rate: float = 0.1,
) -> torch.Tensor:
    """Lightweight per-clip augmentation."""
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (
            torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)
            > dropout_rate
        ).float()
        x[..., 2] = x[..., 2] * mask
    return x


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    for xb, yb, K, R, t, ids in loader:
        xb, yb = xb.to(device), yb.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        ids = ids.to(device)
        pred, mask = model(xb, K=K, R=R, t=t, dataset_ids=ids)
        err = (pred - yb).norm(dim=-1) * mask.float()
        total_err += err.sum().item()
        total_count += mask.sum().item()
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser(
        description="Train mixed-dataset temporal ray-attention fusion with residual refinement"
    )
    parser.add_argument(
        "--mpi_train", type=str, nargs="+", default=[], help="MPI-INF-3DHP train .npz files"
    )
    parser.add_argument(
        "--aist_train", type=str, nargs="+", default=[], help="AIST++ train .npz files"
    )
    parser.add_argument(
        "--h36m_train", type=str, nargs="+", default=[], help="Human3.6M train .npz files"
    )
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument(
        "--val_dataset",
        type=str,
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset name for the validation file",
    )
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--train_samples", type=int, default=500, help="Random clips per train sequence"
    )
    parser.add_argument("--val_stride", type=int, default=1, help="Stride for validation clips")
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="outputs/ray_attention_temporal_mixed_dataset.pth"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Override hyperparameters for a fast CPU smoke test",
    )
    args = parser.parse_args()

    if args.smoke:
        args.d = 8
        args.n_temporal_layers = 1
        args.residual_hidden = 16
        args.train_samples = 4
        args.batch_size = 2
        args.epochs = 1
        args.clip_len = 9
        print("Smoke mode: overriding hyperparameters to tiny values.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_paths = {
        "mpi": args.mpi_train,
        "aist": args.aist_train,
        "h36m": args.h36m_train,
    }

    train_loader, val_loader = build_mixed_dataloaders(
        train_paths=train_paths,
        val_path=args.val,
        val_dataset=args.val_dataset,
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_stride=args.val_stride,
    )

    model = RayAttentionFusionModelTemporalMixedResidual(
        d=args.d,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for xb, yb, K, R, t, ids in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            ids = ids.to(device)
            xb = augment_clip(xb, noise_std=args.noise_std, dropout_rate=args.dropout_rate)

            pred, mask = model(xb, K=K, R=R, t=t, dataset_ids=ids)
            loss = (((pred - yb) ** 2).sum(dim=-1) * mask.float()).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * mask.sum().item()
            train_count += mask.sum().item()

        train_loss /= train_count
        val_err = evaluate(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"val_MPJPE={val_err*1000:.2f}mm (saved)"
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"val_MPJPE={val_err*1000:.2f}mm"
            )

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
