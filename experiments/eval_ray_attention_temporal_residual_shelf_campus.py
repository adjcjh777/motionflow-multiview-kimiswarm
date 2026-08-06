"""Evaluate RayAttentionFusionModelTemporalResidual on Shelf/Campus canonical data.

If a checkpoint does not exist, the script performs a short smoke-run training on
the corresponding train split and then evaluates MPJPE on the validation split.

Usage
-----
    # Evaluate Shelf (train a 3-epoch smoke model if checkpoint is missing)
    conda run -n mf python experiments/eval_ray_attention_temporal_residual_shelf_campus.py --dataset shelf

    # Evaluate Campus
    conda run -n mf python experiments/eval_ray_attention_temporal_residual_shelf_campus.py --dataset campus

    # Specify your own checkpoint / data
    conda run -n mf python experiments/eval_ray_attention_temporal_residual_shelf_campus.py \
        --dataset shelf \
        --checkpoint outputs/ray_attention_temporal_residual_shelf.pth \
        --val data/webbridge/shelf_campus/shelf_seq1_val_v5_multiview_m.npz \
        --clip_len 13 --d 64 --epochs 3

The script reports per-clip and per-frame MPJPE in millimetres.
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
    TemporalClipDataset,
    RandomClipDataset,
    collate_fn,
    set_seed,
    augment_clip,
)
from motionflow_mv.fusion.ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "webbridge" / "shelf_campus"

DEFAULTS = {
    "shelf": {
        "train": DATA_ROOT / "shelf_seq1_train_v5_multiview_m.npz",
        "val": DATA_ROOT / "shelf_seq1_val_v5_multiview_m.npz",
        "checkpoint": PROJECT_ROOT / "outputs" / "ray_attention_temporal_residual_shelf.pth",
        "n_views": 5,
    },
    "campus": {
        "train": DATA_ROOT / "campus_seq1_train_v3_multiview_m.npz",
        "val": DATA_ROOT / "campus_seq1_val_v3_multiview_m.npz",
        "checkpoint": PROJECT_ROOT / "outputs" / "ray_attention_temporal_residual_campus.pth",
        "n_views": 3,
    },
}


def make_dataloaders(train_path: str, val_path: str, clip_len: int,
                     batch_size: int, train_samples: int = 500):
    """Build train/val loaders from canonical .npz paths."""
    train_dataset = RandomClipDataset(train_path, clip_len, n_samples=train_samples)
    val_dataset = TemporalClipDataset(val_path, clip_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    return train_loader, val_loader


def train_smoke(train_path: str, val_path: str, n_views: int,
                clip_len: int, d: int, n_temporal_layers: int,
                residual_hidden: int, batch_size: int, epochs: int,
                lr: float, train_samples: int, output_path: Path,
                device: torch.device, seed: int = 42) -> RayAttentionFusionModelTemporalResidual:
    """Run a short residual model training run and save the best checkpoint."""
    set_seed(seed)

    train_loader, val_loader = make_dataloaders(
        train_path, val_path, clip_len, batch_size, train_samples=train_samples
    )

    sample = np.load(train_path)
    n_views_data = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    if n_views is not None and n_views != n_views_data:
        raise ValueError(
            f"Requested n_views={n_views} but data has {n_views_data} views"
        )

    print(
        f"Training residual model: n_views={n_views_data}, j={j}, "
        f"clip_len={clip_len}, d={d}, residual_hidden={residual_hidden}, "
        f"epochs={epochs}"
    )

    model = RayAttentionFusionModelTemporalResidual(
        j=j,
        d=d,
        n_views=n_views_data,
        n_temporal_layers=n_temporal_layers,
        residual_hidden=residual_hidden,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, epochs + 1):
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
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"val_MPJPE={val_err * 1000:.2f}mm (saved)"
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"val_MPJPE={val_err * 1000:.2f}mm"
            )

    print(f"Best val MPJPE: {best_val * 1000:.2f}mm -> {output_path}")
    # Load best checkpoint for final evaluation.
    model.load_state_dict(torch.load(output_path, map_location=device, weights_only=True))
    return model


def evaluate(model, loader, device):
    """Return mean MPJPE (in metres) over the validation loader."""
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


def evaluate_per_frame(model, val_path: str, clip_len: int,
                       batch_size: int, device: torch.device):
    """Evaluate on every full clip and report per-frame MPJPE."""
    dataset = TemporalClipDataset(val_path, clip_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=0,
    )
    model.eval()
    all_errors = []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            # Per-frame, per-joint Euclidean distance in metres.
            err = (pred - yb).norm(dim=-1)  # (B, T, J)
            all_errors.append(err.cpu().numpy())
    all_errors = np.concatenate(all_errors, axis=0)  # (N, T, J)
    return {
        "clip_mpjpe_m": float(all_errors.mean()),
        "clip_mpjpe_mm": float(all_errors.mean() * 1000),
        "frame_mpjpe_mm": {
            i: float(all_errors[:, i, :].mean() * 1000)
            for i in range(all_errors.shape[1])
        },
    }


def load_or_train(args, dataset_key: str, device: torch.device):
    """Load a checkpoint if it exists; otherwise train a smoke-run model."""
    config = DEFAULTS[dataset_key]
    train_path = args.train or str(config["train"])
    val_path = args.val or str(config["val"])
    checkpoint = args.checkpoint or str(config["checkpoint"])
    n_views = config["n_views"]
    checkpoint_path = Path(checkpoint)

    if not checkpoint_path.exists() or args.retrain:
        print(f"Checkpoint {checkpoint_path} not found or --retrain set; training.")
        model = train_smoke(
            train_path=train_path,
            val_path=val_path,
            n_views=n_views,
            clip_len=args.clip_len,
            d=args.d,
            n_temporal_layers=args.n_temporal_layers,
            residual_hidden=args.residual_hidden,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            train_samples=args.train_samples,
            output_path=checkpoint_path,
            device=device,
            seed=args.seed,
        )
    else:
        print(f"Loading checkpoint from {checkpoint_path}")
        sample = np.load(train_path)
        n_views_data = sample["camera_K"].shape[0]
        j = sample["points_2d"].shape[2]
        model = RayAttentionFusionModelTemporalResidual(
            j=j,
            d=args.d,
            n_views=n_views_data,
            n_temporal_layers=args.n_temporal_layers,
            residual_hidden=args.residual_hidden,
        ).to(device)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
    return model, train_path, val_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate residual temporal ray-attention model on Shelf/Campus."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["shelf", "campus", "both"],
        default="both",
        help="Which dataset(s) to evaluate."
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path.")
    parser.add_argument("--train", type=str, default=None,
                        help="Override train .npz path.")
    parser.add_argument("--val", type=str, default=None,
                        help="Override validation .npz path.")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3,
                        help="Smoke-run epochs when training is needed.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_samples", type=int, default=500,
                        help="Random clips per epoch during smoke training.")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if checkpoint exists.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    datasets = ["shelf", "campus"] if args.dataset == "both" else [args.dataset]
    results = {}

    for dataset_key in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_key.upper()}")
        print(f"{'=' * 60}")

        model, train_path, val_path = load_or_train(args, dataset_key, device)

        # Standard validation MPJPE (same metric used during training).
        _, val_loader = make_dataloaders(
            train_path, val_path, args.clip_len, args.batch_size, train_samples=args.train_samples
        )
        mpjpe_m = evaluate(model, val_loader, device)
        mpjpe_mm = mpjpe_m * 1000

        # Per-frame breakdown.
        frame_report = evaluate_per_frame(
            model, val_path, args.clip_len, args.batch_size, device
        )

        print(f"\nValidation MPJPE: {mpjpe_mm:.2f} mm")
        print(f"Per-frame MPJPE (mm): {frame_report['frame_mpjpe_mm']}")

        results[dataset_key] = {
            "mpjpe_mm": round(mpjpe_mm, 2),
            "frame_mpjpe_mm": frame_report["frame_mpjpe_mm"],
        }

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for k, v in results.items():
        print(f"{k.upper()}: {v['mpjpe_mm']:.2f} mm")


if __name__ == "__main__":
    main()
