"""Robustness evaluation for an OmniMultiViewFusionV5 checkpoint.

Loads a checkpoint (and the adjacent ``.config.json``), runs the trained model on the
validation set, and reports:

* standard full-view MPJPE / PA-MPJPE / PCK,
* per-view-count MPJPE (2, 4, 8, 14 active views),
* MPJPE under increasing outlier-view corruption.

Example::

    python experiments/eval_omniview_fusion_v5_robustness.py \
        --checkpoint outputs/omniview_fusion_v10_aleatoric_outlier.pth \
        --out_dir outputs/eval_v10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.benchmark_protocol import BenchmarkConfig, BenchmarkProtocol
from motionflow_mv.eval.metrics import compute_all_metrics

# Reuse helpers from the training script.
from train_omniview_fusion_v5_webbridge_multi import (
    build_datasets,
    build_model_from_args,
    inject_outlier_views,
)


def _move_to_device(batch: Tuple[torch.Tensor, ...], device: torch.device) -> Tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in batch)


def _forward_with_mask(
    model: torch.nn.Module,
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
    view_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run model on a batch with an optional view mask and domain id."""
    if len(batch) == 6:
        x, y, K, R, t, dataset_id = batch
    else:
        x, y, K, R, t = batch
        dataset_id = None

    x = x.to(device)
    K = K.to(device)
    R = R.to(device)
    t = t.to(device)

    kwargs: Dict[str, Any] = {}
    if view_mask is not None:
        kwargs["view_mask"] = view_mask.to(device)
    if dataset_id is not None and getattr(model, "use_domain_embedding", False):
        kwargs["domain_id"] = dataset_id.to(device)

    out = model(x, K=K, R=R, t=t, **kwargs)
    return out[0]


def _sample_view_mask(shape: Tuple[int, int, int], k: int) -> torch.Tensor:
    """Create a binary mask with exactly ``k`` active views for each clip.

    Args:
        shape: (B, T, V) shape of the mask to create.
        k: number of active views.

    Returns:
        (B, T, V) float mask.
    """
    B, T, V = shape
    k = max(1, min(k, V))
    mask = torch.zeros(B, T, V)
    for i in range(B):
        idx = torch.randperm(V)[:k]
        mask[i, :, idx] = 1.0
    return mask


def evaluate_full_view(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Baseline evaluation using all available views."""
    protocol = BenchmarkProtocol(BenchmarkConfig(dataset="webbridge", split="val"))
    return protocol.evaluate_model(model, dataloader, device)


def evaluate_variable_views(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    view_counts: List[int],
) -> Dict[int, Dict[str, Any]]:
    """Evaluate the model with a fixed number of active views per clip.

    For each view count ``k`` a random subset of ``k`` views is activated per
    clip and the mean per-frame MPJPE is reported.
    """
    model.eval()
    results: Dict[int, Dict[str, Any]] = {}

    for k in view_counts:
        preds, gts = [], []
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 6:
                    x, y, K, R, t, dataset_id = batch
                else:
                    x, y, K, R, t = batch
                    dataset_id = None

                B, T = x.shape[0], x.shape[1]
                V = x.shape[2]
                view_mask = _sample_view_mask((B, T, V), k)

                pred = _forward_with_mask(model, batch, device, view_mask=view_mask)
                preds.append(pred.cpu().numpy().reshape(-1, pred.shape[-2], 3))
                gts.append(y.numpy().reshape(-1, y.shape[-2], 3))

        pred = np.concatenate(preds, axis=0)
        gt = np.concatenate(gts, axis=0)
        pred_mm = pred * 1000.0
        gt_mm = gt * 1000.0
        report = compute_all_metrics(pred_mm, gt_mm)
        results[k] = {
            "mpjpe": float(report["mpjpe"]),
            "pa_mpjpe": float(report["pa_mpjpe"]),
            "n_frames": pred.shape[0],
        }

    return results


def evaluate_outlier_robustness(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    levels: List[Dict[str, float]],
) -> Dict[float, Dict[str, Any]]:
    """Evaluate MPJPE when a random view is corrupted with outlier noise.

    ``levels`` is a list of dicts with ``offset_std`` and ``noise_std``.
    The key in the returned dict is the offset standard deviation.
    """
    model.eval()
    results: Dict[float, Dict[str, Any]] = {}

    for level in levels:
        offset_std = level["offset_std"]
        noise_std = level["noise_std"]
        preds, gts = [], []

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0], batch[1]
                x_corr = inject_outlier_views(
                    x.to(device) if x.device != device else x,
                    prob=1.0,
                    max_views=1,
                    offset_std=offset_std,
                    noise_std=noise_std,
                    min_views=2,
                )
                # Replace x in the batch with corrupted version.
                if len(batch) == 6:
                    batch_corr = (x_corr, batch[1], batch[2], batch[3], batch[4], batch[5])
                else:
                    batch_corr = (x_corr, batch[1], batch[2], batch[3], batch[4])

                pred = _forward_with_mask(model, batch_corr, device, view_mask=None)
                preds.append(pred.cpu().numpy().reshape(-1, pred.shape[-2], 3))
                gts.append(y.numpy().reshape(-1, y.shape[-2], 3))

        pred = np.concatenate(preds, axis=0)
        gt = np.concatenate(gts, axis=0)
        pred_mm = pred * 1000.0
        gt_mm = gt * 1000.0
        report = compute_all_metrics(pred_mm, gt_mm)
        results[offset_std] = {
            "mpjpe": float(report["mpjpe"]),
            "pa_mpjpe": float(report["pa_mpjpe"]),
            "n_frames": pred.shape[0],
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OmniMultiViewFusionV5 robustness")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write results.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_clips", type=int, default=None, help="Limit number of validation clips")
    parser.add_argument(
        "--view_counts",
        type=int,
        nargs="+",
        default=[2, 4, 8, 14],
        help="Active view counts to evaluate",
    )
    parser.add_argument(
        "--outlier_offsets",
        type=float,
        nargs="+",
        default=[0.0, 10.0, 25.0, 50.0, 100.0],
        help="Outlier per-view offset standard deviations (pixels)",
    )
    parser.add_argument(
        "--outlier_noises",
        type=float,
        nargs="+",
        default=[0.0, 15.0, 30.0, 60.0, 100.0],
        help="Outlier per-pixel noise standard deviations (pixels)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    config_path = checkpoint_path.with_suffix(".config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config file next to checkpoint: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Convert config back to a Namespace for the helper functions.
    train_args = argparse.Namespace(**config)
    # Make sure the model is built with the same architecture, not trained further.
    train_args.warm_start = None
    train_args.warm_start_freeze_epochs = 0

    # Build val dataset using the same pipeline as training.
    train_dataset, val_dataset, n_views, n_joints = build_datasets(train_args)
    val_subset = val_dataset
    if args.n_clips is not None and args.n_clips < len(val_dataset):
        indices = torch.randperm(len(val_dataset))[: args.n_clips].tolist()
        val_subset = torch.utils.data.Subset(val_dataset, indices)

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: tuple(torch.stack(b, dim=0) for b in zip(*batch)),
        num_workers=0,
    )

    model = build_model_from_args(train_args, n_joints, n_views, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded checkpoint {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline.
    print("Evaluating full-view baseline...")
    baseline = evaluate_full_view(model, val_loader, device)

    # 2. Variable view counts.
    print(f"Evaluating variable views {args.view_counts}...")
    variable_views = evaluate_variable_views(model, val_loader, device, args.view_counts)

    # 3. Outlier robustness.
    levels = [
        {"offset_std": off, "noise_std": noise}
        for off, noise in zip(args.outlier_offsets, args.outlier_noises)
    ]
    print(f"Evaluating outlier robustness with {len(levels)} corruption levels...")
    outlier = evaluate_outlier_robustness(model, val_loader, device, levels)

    report = {
        "checkpoint": str(checkpoint_path),
        "config": config,
        "baseline": {
            "mpjpe": float(baseline["mpjpe"]),
            "pa_mpjpe": float(baseline["pa_mpjpe"]),
        },
        "variable_views": variable_views,
        "outlier_robustness": outlier,
    }

    report_path = out_dir / "robustness_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved robustness report to {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
