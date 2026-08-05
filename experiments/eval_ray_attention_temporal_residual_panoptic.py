"""Evaluate the residual temporal ray-attention model on a CMU Panoptic sample.

Loads ``RayAttentionFusionModelTemporalResidual`` from the provided checkpoint
and runs it on the canonical ``.npz`` produced by ``experiments/convert_panoptic_v1.py``.
The script reports MPJPE, PA-MPJPE, PCK, and AUC, and writes a JSON summary.

Example
-------
    conda run -n mf python experiments/eval_ray_attention_temporal_residual_panoptic.py \
        --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
        --panoptic data/webbridge/panoptic/171204_pose1_sample/171204_pose1_sample_canonical.npz \
        --clip_len 13 --batch_size 8 \
        --out outputs/eval_residual_panoptic.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics, summarize_metrics
from motionflow_mv.fusion.ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
)


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield clips (T, V, J, 3) from a canonical multi-view ``.npz``."""

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


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate residual temporal ray-attention model on CMU Panoptic sample"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--panoptic", type=str, required=True, help="Path to Panoptic canonical .npz")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out", type=str, default="outputs/eval_residual_panoptic.json",
                        help="JSON path to write metric report")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Infer dimensions from the Panoptic sample.
    data = np.load(args.panoptic)
    n_views = int(data["camera_K"].shape[0])
    j = int(data["points_2d"].shape[2])
    print(f"Data: {args.panoptic}")
    print(f"n_views={n_views}, n_joints={j}, clip_len={args.clip_len}, d={args.d}")

    # Build dataset.
    val_dataset = TemporalClipDataset(args.panoptic, args.clip_len, stride=1)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model.
    model = RayAttentionFusionModelTemporalResidual(
        j=j,
        d=args.d,
        n_views=n_views,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    all_pred = []
    all_gt = []

    with torch.no_grad():
        for xb, yb, K, R, t in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)

            pred, _ = model(xb, K=K, R=R, t=t)

            all_pred.append(pred.cpu().numpy())
            all_gt.append(yb.cpu().numpy())

    # Concatenate over all clips and flatten time/batch dimensions.
    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    n_clips, T, J, _ = all_pred.shape
    pred_flat = all_pred.reshape(-1, J, 3)
    gt_flat = all_gt.reshape(-1, J, 3)

    # Convert meters to millimeters for reporting.
    pred_mm = pred_flat * 1000.0
    gt_mm = gt_flat * 1000.0

    report = compute_all_metrics(pred_mm, gt_mm)

    summary = {
        "checkpoint": str(args.checkpoint),
        "panoptic_dataset": str(args.panoptic),
        "n_clips": int(n_clips),
        "clip_len": int(T),
        "n_joints": int(J),
        "n_views": int(n_views),
        "d": int(args.d),
        "n_temporal_layers": int(args.n_temporal_layers),
        "residual_hidden": int(args.residual_hidden),
        "mpjpe_mm": float(report["mpjpe"]),
        "pa_mpjpe_mm": float(report["pa_mpjpe"]),
        "pck_50mm": float(report.get("pck@50mm", None)),
        "pck_100mm": float(report.get("pck@100mm", None)),
        "pck_150mm": float(report.get("pck@150mm", None)),
        "pck_auc_150mm": float(report["pck_auc"]),
        "per_joint_mpjpe_mm": report["per_joint_mpjpe"].tolist() if "per_joint_mpjpe" in report else None,
        "per_joint_pa_mpjpe_mm": report["per_joint_pa_mpjpe"].tolist() if "per_joint_pa_mpjpe" in report else None,
    }

    print("\n=== Evaluation summary ===")
    print(f"Clips evaluated: {n_clips}  (T={T}, J={J})")
    print(f"MPJPE:  {summary['mpjpe_mm']:.4f} mm")
    print(f"PA-MPJPE: {summary['pa_mpjpe_mm']:.4f} mm")
    print(f"PCK@50mm:  {summary['pck_50mm']:.4f}")
    print(f"PCK@100mm: {summary['pck_100mm']:.4f}")
    print(f"PCK@150mm: {summary['pck_150mm']:.4f}")
    print(f"PCK AUC (0-150mm): {summary['pck_auc_150mm']:.4f}")
    print("\nDetailed metric report:")
    print(summarize_metrics(report))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON summary to {out_path}")


if __name__ == "__main__":
    main()
