"""Evaluate the pre-trained residual temporal ray-attention model on converted
3DPW pseudo-multi-view data.

The script loads the checkpoint ``outputs/ray_attention_temporal_residual_v2.pth``
(which uses ``RayAttentionFusionModelTemporalResidual``) and runs it on the
pseudo-multi-view 3DPW sequences produced by ``experiments/convert_3dpw_multiview.py``.

Important: the published residual checkpoint was trained on MPI-INF-3DHP with
14 camera views.  To run it here we converted the 3DPW sequences with
``--n_views 14``; the original 4-view conversion triggers a ``fusion_mlp`` shape
mismatch and cannot be used directly.

Usage
-----
    conda run -n mf python experiments/eval_residual_3dpw_pseudo.py \
        --split data/webbridge/3dpw/converted_14views/validation \
        --clip_len 13 --batch_size 8 --save_json outputs/residual_3dpw_val.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
)


class PseudoClipDataset(torch.utils.data.Dataset):
    """Yield fixed-length clips from a converted 3DPW .npz."""

    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()  # (T, V, J, 2)
        self.confidences = torch.from_numpy(data["confidences"]).float()  # (T, V, J)
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()  # (T, J, 3)
        self.K = torch.from_numpy(data["camera_K"]).float()  # (V, 3, 3)
        self.R = torch.from_numpy(data["camera_R"]).float()  # (V, 3, 3)
        self.t = torch.from_numpy(data["camera_t"]).float()  # (V, 3)

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
            [
                self.points_2d[start:end],
                self.confidences[start:end].unsqueeze(-1),
            ],
            dim=-1,
        )  # (T, V, J, 3)
        y = self.joints_3d[start:end]  # (T, J, 3)
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def compute_metrics(pred, gt):
    """Return raw and root-aligned MPJPE (meters -> mm)."""
    # pred, gt: (B, T, J, 3)
    raw = pred - gt
    raw_err = raw.norm(dim=-1).mean() * 1000.0

    # Root align at pelvis (joint 0).
    pred_r = pred - pred[:, :, 0:1, :]
    gt_r = gt - gt[:, :, 0:1, :]
    root_err = (pred_r - gt_r).norm(dim=-1).mean() * 1000.0
    return raw_err.item(), root_err.item()


def evaluate_sequence(model, loader, device):
    model.eval()
    total_raw = 0.0
    total_root = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            raw_err, root_err = compute_metrics(pred, yb)
            n = xb.size(0)
            total_raw += raw_err * n
            total_root += root_err * n
            total_count += n
    return total_raw / total_count, total_root / total_count


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate residual temporal model on 3DPW pseudo-multi-view data."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="data/webbridge/3dpw/converted/validation",
        help="Directory containing converted _pseudo.npz files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/ray_attention_temporal_residual_v2.pth",
        help="Path to the residual temporal checkpoint.",
    )
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--save_json", type=str, default=None, help="Optional JSON path to write per-sequence results.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split_dir = Path(args.split)
    npz_files = sorted(split_dir.glob("*_pseudo.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_pseudo.npz files found in {split_dir}")

    # Use the first file only to infer skeleton / camera geometry.
    sample_data = np.load(npz_files[0])
    n_views = sample_data["camera_K"].shape[0]
    j = sample_data["points_2d"].shape[2]
    print(f"Sequences: {len(npz_files)}, n_views={n_views}, j={j}, clip_len={args.clip_len}")

    model = RayAttentionFusionModelTemporalResidual(
        j=j,
        d=args.d,
        n_views=n_views,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # The published residual checkpoint was trained on MPI-INF-3DHP with 14
    # camera views.  It contains an few extra keys (e.g. `fusion_mlp`) that are
    # not part of the current residual model, so we load non-strictly and only
    # use the matching parameters.
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    print(f"  Loaded parameters: {len(model.state_dict())} modules, {sum(p.numel() for p in model.parameters())} params")
    model.eval()

    results = []
    for npz_file in npz_files:
        dataset = PseudoClipDataset(str(npz_file), args.clip_len, stride=args.clip_len)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        raw_mpjpe, root_mpjpe = evaluate_sequence(model, loader, device)
        results.append({
            "sequence": npz_file.stem,
            "raw_mm": raw_mpjpe,
            "root_aligned_mm": root_mpjpe,
            "n_clips": len(dataset),
        })
        print(f"{npz_file.stem}: raw={raw_mpjpe:.2f} mm, root-aligned={root_mpjpe:.2f} mm, clips={len(dataset)}")

    raw_mean = np.mean([r["raw_mm"] for r in results])
    root_mean = np.mean([r["root_aligned_mm"] for r in results])
    print(f"\n=== 3DPW pseudo-multi-view summary ({len(results)} sequences) ===")
    print(f"Mean raw MPJPE:       {raw_mean:.2f} mm")
    print(f"Mean root-aligned MPJPE: {root_mean:.2f} mm")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "split": str(args.split),
                    "checkpoint": str(args.checkpoint),
                    "clip_len": args.clip_len,
                    "mean_raw_mm": raw_mean,
                    "mean_root_aligned_mm": root_mean,
                    "sequences": results,
                },
                indent=2,
            )
        )
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
