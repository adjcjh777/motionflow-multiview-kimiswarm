"""Evaluate a trained ray_attention_v2 checkpoint on H36M and write a Markdown table.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/summarize_h36m_robustness.py \
        --dataset data/h36m_hf/s_01_acts_02_03_..._16_multiview.npz \
        --checkpoint outputs/ray_attention_v2_...pth \
        --output docs/results_h36m.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_v2_model import RayAttentionFusionModelV2
from motionflow_mv.fusion.triangulation import triangulate_dlt_torch


@torch.no_grad()
def evaluate(model, x, joints_3d, K, R, t):
    device = next(model.parameters()).device
    x = x.to(device)
    K = K.to(device)
    R = R.to(device)
    t = t.to(device)
    B = x.shape[0]
    pred, _ = model(x, K=K.expand(B, -1, -1, -1), R=R.expand(B, -1, -1, -1), t=t.expand(B, -1, -1))
    err = (pred - joints_3d.to(device)).norm(dim=-1).mean().item()
    return err


def dlt_error(points_2d, confidences, joints_3d, K, R, t):
    device = K.device
    P = K @ torch.cat([R, t[..., None]], dim=-1)
    B, V, J, _ = points_2d.shape
    X = torch.zeros((B, J, 3), dtype=points_2d.dtype, device=device)
    for b in range(B):
        for j in range(J):
            w = confidences[b, :, j]
            if w.sum() == 0:
                w = torch.ones_like(w)
            X[b, j] = triangulate_dlt_torch(points_2d[b, :, j], P, weights=w)
    err = (X - joints_3d).norm(dim=-1).mean().item()
    return err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 2.0, 5.0])
    parser.add_argument("--dropout_rates", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    parser.add_argument("--outlier_rates", type=float, nargs="+", default=[0.0, 0.05])
    parser.add_argument("--outlier_scale", type=float, default=100.0)
    parser.add_argument("--output", type=str, default="docs/results_h36m.md")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    points_2d_full = data["points_2d"]
    confidences_full = data["confidences"]
    joints_3d = torch.from_numpy(data["joints_3d"]).float()

    K = torch.from_numpy(data["camera_K"]).float().unsqueeze(0).to(device)
    R = torch.from_numpy(data["camera_R"]).float().unsqueeze(0).to(device)
    t = torch.from_numpy(data["camera_t"]).float().unsqueeze(0).to(device)

    n_views = data["camera_K"].shape[0]
    n_joints = data["points_2d"].shape[2]
    model = RayAttentionFusionModelV2(j=n_joints, d=args.d, n_views=n_views).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Reduce to a manageable subset for speed.
    max_frames = 5000
    if len(points_2d_full) > max_frames:
        rng = np.random.default_rng(2027)
        idx = rng.choice(len(points_2d_full), max_frames, replace=False)
        points_2d_full = points_2d_full[idx]
        confidences_full = confidences_full[idx]
        joints_3d = joints_3d[idx]

    rows = []
    for drop in args.dropout_rates:
        for noise in args.noise_levels:
            for outlier_rate in args.outlier_rates:
                p2d = points_2d_full.copy()
                conf = confidences_full.copy()
                if noise > 0:
                    p2d = p2d + np.random.randn(*p2d.shape).astype(np.float64) * noise
                if drop > 0:
                    mask = np.random.rand(p2d.shape[0], n_views, n_joints) > drop
                    conf = conf * mask
                if outlier_rate > 0:
                    out_mask = np.random.rand(p2d.shape[0], n_views, n_joints) < outlier_rate
                    outlier = (np.random.rand(p2d.shape[0], n_views, n_joints, 2) - 0.5) * 2 * args.outlier_scale
                    p2d = np.where(out_mask[..., None], outlier, p2d)

                x = torch.from_numpy(np.concatenate([p2d, conf[..., None], p2d[..., :1] * 0], axis=-1)[..., :3]).float()
                # Actually x should be (x,y,conf)
                x = torch.from_numpy(np.concatenate([p2d, conf[..., None]], axis=-1)).float()

                loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(x, joints_3d),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                preds = []
                for xb, _ in loader:
                    pred, _ = model(xb.to(device), K=K.expand(xb.size(0), -1, -1, -1), R=R.expand(xb.size(0), -1, -1, -1), t=t.expand(xb.size(0), -1, -1))
                    preds.append(pred)
                preds = torch.cat(preds, dim=0)
                model_err = (preds - joints_3d.to(device)).norm(dim=-1).mean().item()

                dlt_err = dlt_error(
                    torch.from_numpy(p2d).float(),
                    torch.from_numpy(conf).float(),
                    joints_3d,
                    K, R, t,
                )
                rows.append((drop, noise, outlier_rate, model_err, dlt_err))
                print(f"drop={drop:.1f} noise={noise:.2f} outlier={outlier_rate:.2f} -> model={model_err:.4f}  DLT={dlt_err:.4f}")

    # Write markdown table.
    lines = [
        "# H36M Robustness Results\n\n",
        "| drop | noise | outlier | ray_attention (mm) | DLT (mm) |\n",
        "|------|-------|---------|--------------------|----------|\n",
    ]
    for drop, noise, outlier, model_err, dlt_err in rows:
        lines.append(f"| {drop:.1f} | {noise:.2f} | {outlier:.2f} | {model_err:.4f} | {dlt_err:.4f} |\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("".join(lines), encoding="utf-8")
    print(f"\nSaved table to {args.output}")


if __name__ == "__main__":
    main()
