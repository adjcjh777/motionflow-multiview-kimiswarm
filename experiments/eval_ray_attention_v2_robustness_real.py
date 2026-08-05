"""Robustness evaluation of ray_attention vs DLT on real Shelf/Campus GT.

Adds 2D keypoint noise and random view dropout, then compares MPJPE.
Example:
    /d/anaconda3/envs/jz_py310/python.exe experiments/eval_ray_attention_robustness_real.py \
        --data_root data/shelf_campus/Campus_Seq1 --dataset_name campus
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.shelf_loader import build_shelf_dataset
from motionflow_mv.fusion.ray_attention_v2_model import RayAttentionFusionModelV2
from motionflow_mv.fusion.triangulation import triangulate_dlt


def build_projection_matrices(K, R, t):
    Rt = np.concatenate([R, t[..., None]], axis=-1)
    return K @ Rt


def dlt_baseline(points_2d, confidences, K_orig, R_orig, t_m):
    P = build_projection_matrices(K_orig, R_orig, t_m)
    B, V, J, _ = points_2d.shape
    X = np.zeros((B, J, 3), dtype=np.float64)
    for b in range(B):
        for j in range(J):
            w = confidences[b, :, j]
            if w.sum() == 0:
                w = np.ones_like(w)
            X[b, j] = triangulate_dlt(points_2d[b, :, j], P, weights=w)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--person_id", type=int, default=0)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--dropout_rates", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    parser.add_argument("--outlier_rate", type=float, default=0.0)
    parser.add_argument("--outlier_scale", type=float, default=5000.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    points_2d, confidences, joints_3d, cameras = build_shelf_dataset(Path(args.data_root), person_id=args.person_id)
    n_frames, n_views, n_joints, _ = points_2d.shape
    joints_3d = joints_3d / 100.0
    for cam in cameras:
        cam.t = cam.t / 100.0

    K_orig = np.stack([cam.K for cam in cameras])
    R_orig = np.stack([cam.R for cam in cameras])
    t_m = np.stack([cam.t for cam in cameras])

    K = torch.from_numpy(K_orig).float().unsqueeze(0).to(device)
    R = torch.from_numpy(R_orig).float().unsqueeze(0).to(device)
    t = torch.from_numpy(t_m).float().unsqueeze(0).to(device)

    ckpt_path = Path("outputs") / f"ray_attention_{args.dataset_name}.pth"
    model = RayAttentionFusionModelV2(j=n_joints, d=args.d, n_views=n_views).to(device)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint {ckpt_path}")
    else:
        print(f"Warning: checkpoint {ckpt_path} not found, using random weights")

    model.eval()

    results = []

    for drop in args.dropout_rates:
        for noise in args.noise_levels:
            # Add noise
            p2d_noisy = points_2d.copy()
            if noise > 0:
                p2d_noisy = p2d_noisy + np.random.randn(*p2d_noisy.shape).astype(np.float64) * noise
            conf_noisy = confidences.copy()
            if drop > 0:
                # Randomly zero out whole views per joint per frame with probability drop.
                mask = np.random.rand(n_frames, n_views, n_joints) > drop
                conf_noisy = conf_noisy * mask
            if args.outlier_rate > 0:
                outlier_mask = np.random.rand(n_frames, n_views, n_joints) < args.outlier_rate
                p2d_noisy = np.where(outlier_mask[..., None], (np.random.rand(n_frames, n_views, n_joints, 2) - 0.5) * 2 * args.outlier_scale, p2d_noisy)

            # Model inference
            x_tensor = torch.from_numpy(np.concatenate([p2d_noisy, conf_noisy[..., None]], axis=-1)).float()
            dataset = torch.utils.data.TensorDataset(x_tensor, torch.from_numpy(joints_3d).float())
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            preds = []
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(device)
                    pred, _ = model(xb, K=K.expand(xb.size(0), -1, -1, -1), R=R.expand(xb.size(0), -1, -1, -1), t=t.expand(xb.size(0), -1, -1))
                    preds.append(pred.cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            model_mpjpe = np.linalg.norm(preds - joints_3d, axis=-1).mean()

            # DLT baseline
            dlt_X = dlt_baseline(p2d_noisy, conf_noisy, K_orig, R_orig, t_m)
            dlt_mpjpe = np.linalg.norm(dlt_X - joints_3d, axis=-1).mean()

            results.append({"dropout": drop, "noise": noise, "model_mpjpe_m": float(model_mpjpe), "dlt_mpjpe_m": float(dlt_mpjpe)})
            print(f"drop={drop:.1f} noise={noise:.4f} -> ray_attention={model_mpjpe:.4f}m DLT={dlt_mpjpe:.4f}m")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"robustness_{args.dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
