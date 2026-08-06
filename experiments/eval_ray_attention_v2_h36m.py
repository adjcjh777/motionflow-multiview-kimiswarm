"""Evaluate ray_attention_v2 on H36M multi-view data under noise/dropout/outliers.

Example:
    /d/anaconda3/envs/jz_py310/python.exe experiments/eval_ray_attention_v2_h36m.py \
        --dataset data/h36m_hf/s_01_act_02_multiview.npz \
        --checkpoint outputs/ray_attention_v2_h36m_s1a2.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_v2_model import RayAttentionFusionModelV2
from motionflow_mv.fusion.triangulation import triangulate_dlt_torch


def dlt_baseline(points_2d, confidences, K, R, t):
    P = K @ torch.cat([R, t[..., None]], dim=-1)
    B, V, J, _ = points_2d.shape
    X = torch.zeros((B, J, 3), dtype=points_2d.dtype, device=points_2d.device)
    for b in range(B):
        for j in range(J):
            w = confidences[b, :, j]
            if w.sum() == 0:
                w = torch.ones_like(w)
            X[b, j] = triangulate_dlt_torch(points_2d[b, :, j], P, weights=w)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/h36m_hf/s_01_act_02_multiview.npz")
    parser.add_argument("--checkpoint", type=str, default="outputs/ray_attention_v2_h36m_s1a2.pth")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 2.0, 5.0])
    parser.add_argument("--dropout_rates", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    parser.add_argument("--outlier_rate", type=float, default=0.0)
    parser.add_argument("--outlier_scale", type=float, default=50.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.dataset)
    points_2d = data["points_2d"]
    confidences = data["confidences"]
    joints_3d = data["joints_3d"]
    K = torch.from_numpy(data["camera_K"]).float().unsqueeze(0).to(device)
    R = torch.from_numpy(data["camera_R"]).float().unsqueeze(0).to(device)
    t = torch.from_numpy(data["camera_t"]).float().unsqueeze(0).to(device)

    n_views = data["camera_K"].shape[0]
    n_joints = data["points_2d"].shape[2]
    model = RayAttentionFusionModelV2(j=n_joints, d=args.d, n_views=n_views).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    for drop in args.dropout_rates:
        for noise in args.noise_levels:
            p2d = points_2d.copy()
            conf = confidences.copy()
            if noise > 0:
                p2d = p2d + np.random.randn(*p2d.shape).astype(np.float64) * noise
            if drop > 0:
                mask = np.random.rand(p2d.shape[0], n_views, n_joints) > drop
                conf = conf * mask
            if args.outlier_rate > 0:
                out_mask = np.random.rand(p2d.shape[0], n_views, n_joints) < args.outlier_rate
                outlier = (np.random.rand(p2d.shape[0], n_views, n_joints, 2) - 0.5) * 2 * args.outlier_scale
                p2d = np.where(out_mask[..., None], outlier, p2d)

            x = torch.from_numpy(np.concatenate([p2d, conf[..., None]], axis=-1)).float()
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x, torch.from_numpy(joints_3d).float()),
                batch_size=args.batch_size,
                shuffle=False,
            )

            preds = []
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(device)
                    pred, _ = model(xb, K=K.expand(xb.size(0), -1, -1, -1), R=R.expand(xb.size(0), -1, -1, -1), t=t.expand(xb.size(0), -1, -1))
                    preds.append(pred.cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            model_err = np.linalg.norm(preds - joints_3d, axis=-1).mean()

            dlt_X = dlt_baseline(torch.from_numpy(p2d).float().to(device), torch.from_numpy(conf).float().to(device), K, R, t)
            dlt_err = np.linalg.norm(dlt_X.cpu().numpy() - joints_3d, axis=-1).mean()
            print(f"drop={drop:.1f} noise={noise:.2f} -> ray_attention={model_err:.4f}  DLT={dlt_err:.4f}")


if __name__ == "__main__":
    main()
