"""Calibration-robustness smoke evaluation for the cross-view PP curriculum checkpoint.

Evaluates the ``ray_attention_temporal_crossview_residual_principal_point_curriculum_v1``
checkpoint under the four calibration perturbations from the P0 robustness direction:
rot_0.5deg, trans_5mm, focal_1pct and pp_10px. Runs entirely on CPU and uses the
small smoke dataset so it can be used as a quick regression check.

Usage
-----
    python experiments/eval_curriculum_robustness.py \
        --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
        --out_json outputs/eval_curriculum_robustness_smoke.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint


class TemporalClipDataset(torch.utils.data.Dataset):
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
        self.num_clips = max(1, (self.total_frames - self.clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
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


def so3_exp(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = axis_angle.norm(dim=-1, keepdim=True)[..., None]
    k = axis_angle / (axis_angle.norm(dim=-1, keepdim=True) + 1e-8)
    K = torch.zeros(*axis_angle.shape[:-1], 3, 3, dtype=axis_angle.dtype, device=axis_angle.device)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]
    I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    I = I.view(*([1] * (axis_angle.ndim - 1)), 3, 3).expand(*axis_angle.shape[:-1], 3, 3)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def corrupt_intrinsics(K, focal_err, cxcy_err):
    """Apply focal/principal-point perturbation to a (V, 3, 3) intrinsic matrix."""
    K_out = K.clone()
    K_out[:, 0, 0] *= 1 + focal_err
    K_out[:, 1, 1] *= 1 + focal_err
    K_out[:, 0, 2] += cxcy_err
    K_out[:, 1, 2] += cxcy_err
    return K_out


def corrupt_extrinsics(R, t, rot_std_deg, trans_std):
    V = R.shape[0]
    R_out = R.clone()
    t_out = t.clone()
    for v in range(V):
        noise = torch.randn(3, device=R.device) * np.deg2rad(rot_std_deg)
        delta_R = so3_exp(noise)
        R_out[v] = delta_R @ R[v]
        t_out[v] = t[v] + torch.randn(3, device=t.device) * trans_std
    return R_out, t_out


def evaluate_perturbed(model, loader, cfg, device):
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            K = corrupt_intrinsics(K, cfg["focal_err"], cfg["cxcy_err"])
            R, t = corrupt_extrinsics(R, t, cfg["rot_std"], cfg["trans_std"])
            pred, *_ = model(xb, K=K, R=R, t=t)
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    return compute_all_metrics(preds, gts)


def main():
    parser = argparse.ArgumentParser(
        description="CPU smoke evaluation of the cross-view PP curriculum checkpoint under calibration perturbations."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth",
        help="Path to the cross-view residual+PP curriculum checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz",
        help="Path to the smoke .npz dataset.",
    )
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--focal_max_scale", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_stride", type=int, default=50)
    parser.add_argument("--out_json", type=str, default="outputs/eval_curriculum_robustness_smoke.json")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on. Defaults to CPU for smoke testing.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=j,
        d=args.d,
        n_views=n_views,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=args.principal_point_max_offset,
        focal_max_scale=args.focal_max_scale,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    dataset = TemporalClipDataset(args.dataset, args.clip_len, stride=args.val_stride)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    conditions = {
        "clean": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "rot_0.5": {"rot_std": 0.5, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "trans_5": {"rot_std": 0.0, "trans_std": 0.005, "focal_err": 0.0, "cxcy_err": 0.0},
        "focal_1%": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.01, "cxcy_err": 0.0},
        "pp_10px": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 10.0},
    }

    results = {}
    print("Condition           | MPJPE (mm) | PA-MPJPE (mm)")
    print("-" * 45)
    for name, cfg in conditions.items():
        report = evaluate_perturbed(model, loader, cfg, device)
        results[name] = {
            k: float(v)
            for k, v in report.items()
            if not k.endswith("_per_joint") and not isinstance(v, np.ndarray)
        }
        print(f"{name:<18} | {results[name]['mpjpe']:>10.2f} | {results[name]['pa_mpjpe']:>13.2f}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
