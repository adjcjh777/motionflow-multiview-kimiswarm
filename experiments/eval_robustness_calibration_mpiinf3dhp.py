"""Robustness of the residual temporal model to camera calibration errors."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual


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
        self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat([self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)], dim=-1)
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def so3_exp(axis_angle):
    """Convert an axis-angle vector (3,) to a rotation matrix."""
    theta = axis_angle.norm()
    if theta < 1e-6:
        return torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    k = axis_angle / theta
    K = torch.zeros(3, 3, dtype=axis_angle.dtype, device=axis_angle.device)
    K[0, 1] = -k[2]
    K[0, 2] = k[1]
    K[1, 2] = -k[0]
    K[1, 0] = k[2]
    K[2, 0] = -k[1]
    K[2, 1] = k[0]
    return torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


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


def corrupt_intrinsics(K, focal_err, cxcy_err):
    K_out = K.clone()
    K_out[:, 0, 0] *= (1 + focal_err)
    K_out[:, 1, 1] *= (1 + focal_err)
    K_out[:, 0, 2] += cxcy_err
    K_out[:, 1, 2] += cxcy_err
    return K_out


def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    report = compute_all_metrics(preds, gts)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_clips", type=int, default=None, help="If set, evaluate on at most this many clips per condition for speed.")
    parser.add_argument("--out_json", type=str, default="outputs/robustness_calibration_mpiinf3dhp.json")
    parser.add_argument("--use_reproj_gate", action="store_true", help="Use reprojection-error gate in the residual head")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    dataset = TemporalClipDataset(args.dataset, args.clip_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = RayAttentionFusionModelTemporalResidual(
        j=j, d=args.d, n_views=n_views, n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden, use_reproj_gate=args.use_reproj_gate,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    conditions = {
        "clean": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "rot_0.5_deg": {"rot_std": 0.5, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "rot_1.0_deg": {"rot_std": 1.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "trans_5mm": {"rot_std": 0.0, "trans_std": 0.005, "focal_err": 0.0, "cxcy_err": 0.0},
        "trans_10mm": {"rot_std": 0.0, "trans_std": 0.010, "focal_err": 0.0, "cxcy_err": 0.0},
        "focal_1pct": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.01, "cxcy_err": 0.0},
        "focal_2pct": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.02, "cxcy_err": 0.0},
        "cxcy_3px": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 3.0},
        "cxcy_5px": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 5.0},
    }

    results = {}
    # To keep the original loader's K/R/t, we monkey-patch the collated K/R/t in evaluate by creating a custom loader.
    # Instead, we re-evaluate directly in the loop below by wrapping the dataset.
    for name, cfg in conditions.items():
        class PerturbedDataset(torch.utils.data.Dataset):
            def __init__(self, base, cfg):
                self.base = base
                self.cfg = cfg
            def __len__(self):
                return len(self.base)
            def __getitem__(self, idx):
                x, y, K, R, t = self.base[idx]
                K = corrupt_intrinsics(K.unsqueeze(0), self.cfg["focal_err"], self.cfg["cxcy_err"]).squeeze(0)
                R, t = corrupt_extrinsics(R, t, self.cfg["rot_std"], self.cfg["trans_std"])
                return x, y, K, R, t

        perturbed = PerturbedDataset(dataset, cfg)
        if args.max_clips is not None and args.max_clips < len(perturbed):
            perturbed = torch.utils.data.Subset(perturbed, range(args.max_clips))
        loader_pert = torch.utils.data.DataLoader(perturbed, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        report = evaluate(model, loader_pert, device)
        results[name] = {k: float(v) for k, v in report.items() if not k.endswith("_per_joint") and not isinstance(v, np.ndarray)}
        print(f"{name}: MPJPE={results[name]['mpjpe']:.2f}mm PA={results[name]['pa_mpjpe']:.2f}mm")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
