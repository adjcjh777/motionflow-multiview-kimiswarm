"""Combined clean + calibration-robustness evaluation for the cross-view temporal residual + principal-point model.

Usage
-----
    python experiments/eval_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
        --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_small.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --out_json outputs/crossview_pp_eval.json
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
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_model import RayAttentionFusionModelTemporalCrossviewResidual
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_campe_v2_model import RayAttentionFusionModelTemporalCrossviewResidualCamPEV2
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_adaptive_view_selection_model import RayAttentionFusionModelTemporalCrossviewResidualAdaptiveViewSelection


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
    """Apply focal/principal-point perturbation to intrinsic matrices.

    Accepts (V, 3, 3) or (B, V, 3, 3) inputs.
    """
    K_out = K.clone()
    K_out[..., 0, 0] *= 1 + focal_err
    K_out[..., 1, 1] *= 1 + focal_err
    K_out[..., 0, 2] += cxcy_err
    K_out[..., 1, 2] += cxcy_err
    return K_out


def corrupt_extrinsics(R, t, rot_std_deg, trans_std):
    """Apply rotation/translation perturbation to extrinsics.

    Accepts (V, 3, 3)/(V, 3) or (B, V, 3, 3)/(B, V, 3) inputs.
    """
    R_out = R.clone()
    t_out = t.clone()
    if rot_std_deg > 0:
        axis = torch.randn(*R.shape[:-1], 3, device=R.device, dtype=R.dtype)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        theta = torch.randn(*R.shape[:-1], 1, device=R.device, dtype=R.dtype) * np.deg2rad(rot_std_deg)
        delta_R = so3_exp((axis * theta).squeeze(-1))
        R_out = torch.einsum("...ij,...jk->...ik", delta_R, R)
    if trans_std > 0:
        t_out = t + torch.randn_like(t) * trans_std
    return R_out, t_out


def build_model(args, n_views, j):
    common = {"j": j, "d": args.d, "n_views": n_views, "n_st_layers": args.n_st_layers, "residual_hidden": args.residual_hidden}
    if args.model_class == "crossview_residual_principal_point":
        return RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
            **common,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
        )
    elif args.model_class == "crossview_residual":
        return RayAttentionFusionModelTemporalCrossviewResidual(**common)
    elif args.model_class == "crossview_residual_campe_v2":
        return RayAttentionFusionModelTemporalCrossviewResidualCamPEV2(**common)
    elif args.model_class == "crossview_residual_adaptive":
        return RayAttentionFusionModelTemporalCrossviewResidualAdaptiveViewSelection(**common, selector_k=min(args.selector_k, n_views))
    else:
        raise ValueError(f"Unknown model_class: {args.model_class}")


def evaluate_clean(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, *_ = model(xb, K=K, R=R, t=t)
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    return compute_all_metrics(preds, gts)


def evaluate_perturbed(model, dataset, cfg, batch_size, device):
    class PerturbedDataset(torch.utils.data.Dataset):
        def __init__(self, base, cfg):
            self.base = base
            self.cfg = cfg

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            x, y, K, R, t = self.base[idx]
            K = corrupt_intrinsics(K, self.cfg["focal_err"], self.cfg["cxcy_err"])
            R, t = corrupt_extrinsics(R, t, self.cfg["rot_std"], self.cfg["trans_std"])
            return x, y, K, R, t

    loader = torch.utils.data.DataLoader(
        PerturbedDataset(dataset, cfg),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    return evaluate_clean(model, loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=1, help="Stride for validation clips")
    parser.add_argument("--out_json", type=str, default="outputs/crossview_principal_point_model_eval.json")
    parser.add_argument("--focal_max_scale", type=float, default=0.0, help="Maximum predicted focal-length scale; 0 disables focal correction")
    parser.add_argument("--model_class", type=str, default="crossview_residual_principal_point",
                        choices=["crossview_residual_principal_point", "crossview_residual", "crossview_residual_campe_v2", "crossview_residual_adaptive"],
                        help="Model architecture to evaluate")
    parser.add_argument("--selector_k", type=int, default=4, help="Number of views to select for adaptive model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    model = build_model(args, n_views, j).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    dataset = TemporalClipDataset(args.dataset, args.clip_len, stride=args.val_stride)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    print("Clean evaluation...")
    clean_report = evaluate_clean(model, loader, device)
    clean_summary = {k: float(v) for k, v in clean_report.items() if not k.endswith("_per_joint") and not isinstance(v, np.ndarray)}
    print(f"Clean: MPJPE={clean_summary['mpjpe']:.2f}mm PA={clean_summary['pa_mpjpe']:.2f}mm")

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

    robustness = {"clean": clean_summary}
    print("Calibration robustness evaluation...")
    for name, cfg in conditions.items():
        if name == "clean":
            continue
        report = evaluate_perturbed(model, dataset, cfg, args.batch_size, device)
        robustness[name] = {k: float(v) for k, v in report.items() if not k.endswith("_per_joint") and not isinstance(v, np.ndarray)}
        print(f"{name}: MPJPE={robustness[name]['mpjpe']:.2f}mm PA={robustness[name]['pa_mpjpe']:.2f}mm")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"clean": clean_summary, "robustness": robustness}, f, indent=2)
    print(f"Saved summary to {out_path}")

    print("\nMarkdown table (MPJPE / PA-MPJPE):")
    print("| Condition | MPJPE (mm) | PA-MPJPE (mm) |")
    print("|---|---|---|")
    for name, cfg in conditions.items():
        r = robustness.get(name, clean_summary)
        print(f"| {name} | {r['mpjpe']:.2f} | {r['pa_mpjpe']:.2f} |")


if __name__ == "__main__":
    main()
