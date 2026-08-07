"""Unified 6-axis robustness matrix for the registered model family.

Covers: rotation, translation, focal length, principal point, radial distortion,
and occlusion (view/joint dropout).

The matrix now works with any model in ``MODEL_CLASSES`` from
``experiments/eval_full_metrics.py``.  The legacy principal-point baseline is
still the default.

Usage
-----
    # PP baseline (legacy default)
    python experiments/eval_robustness_matrix_pp_mpiinf3dhp.py \
        --model crossview_residual_pp \
        --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_full_ppw005_20ep.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --out_json outputs/robustness_matrix_pp_full.json \
        --out_md docs/tables/icra2027/robustness_matrix.md

    # Hierarchical attention variant
    python experiments/eval_robustness_matrix_pp_mpiinf3dhp.py \
        --model hierarchical_view_temporal_joint_pp \
        --checkpoint outputs/hierarchical_attention_pp_full_mpiinf3dhp.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --n_view_groups 2 --n_view_layers 2 --n_temporal_layers 2 --n_joint_graph_layers 1 \
        --out_json outputs/robustness_matrix_hierarchical.json \
        --out_md docs/tables/icra2027/robustness_matrix_hierarchical.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_full_metrics import build_model, MODEL_CLASSES
from motionflow_mv.calibration.perturb import perturb_radial_distortion
from motionflow_mv.data.occlusion_aug import random_occlude_views, random_occlude_joints
from motionflow_mv.eval.metrics import compute_all_metrics


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


def evaluate_condition(model, loader, cfg, device):
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)

            # Calibration perturbations
            K = corrupt_intrinsics(K, cfg["focal_err"], cfg["cxcy_err"])
            R, t = corrupt_extrinsics(R, t, cfg["rot_std"], cfg["trans_std"])

            # Radial distortion on 2D keypoints
            if cfg.get("distortion_k1", 0.0) != 0.0:
                # xb shape: (B, T, V, J, 3) where last dim is (x, y, conf)
                B, T, V, J = xb.shape[:4]
                points = xb[..., :2]  # (B, T, V, J, 2)
                points_flat = points.reshape(B * T, V, J, 2)
                # All clips in the batch share the same camera rig.
                K_base = K[0] if K.dim() == 4 else K
                K_flat = K_base.unsqueeze(0).expand(B * T, -1, -1, -1)
                points_flat = perturb_radial_distortion(points_flat, K_flat, k1_std=cfg["distortion_k1"])
                points = points_flat.reshape(B, T, V, J, 2)
                xb = torch.cat([points, xb[..., 2:]], dim=-1)

            # Occlusion (per_sample=False to match the loader's stacked shape and
            # avoid a dimension-conflict in the occlusion helper).
            if cfg.get("occlude_views", 0.0) > 0.0:
                xb = random_occlude_views(xb, rate=cfg["occlude_views"], per_sample=False)
            if cfg.get("occlude_joints", 0.0) > 0.0:
                xb = random_occlude_joints(xb, rate=cfg["occlude_joints"], per_view=True, per_sample=False)

            out = model(xb, K=K, R=R, t=t)
            pred = out[0] if isinstance(out, tuple) else out
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    return compute_all_metrics(preds, gts)


def main():
    parser = argparse.ArgumentParser(description="Unified 6-axis robustness matrix for the registered model family")
    parser.add_argument("--model", type=str, choices=list(MODEL_CLASSES), default="crossview_residual_pp")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="outputs/robustness_matrix_pp_full.json")
    parser.add_argument("--out_md", type=str, default="docs/tables/icra2027/robustness_matrix.md")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--n_view_layers", type=int, default=2)
    parser.add_argument("--n_view_groups", type=int, default=2)
    parser.add_argument("--n_joint_graph_layers", type=int, default=1)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--target_k", type=int, default=4)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--focal_max_scale", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    if args.model == "crossview_residual_pp":
        # Preserve principal-point specific overrides for the legacy default.
        from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
            RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
        )

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
    else:
        model = build_model(args, n_views, j).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys in checkpoint: {missing[:5]}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint (ignored): {unexpected[:5]}")
    model.eval()

    dataset = TemporalClipDataset(args.dataset, args.clip_len, stride=args.val_stride)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    conditions = [
        {"name": "clean", "rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0, "distortion_k1": 0.0, "occlude_views": 0.0, "occlude_joints": 0.0},
        {"name": "rot_0.5_deg", "rot_std": 0.5},
        {"name": "rot_1.0_deg", "rot_std": 1.0},
        {"name": "trans_5mm", "trans_std": 0.005},
        {"name": "trans_10mm", "trans_std": 0.010},
        {"name": "focal_1pct", "focal_err": 0.01},
        {"name": "focal_2pct", "focal_err": 0.02},
        {"name": "cxcy_3px", "cxcy_err": 3.0},
        {"name": "cxcy_5px", "cxcy_err": 5.0},
        {"name": "distortion_k1_0.01", "distortion_k1": 0.01},
        {"name": "distortion_k1_0.05", "distortion_k1": 0.05},
        {"name": "distortion_k1_0.10", "distortion_k1": 0.10},
        {"name": "view_dropout_0.2", "occlude_views": 0.2},
        {"name": "view_dropout_0.4", "occlude_views": 0.4},
        {"name": "joint_dropout_0.2", "occlude_joints": 0.2},
        {"name": "joint_dropout_0.4", "occlude_joints": 0.4},
    ]
    # Fill defaults
    base = {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0, "distortion_k1": 0.0, "occlude_views": 0.0, "occlude_joints": 0.0}
    for c in conditions:
        for k, v in base.items():
            c.setdefault(k, v)

    results = {}
    print(f"{'Condition':<25} {'MPJPE':>10} {'PA-MPJPE':>10} {'PCK@50':>8} {'PCK@100':>8} {'PCK@150':>8} {'AUC':>8}")
    print("-" * 90)
    for c in conditions:
        report = evaluate_condition(model, loader, c, device)
        summary = {k: float(v) for k, v in report.items() if not k.endswith("_per_joint") and not isinstance(v, np.ndarray)}
        results[c["name"]] = summary
        print(f"{c['name']:<25} {summary['mpjpe']:>10.2f} {summary['pa_mpjpe']:>10.2f} {summary['pck@50mm']:>8.3f} {summary['pck@100mm']:>8.3f} {summary['pck@150mm']:>8.3f} {summary['pck_auc']:>8.3f}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved JSON to {out_path}")

    md_lines = [
        "| Condition | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        md_lines.append(
            f"| {name} | {r['mpjpe']:.2f} | {r['pa_mpjpe']:.2f} | {r['pck@50mm']:.3f} | "
            f"{r['pck@100mm']:.3f} | {r['pck@150mm']:.3f} | {r['pck_auc']:.3f} |"
        )
    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines))
    print(f"Saved Markdown table to {md_path}")


if __name__ == "__main__":
    main()
