"""Extended robustness matrix v2 for the temporal-residual ray-attention model.

Extends ``experiments/eval_perturb_model_mpiinf3dhp.py`` with:
* batch-level calibration perturbations (intrinsics + extrinsics)
* occlusion / view-dropout / joint-dropout sweeps
* richer metrics table (PCK + AUC) and Markdown output
* built-in CPU smoke mode when no checkpoint/dataset are provided

Usage (real evaluation)
-----------------------
    python experiments/prototypes/swarm_iter18/eval_robustness_matrix_v2.py \
        --checkpoint outputs/ray_attention_temporal_residual_perturb_mpiinf3dhp.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --out_json outputs/robustness_matrix_v2.json \
        --out_md docs/swarm_iter18/robustness_matrix_v2.md

Usage (CPU smoke test)
----------------------
    python experiments/prototypes/swarm_iter18/eval_robustness_matrix_v2.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from motionflow_mv.data.occlusion_aug import random_occlude_joints, random_occlude_views
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


class SyntheticTemporalClipDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset for the built-in CPU smoke test."""

    def __init__(
        self,
        points_2d: torch.Tensor,
        confidences: torch.Tensor,
        joints_3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        clip_len: int,
        stride: int = 1,
    ):
        self.points_2d = points_2d
        self.confidences = confidences
        self.joints_3d = joints_3d
        self.K = K
        self.R = R
        self.t = t
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
    """Corrupt focal length and principal point.

    Supports both single rigs ``(V, 3, 3)`` and batched rigs ``(B, V, 3, 3)``.
    """
    K_out = K.clone()
    K_out[..., 0, 0] *= 1 + focal_err
    K_out[..., 1, 1] *= 1 + focal_err
    K_out[..., 0, 2] += cxcy_err
    K_out[..., 1, 2] += cxcy_err
    return K_out


def corrupt_extrinsics(R, t, rot_std_deg, trans_std):
    """Corrupt camera rotation and translation.

    Supports both single rigs ``(V, 3, 3) / (V, 3)`` and batched rigs
    ``(B, V, 3, 3) / (B, V, 3)``.
    """
    if rot_std_deg == 0.0 and trans_std == 0.0:
        return R, t
    R_out = R.clone()
    t_out = t.clone()
    V = R.shape[-3]
    for v in range(V):
        noise = torch.randn(3, device=R.device) * np.deg2rad(rot_std_deg)
        delta_R = so3_exp(noise)
        R_out[..., v, :, :] = delta_R @ R[..., v, :, :]
        t_out[..., v, :] = t[..., v, :] + torch.randn(3, device=t.device) * trans_std
    return R_out, t_out


def apply_occlusion(x, occlude_views_rate=0.0, occlude_joints_rate=0.0):
    """Apply random view/joint occlusion to a tensor of shape ``(..., V, J, C)``.

    The confidence channel (last channel) is zeroed for occluded entries.
    """
    if occlude_views_rate > 0.0:
        x = random_occlude_views(x, rate=occlude_views_rate, per_sample=False)
    if occlude_joints_rate > 0.0:
        x = random_occlude_joints(x, rate=occlude_joints_rate, per_view=True, per_sample=False)
    return x


def build_model(args, n_views, j):
    return RayAttentionFusionModelTemporalResidual(
        j=j,
        d=args.d,
        n_views=n_views,
        n_heads=args.n_heads,
        n_joint_layers=args.n_joint_layers,
        n_temporal_layers=args.n_temporal_layers,
        max_temporal_len=args.max_temporal_len,
        residual_hidden=args.residual_hidden,
        use_reproj_gate=args.use_reproj_gate,
    )


def evaluate_condition(model, loader, cfg, device):
    """Evaluate one robustness condition (calibration + occlusion)."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)

            K = corrupt_intrinsics(K, cfg["focal_err"], cfg["cxcy_err"])
            R, t = corrupt_extrinsics(R, t, cfg["rot_std"], cfg["trans_std"])
            xb = apply_occlusion(
                xb,
                occlude_views_rate=cfg.get("occlude_views", 0.0),
                occlude_joints_rate=cfg.get("occlude_joints", 0.0),
            )

            pred, _ = model(xb, K=K, R=R, t=t)
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    return compute_all_metrics(preds, gts)


def summarize(report):
    return {k: float(v) for k, v in report.items() if not k.endswith("_per_joint") and not isinstance(v, np.ndarray)}


def smoke_test():
    """Lightweight CPU smoke test with random weights and synthetic data."""
    print("=" * 60)
    print("Robustness matrix v2 - CPU smoke test")
    print("=" * 60)

    torch.manual_seed(0)
    np.random.seed(0)

    T, V, J = 32, 4, 17
    points_2d = torch.rand(T, V, J, 2) * 640.0
    confidences = torch.ones(T, V, J)
    joints_3d = torch.rand(T, J, 3)

    K = torch.eye(3).unsqueeze(0).repeat(V, 1, 1).float()
    K[:, 0, 0] = K[:, 1, 1] = 800.0
    K[:, 0, 2] = 320.0
    K[:, 1, 2] = 240.0
    R = torch.eye(3).unsqueeze(0).repeat(V, 1, 1).float()
    t = torch.zeros(V, 3).float()

    dataset = SyntheticTemporalClipDataset(
        points_2d, confidences, joints_3d, K, R, t, clip_len=9, stride=4
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    model = RayAttentionFusionModelTemporalResidual(
        j=J,
        d=16,
        n_views=V,
        n_heads=2,
        n_joint_layers=1,
        n_temporal_layers=1,
        max_temporal_len=64,
        residual_hidden=32,
    )
    model.eval()
    device = torch.device("cpu")

    base = {
        "rot_std": 0.0,
        "trans_std": 0.0,
        "focal_err": 0.0,
        "cxcy_err": 0.0,
        "occlude_views": 0.0,
        "occlude_joints": 0.0,
    }
    conditions = [
        {"name": "clean"},
        {"name": "rot_0.5_deg", "rot_std": 0.5},
        {"name": "focal_2pct", "focal_err": 0.02},
        {"name": "view_dropout_0.4", "occlude_views": 0.4},
        {"name": "joint_dropout_0.4", "occlude_joints": 0.4},
        {"name": "view_joint_dropout_0.2", "occlude_views": 0.2, "occlude_joints": 0.2},
    ]
    for c in conditions:
        for k, v in base.items():
            c.setdefault(k, v)

    results = {}
    print(f"{'Condition':<25} {'MPJPE':>10} {'PA-MPJPE':>10} {'PCK@50':>8} {'AUC':>8}")
    print("-" * 70)
    for c in conditions:
        report = evaluate_condition(model, loader, c, device)
        summary = summarize(report)
        results[c["name"]] = summary
        print(
            f"{c['name']:<25} {summary['mpjpe']:>10.2f} {summary['pa_mpjpe']:>10.2f} "
            f"{summary['pck@50mm']:>8.3f} {summary['pck_auc']:>8.3f}"
        )

    assert all("mpjpe" in r for r in results.values())
    assert all(not np.isnan(r["mpjpe"]) for r in results.values())
    print("\nSmoke test passed.")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extended robustness matrix v2 for temporal-residual ray-attention model"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--out_json", type=str, default="outputs/robustness_matrix_v2.json")
    parser.add_argument("--out_md", type=str, default="docs/swarm_iter18/robustness_matrix_v2.md")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_joint_layers", type=int, default=1)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--max_temporal_len", type=int, default=256)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=1, help="Stride for validation clips")
    parser.add_argument("--use_reproj_gate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.checkpoint is None or args.dataset is None:
        smoke_test()
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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

    base = {
        "rot_std": 0.0,
        "trans_std": 0.0,
        "focal_err": 0.0,
        "cxcy_err": 0.0,
        "occlude_views": 0.0,
        "occlude_joints": 0.0,
    }
    conditions = [
        {"name": "clean"},
        {"name": "rot_0.5_deg", "rot_std": 0.5},
        {"name": "rot_1.0_deg", "rot_std": 1.0},
        {"name": "trans_5mm", "trans_std": 0.005},
        {"name": "trans_10mm", "trans_std": 0.010},
        {"name": "focal_1pct", "focal_err": 0.01},
        {"name": "focal_2pct", "focal_err": 0.02},
        {"name": "cxcy_3px", "cxcy_err": 3.0},
        {"name": "cxcy_5px", "cxcy_err": 5.0},
        {"name": "view_dropout_0.2", "occlude_views": 0.2},
        {"name": "view_dropout_0.4", "occlude_views": 0.4},
        {"name": "view_dropout_0.6", "occlude_views": 0.6},
        {"name": "joint_dropout_0.2", "occlude_joints": 0.2},
        {"name": "joint_dropout_0.4", "occlude_joints": 0.4},
        {"name": "joint_dropout_0.6", "occlude_joints": 0.6},
        {"name": "view_joint_dropout_0.2", "occlude_views": 0.2, "occlude_joints": 0.2},
    ]
    for c in conditions:
        for k, v in base.items():
            c.setdefault(k, v)

    results = {}
    print(f"{'Condition':<25} {'MPJPE':>10} {'PA-MPJPE':>10} {'PCK@50':>8} {'AUC':>8}")
    print("-" * 70)
    for c in conditions:
        report = evaluate_condition(model, loader, c, device)
        summary = summarize(report)
        results[c["name"]] = summary
        print(
            f"{c['name']:<25} {summary['mpjpe']:>10.2f} {summary['pa_mpjpe']:>10.2f} "
            f"{summary['pck@50mm']:>8.3f} {summary['pck_auc']:>8.3f}"
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved JSON to {out_path}")

    md_lines = [
        "| Condition | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
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
