"""Evaluation script for OmniMultiViewFusionV3 on MPI-INF-3DHP S2/Seq1.

Mirrors the v2 evaluation driver but instantiates OmniMultiViewFusionV3 and
supports the v3 ablation flags (multiscale fusion, camera conditioning,
epipolar bias).

Usage
-----
    python experiments/eval_omniview_fusion_v3_mpiinf3dhp.py \
        --checkpoint outputs/omniview_fusion_v3_d128_dense_graph_v2_a800_no_warm.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --use_multiscale_fusion --use_camera_conditioning --use_epipolar_bias \
        --run_robustness --run_variable_views

    # CPU/GPU smoke test
    python experiments/eval_omniview_fusion_v3_mpiinf3dhp.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.omniview_fusion_v3 import OmniMultiViewFusionV3
from motionflow_mv.training.trainer_v2 import checkpoint_eval_state_dict
from motionflow_mv.fusion.variable_view_inference import (
    VariableViewInferenceWrapper,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


# ---------------------------------------------------------------------------
# Synthetic smoke dataset
# ---------------------------------------------------------------------------

def _make_synthetic_cameras(n_views: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K_list, R_list, t_list = [], [], []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        K_list.append(K)
        R_list.append(R)
        t_list.append(t)
    return (
        torch.from_numpy(np.stack(K_list, axis=0)).float(),
        torch.from_numpy(np.stack(R_list, axis=0)).float(),
        torch.from_numpy(np.stack(t_list, axis=0)).float(),
    )


class SyntheticSmokeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        n_frames: int = 60,
        n_joints: int = 17,
        clip_len: int = 9,
        noise_std: float = 0.5,
    ):
        self.K = K
        self.R = R
        self.t = t
        self.n_joints = n_joints
        self.clip_len = clip_len
        self.noise_std = noise_std

        torch.manual_seed(42)
        joints_3d = torch.randn(n_frames, n_joints, 3) * 0.3
        for _ in range(2):
            joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

        points_2d = self._project_points(joints_3d, K, R, t)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        self.points_2d = points_2d
        self.confidences = torch.ones_like(points_2d[..., 0])
        self.joints_3d = joints_3d
        self.total_frames = n_frames
        self.num_clips = max(1, self.total_frames - self.clip_len + 1)

    @staticmethod
    def _project_points(
        joints_3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        X = joints_3d
        t = t[:, None, None, :]
        X_cam = torch.einsum("vab,fjb->vfja", R, X) + t
        z = X_cam[..., 2:3].clamp(min=1e-6)
        uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])
        points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
        return points_2d.permute(1, 0, 2, 3)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx: int):
        start = idx
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


# ---------------------------------------------------------------------------
# Model construction / loading
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace, n_views: int, j: int) -> OmniMultiViewFusionV3:
    model = OmniMultiViewFusionV3(
        j=j,
        d=args.d,
        n_views=n_views,
        n_heads=args.n_heads,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        graph_num_layers=args.graph_num_layers,
        n_joint_layers=args.n_joint_layers,
        return_pp_delta=False,
        return_covariance=True,
        use_multiscale_fusion=args.use_multiscale_fusion,
        use_camera_conditioning=args.use_camera_conditioning,
        use_epipolar_bias=args.use_epipolar_bias,
    )
    if j != 17:
        model.rebuild_graph(j, dataset="mpiinf3dhp")
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint_eval_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Checkpoint load: missing keys {missing[:10]}")
    if unexpected:
        print(f"Checkpoint load: unexpected keys ignored {unexpected[:10]}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_clean(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            out = model(xb, K=K, R=R, t=t)
            pred = out[0]
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    return compute_all_metrics(preds, gts)


# ---------------------------------------------------------------------------
# Robustness matrix
# ---------------------------------------------------------------------------

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


def corrupt_intrinsics(K: torch.Tensor, focal_err: float, cxcy_err: float) -> torch.Tensor:
    K_out = K.clone()
    K_out[..., 0, 0] *= 1 + focal_err
    K_out[..., 1, 1] *= 1 + focal_err
    K_out[..., 0, 2] += cxcy_err
    K_out[..., 1, 2] += cxcy_err
    return K_out


def corrupt_extrinsics(R: torch.Tensor, t: torch.Tensor, rot_std_deg: float, trans_std: float) -> Tuple[torch.Tensor, torch.Tensor]:
    R_out = R.clone()
    t_out = t.clone()
    if rot_std_deg > 0:
        noise = torch.randn(R.shape[0], 3, device=R.device, dtype=R.dtype) * np.deg2rad(rot_std_deg)
        delta_R = so3_exp(noise)
        R_out = torch.einsum("vij,vjk->vik", delta_R, R)
    if trans_std > 0:
        t_out = t + torch.randn_like(t) * trans_std
    return R_out, t_out


def evaluate_perturbed(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    cfg: Dict[str, float],
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    class PerturbedDataset(torch.utils.data.Dataset):
        def __init__(self, base: torch.utils.data.Dataset, cfg: Dict[str, float]):
            self.base = base
            self.cfg = cfg

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx: int):
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


def robustness_conditions(smoke: bool = False) -> Dict[str, Dict[str, float]]:
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
    if smoke:
        conditions = {
            "clean": conditions["clean"],
            "rot_0.5_deg": conditions["rot_0.5_deg"],
            "focal_1pct": conditions["focal_1pct"],
            "cxcy_3px": conditions["cxcy_3px"],
        }
    return conditions


# ---------------------------------------------------------------------------
# Variable-view MPJPE@k
# ---------------------------------------------------------------------------

def evaluate_variable_views(
    model: torch.nn.Module,
    points_2d: np.ndarray,
    confidences: np.ndarray,
    joints_3d: np.ndarray,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    clip_len: int,
    device: torch.device,
    min_views: int = 2,
    max_views: int | None = None,
    num_subsets_per_k: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    V = points_2d.shape[1]
    J = points_2d.shape[2]
    T = points_2d.shape[0]
    max_views = min(max_views or V, V)

    wrapper = VariableViewInferenceWrapper(model)
    rng = np.random.default_rng(seed)

    all_clips = []
    for start in range(0, T - clip_len + 1, clip_len):
        end = start + clip_len
        x_clip = torch.from_numpy(
            np.concatenate([points_2d[start:end], confidences[start:end, ..., None]], axis=-1)
        ).float()
        all_clips.append((start, end, x_clip))

    results: Dict[str, Any] = {}
    for k in range(min_views, max_views + 1):
        subsets = list(combinations(range(V), k))
        if len(subsets) > num_subsets_per_k:
            idx = rng.choice(len(subsets), size=num_subsets_per_k, replace=False)
            subsets = [subsets[i] for i in idx]

        errors = []
        for subset in subsets:
            preds = []
            gt_list = []
            for start, end, x_clip in all_clips:
                x_clip = x_clip.to(device)
                Kp = K.to(device)
                Rp = R.to(device)
                tp = t.to(device)
                with torch.no_grad():
                    pred = wrapper(x_clip, Kp, Rp, tp, active_views=list(subset))[0]
                preds.append(pred.cpu().numpy())
                gt_list.append(joints_3d[start:end])

            pred_all = np.concatenate(preds, axis=0) * 1000.0
            gt_all = np.concatenate(gt_list, axis=0) * 1000.0
            err = np.mean(np.linalg.norm(pred_all - gt_all, axis=-1))
            errors.append(float(err))

        results[str(k)] = {
            "mean_mm": float(np.mean(errors)) if errors else None,
            "std_mm": float(np.std(errors)) if errors else None,
            "n_subsets": len(errors),
        }
    return results


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _scalar_summary(report: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(v) for k, v in report.items() if not k.endswith("_per_joint") and not isinstance(v, np.ndarray)}


def write_json(out_path: Path, results: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def write_csv(csv_path: Path, results: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if "clean" in results:
        for k, v in results["clean"].items():
            if isinstance(v, (int, float)):
                rows.append({"section": "clean", "condition": k, "mpjpe": "", "pa_mpjpe": "", "value": v})
    if "robustness" in results:
        for name, metrics in results["robustness"].items():
            rows.append({
                "section": "robustness",
                "condition": name,
                "mpjpe": metrics.get("mpjpe", ""),
                "pa_mpjpe": metrics.get("pa_mpjpe", ""),
                "value": "",
            })
    if "variable_views" in results:
        for k, metrics in results["variable_views"].items():
            rows.append({
                "section": "variable_views",
                "condition": f"k={k}",
                "mpjpe": metrics.get("mean_mm", ""),
                "pa_mpjpe": metrics.get("std_mm", ""),
                "value": "",
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "condition", "mpjpe", "pa_mpjpe", "value"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OmniMultiViewFusionV3 on MPI-INF-3DHP S2/Seq1",
    )
    # Inputs
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained OmniMultiViewFusionV3 checkpoint")
    parser.add_argument("--dataset", type=str, default=None, help="Path to MPI-INF-3DHP .npz (S2/Seq1)")
    parser.add_argument("--smoke", action="store_true", help="CPU/GPU smoke test on synthetic data")
    # Model
    parser.add_argument("--d", type=int, default=128, help="Model feature dimension")
    parser.add_argument("--residual_hidden", type=int, default=128, help="Residual MLP hidden size")
    parser.add_argument("--n_st_layers", type=int, default=2, help="Spatio-temporal transformer layers")
    parser.add_argument("--graph_num_layers", type=int, default=1, help="Graph-joint attention layers")
    parser.add_argument("--n_joint_layers", type=int, default=0, help="Dense joint-level transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    # v3 flags
    parser.add_argument("--use_multiscale_fusion", action="store_true", help="Enable v3 hierarchical multi-scale fusion")
    parser.add_argument("--use_camera_conditioning", action="store_true", help="Enable v3 camera conditioning")
    parser.add_argument("--use_epipolar_bias", action="store_true", help="Enable v3 epipolar-biased ST transformer")
    # Evaluation
    parser.add_argument("--clip_len", type=int, default=13, help="Temporal clip length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--val_stride", type=int, default=1, help="Stride for validation clips")
    parser.add_argument("--run_robustness", action="store_true", help="Run calibration-robustness matrix")
    parser.add_argument("--run_variable_views", action="store_true", help="Run variable-view MPJPE@k curve")
    parser.add_argument("--min_views", type=int, default=2, help="Minimum number of views for variable-view curve")
    parser.add_argument("--max_views", type=int, default=None, help="Maximum number of views for variable-view curve")
    parser.add_argument("--num_subsets_per_k", type=int, default=10, help="Number of random view subsets to sample per k")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Outputs
    parser.add_argument("--out_json", type=str, default="outputs/eval_omniview_fusion_v3_mpiinf3dhp.json", help="JSON output path")
    parser.add_argument("--out_csv", type=str, default="outputs/eval_omniview_fusion_v3_mpiinf3dhp.csv", help="CSV output path")
    args = parser.parse_args()

    if args.smoke:
        args.run_robustness = True
        args.run_variable_views = True
        args.clip_len = 9
        args.batch_size = 2
        args.num_subsets_per_k = 2
        if args.checkpoint is None:
            args.checkpoint = "__smoke__"
        # Default to full v3 stack in smoke mode so all code paths are exercised.
        args.use_multiscale_fusion = True
        args.use_camera_conditioning = True
        args.use_epipolar_bias = True
    else:
        if args.checkpoint is None or args.dataset is None:
            parser.error("--checkpoint and --dataset are required unless --smoke is set")

    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.smoke:
        print("Smoke mode: using synthetic dataset")
        K, R, t = _make_synthetic_cameras(n_views=4)
        n_views = 4
        n_joints = 17
        dataset = SyntheticSmokeDataset(K, R, t, n_frames=60, n_joints=n_joints, clip_len=args.clip_len)
        data_npz = None
    else:
        data_npz = np.load(args.dataset)
        n_views = int(data_npz["camera_K"].shape[0])
        n_joints = int(data_npz["points_2d"].shape[2])
        dataset = TemporalClipDataset(args.dataset, args.clip_len, stride=args.val_stride)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(args, n_views=n_views, j=n_joints).to(device)
    if args.checkpoint and args.checkpoint != "__smoke__":
        load_checkpoint(model, args.checkpoint)
    else:
        print("No checkpoint provided; using freshly initialised model for smoke test")
    model.eval()

    # ------------------------------------------------------------------
    # Clean evaluation
    # ------------------------------------------------------------------
    print("Clean evaluation...")
    clean_report = evaluate_clean(model, loader, device)
    clean_summary = _scalar_summary(clean_report)
    print(
        f"Clean: MPJPE={clean_summary['mpjpe']:.2f}mm "
        f"PA-MPJPE={clean_summary['pa_mpjpe']:.2f}mm"
    )

    results: Dict[str, Any] = {"clean": clean_summary}

    # ------------------------------------------------------------------
    # Robustness matrix
    # ------------------------------------------------------------------
    if args.run_robustness:
        conditions = robustness_conditions(smoke=args.smoke)
        robustness: Dict[str, Any] = {}
        print("Calibration-robustness matrix...")
        for name, cfg in conditions.items():
            if name == "clean":
                robustness[name] = clean_summary
                continue
            report = evaluate_perturbed(model, dataset, cfg, args.batch_size, device)
            robustness[name] = _scalar_summary(report)
            print(
                f"{name}: MPJPE={robustness[name]['mpjpe']:.2f}mm "
                f"PA-MPJPE={robustness[name]['pa_mpjpe']:.2f}mm"
            )
        results["robustness"] = robustness

    # ------------------------------------------------------------------
    # Variable-view curve
    # ------------------------------------------------------------------
    if args.run_variable_views:
        if args.smoke or data_npz is not None:
            if args.smoke:
                points_2d = dataset.points_2d.numpy()
                confidences = dataset.confidences.numpy()
                joints_3d = dataset.joints_3d.numpy()
                K = dataset.K
                R = dataset.R
                t = dataset.t
            else:
                points_2d = data_npz["points_2d"]
                confidences = data_npz["confidences"]
                joints_3d = data_npz["joints_3d"]
                K = torch.from_numpy(data_npz["camera_K"]).float()
                R = torch.from_numpy(data_npz["camera_R"]).float()
                t = torch.from_numpy(data_npz["camera_t"]).float()

            print("Variable-view MPJPE@k curve...")
            results["variable_views"] = evaluate_variable_views(
                model,
                points_2d,
                confidences,
                joints_3d,
                K,
                R,
                t,
                clip_len=args.clip_len,
                device=device,
                min_views=args.min_views,
                max_views=args.max_views,
                num_subsets_per_k=args.num_subsets_per_k,
                seed=args.seed,
            )
            for k, r in results["variable_views"].items():
                print(f"  k={k}: mean={r['mean_mm']:.2f}mm std={r['std_mm']:.2f}mm n={r['n_subsets']}")
        else:
            print("Variable-view evaluation skipped: no dataset loaded")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    write_json(out_json, results)
    write_csv(out_csv, results)
    print(f"Saved JSON -> {out_json}")
    print(f"Saved CSV  -> {out_csv}")


if __name__ == "__main__":
    main()
