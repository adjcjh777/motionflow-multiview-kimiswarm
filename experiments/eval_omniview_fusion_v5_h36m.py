"""Evaluation script for OmniMultiViewFusionV5 on Human3.6M (H36M).

Mirrors ``experiments/eval_omniview_fusion_v5_mpiinf3dhp.py`` but targets the
WebBridge H36M .npz files (17-joint skeleton, 4 views, units in metres).

Supports the v25 multi-view geometry fusion module toggles so that a v25
trained checkpoint can be benchmarked on H36M S9 (validation) or S11 (test).

Usage
-----
    # S9 validation, full v25 model
    python experiments/eval_omniview_fusion_v5_h36m.py \
        --checkpoint outputs/omniview_fusion_v5_webbridge_multi.pth \
        --dataset data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz \
        --use_multiview_geometry_fusion_v25 \
        --run_robustness --run_variable_views

    # CPU/GPU smoke test (synthetic data, fresh model)
    python experiments/eval_omniview_fusion_v5_h36m.py --smoke
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
from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5


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

def build_model(args: argparse.Namespace, n_views: int, j: int) -> OmniMultiViewFusionV5:
    model = OmniMultiViewFusionV5(
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
        # v3/v4 geometry toggles
        use_multiscale_fusion=args.use_multiscale_fusion,
        use_camera_conditioning=args.use_camera_conditioning,
        use_epipolar_bias=args.use_epipolar_bias,
        # v4 toggles
        use_context_visibility=args.use_context_visibility,
        use_skeleton_residual=args.use_skeleton_residual,
        use_kinematic_refiner=args.use_kinematic_refiner,
        use_adaptive_view_selection=args.use_adaptive_view_selection,
        use_rotation_correction=args.use_rotation_correction,
        use_entropy_regularization=args.use_entropy_regularization,
        entropy_weight=args.entropy_weight,
        adaptive_view_target_k=args.adaptive_view_target_k,
        rotation_max_rot_deg=args.rotation_max_rot_deg,
        # v5 toggles
        use_camera_view_embedding=args.use_camera_view_embedding,
        use_set_view_aggregator=args.use_set_view_aggregator,
        camera_view_embedding_hidden=args.camera_view_embedding_hidden,
        set_view_n_isab_layers=args.set_view_n_isab_layers,
        set_view_num_inducing_points=args.set_view_num_inducing_points,
        set_view_dropout=args.set_view_dropout,
        # v6 toggles
        use_perceiver_aggregator=args.use_perceiver_aggregator,
        perceiver_n_latents=args.perceiver_n_latents,
        perceiver_n_layers=args.perceiver_n_layers,
        perceiver_n_heads=args.perceiver_n_heads,
        perceiver_dropout=args.perceiver_dropout,
        # v25 toggles
        use_multiview_geometry_fusion_v25=args.use_multiview_geometry_fusion_v25,
        v25_use_geometry_attention=args.v25_use_geometry_attention,
        v25_use_learned_depth_triangulation=args.v25_use_learned_depth_triangulation,
        v25_use_geometry_bundle_adjustment=args.v25_use_geometry_bundle_adjustment,
        v25_use_camera_joint_graph=args.v25_use_camera_joint_graph,
        v25_use_outlier_view_detector=args.v25_use_outlier_view_detector,
        v25_outlier_z_thresh=args.v25_outlier_z_thresh,
        v25_outlier_soft_beta=args.v25_outlier_soft_beta,
        v25_geom_loss_weight=args.v25_geom_loss_weight,
        use_temporal_geometry_fusion_v26=args.use_temporal_geometry_fusion_v26,
        v26_temporal_window=args.v26_temporal_window,
        use_uncertainty_depth_proposals_v27=args.use_uncertainty_depth_proposals_v27,
        v27_uncertainty_loss_weight=args.v27_uncertainty_loss_weight,
        v27_udp_n_mixtures=args.v27_udp_n_mixtures,
        use_physical_space_alignment_v28=args.use_physical_space_alignment_v28,
        v28_floor_loss_weight=args.v28_floor_loss_weight,
        v28_bone_temporal_weight=args.v28_bone_temporal_weight,
        use_test_time_self_evolution_v27=args.use_test_time_self_evolution_v27,
        v27_tte_n_iters=args.v27_tte_n_iters,
        v27_tte_sigma_reproj=args.v27_tte_sigma_reproj,
        v27_tte_residual_thresh_mm=args.v27_tte_residual_thresh_mm,
    )
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = None
    if isinstance(state, dict):
        config = state.get("config")
        if "model" in state:
            state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Checkpoint load: missing keys {missing[:10]}")
    if unexpected:
        print(f"Checkpoint load: unexpected keys ignored {unexpected[:10]}")
    return config


def load_training_config(checkpoint_path: str, config_json: str | None = None) -> Dict[str, Any] | None:
    """Load the training config saved alongside a checkpoint.

    The training script writes ``<checkpoint>.config.json`` next to the
    checkpoint.  If that file exists, return its contents; otherwise return
    ``None``.  An explicit ``--config_json`` path overrides the default
    side-car location.
    """
    if config_json is None:
        candidate = Path(checkpoint_path).with_suffix(".config.json")
    else:
        candidate = Path(config_json)
    if not candidate.exists():
        return None
    try:
        with open(candidate, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: could not load config from {candidate}: {exc}")
        return None


# v25/v27 architecture flags whose defaults should be inferred from the
# training config when the user does not explicitly set them on the CLI.
_V25_FLAG_NAMES = (
    "use_multiview_geometry_fusion_v25",
    "v25_use_geometry_attention",
    "v25_use_learned_depth_triangulation",
    "v25_use_geometry_bundle_adjustment",
    "v25_use_camera_joint_graph",
    "v25_use_outlier_view_detector",
    "use_temporal_geometry_fusion_v26",
    "use_uncertainty_depth_proposals_v27",
)


def restore_v25_flags(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Infer v25 architecture flags from saved training config.

    Boolean flags use ``default=None`` so we can distinguish "not set on the
    CLI" from "explicitly disabled".  When a flag is unset, we copy the value
    from the saved config; otherwise we leave the user's choice alone.
    """
    for name in _V25_FLAG_NAMES:
        if getattr(args, name) is None:
            value = config.get(name)
            if isinstance(value, bool):
                setattr(args, name, value)
            elif isinstance(value, str):
                setattr(args, name, value.lower() in {"true", "1", "yes"})
            else:
                setattr(args, name, False)


def normalise_v25_flags(args: argparse.Namespace) -> None:
    """Convert any remaining ``None`` v25 flags to ``False``."""
    for name in _V25_FLAG_NAMES:
        if getattr(args, name) is None:
            setattr(args, name, False)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_clean(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    visibility_threshold: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    preds, gts = [], []
    vis_preds, vis_gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            out = model(xb, K=K, R=R, t=t)
            pred = out[0]
            visibility = out[2]
            confidences = xb[..., 2]
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())

            vis_pred = (visibility > visibility_threshold).float().cpu().numpy()
            vis_gt = (confidences > 0).float().cpu().numpy()
            vis_preds.append(vis_pred)
            vis_gts.append(vis_gt)

    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    report = compute_all_metrics(preds, gts)

    if vis_preds:
        vis_pred_all = np.concatenate(vis_preds, axis=0).reshape(-1, vis_preds[0].shape[-2])
        vis_gt_all = np.concatenate(vis_gts, axis=0).reshape(-1, vis_gts[0].shape[-2])
        report["visibility_accuracy"] = float((vis_pred_all == vis_gt_all).mean())
    else:
        report["visibility_accuracy"] = float("nan")

    return report


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

                view_mask = torch.zeros(clip_len, V, device=device)
                view_mask[:, list(subset)] = 1.0

                with torch.no_grad():
                    pred = model(x_clip, K=Kp, R=Rp, t=tp, view_mask=view_mask)[0]
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
    summary: Dict[str, float] = {}
    for k, v in report.items():
        if k.endswith("_per_joint") or isinstance(v, np.ndarray):
            continue
        if isinstance(v, (int, float)):
            summary[k] = float(v)
    return summary


def _convert_for_json(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    return obj


def write_json(out_path: Path, results: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_convert_for_json(results), f, indent=2)


def write_csv(csv_path: Path, results: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if "clean" in results:
        for k, v in results["clean"].items():
            if isinstance(v, (int, float)):
                rows.append({"section": "clean", "condition": k, "mpjpe": "", "pa_mpjpe": "", "value": v})
            elif isinstance(v, list):
                rows.append({"section": "clean", "condition": k, "mpjpe": "", "pa_mpjpe": "", "value": str(v)})
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
        description="Evaluate OmniMultiViewFusionV5 on Human3.6M",
    )
    # Inputs
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained OmniMultiViewFusionV5 checkpoint")
    parser.add_argument("--dataset", type=str, default=None, help="Path to H36M .npz (e.g. s_09_acts_02_multiview_m.npz)")
    parser.add_argument("--config_json", type=str, default=None, help="Path to training config JSON (default: <checkpoint>.config.json)")
    parser.add_argument("--smoke", action="store_true", help="CPU/GPU smoke test on synthetic data")
    # Model
    parser.add_argument("--d", type=int, default=128, help="Model feature dimension")
    parser.add_argument("--residual_hidden", type=int, default=128, help="Residual MLP hidden size")
    parser.add_argument("--n_st_layers", type=int, default=2, help="Spatio-temporal transformer layers")
    parser.add_argument("--graph_num_layers", type=int, default=1, help="Graph-joint attention layers")
    parser.add_argument("--n_joint_layers", type=int, default=0, help="Dense joint-level transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    # v4 toggles
    parser.add_argument("--use_multiscale_fusion", action="store_true", help="Enable multiscale fusion")
    parser.add_argument("--use_camera_conditioning", action="store_true", help="Enable camera conditioning")
    parser.add_argument("--use_epipolar_bias", action="store_true", help="Enable epipolar bias")
    parser.add_argument("--use_context_visibility", action="store_true", help="Enable context-aware visibility head")
    parser.add_argument("--use_skeleton_residual", action="store_true", help="Enable skeleton-graph residual refiner")
    parser.add_argument("--use_kinematic_refiner", action="store_true", help="Enable kinematic-chain final refiner")
    parser.add_argument("--use_adaptive_view_selection", action="store_true", help="Enable adaptive view selector")
    parser.add_argument("--use_rotation_correction", action="store_true", help="Enable rotation correction head")
    parser.add_argument("--use_entropy_regularization", action="store_true", help="Enable attention-entropy regularisation")
    parser.add_argument("--entropy_weight", type=float, default=0.01, help="Weight for entropy loss")
    parser.add_argument("--adaptive_view_target_k", type=int, default=2, help="Target number of active views when adaptive selection is enabled")
    parser.add_argument("--rotation_max_rot_deg", type=float, default=2.0, help="Maximum rotation correction residual in degrees")
    # v5 toggles
    parser.add_argument("--use_camera_view_embedding", action="store_true", help="Use camera-conditioned view embedding")
    parser.add_argument("--use_set_view_aggregator", action="store_true", help="Use permutation-invariant set-transformer aggregator over views")
    parser.add_argument("--camera_view_embedding_hidden", type=int, default=32, help="Hidden dimension of camera-conditioned view embedding MLP")
    parser.add_argument("--set_view_n_isab_layers", type=int, default=2, help="Number of ISAB layers in set aggregator")
    parser.add_argument("--set_view_num_inducing_points", type=int, default=32, help="Number of inducing points in each ISAB")
    parser.add_argument("--set_view_dropout", type=float, default=0.0, help="Dropout probability in set aggregator")
    # v6 toggles
    parser.add_argument("--use_perceiver_aggregator", action="store_true", help="Use Perceiver-style view aggregator")
    parser.add_argument("--perceiver_n_latents", type=int, default=16, help="Number of latent vectors in Perceiver aggregator")
    parser.add_argument("--perceiver_n_layers", type=int, default=2, help="Number of Perceiver latent layers")
    parser.add_argument("--perceiver_n_heads", type=int, default=4, help="Number of attention heads in Perceiver aggregator")
    parser.add_argument("--perceiver_dropout", type=float, default=0.0, help="Dropout in Perceiver aggregator")
    # v25 toggles.  default=None lets us tell "not supplied on CLI" apart from
    # "explicitly disabled", so the saved training config can be honoured.
    parser.add_argument("--use_multiview_geometry_fusion_v25", action="store_true", default=None, help="Enable v25 multi-view geometry fusion module")
    parser.add_argument("--v25_use_geometry_attention", action="store_true", default=None, help="Enable v25 geometry-aware cross-view attention")
    parser.add_argument("--v25_use_learned_depth_triangulation", action="store_true", default=None, help="Enable v25 learned depth-proposal triangulation")
    parser.add_argument("--v25_use_geometry_bundle_adjustment", action="store_true", default=None, help="Enable v25 geometry bundle adjustment")
    parser.add_argument("--v25_use_camera_joint_graph", action="store_true", default=None, help="Enable v25 camera-joint graph")
    parser.add_argument("--v25_use_outlier_view_detector", action="store_true", default=None, help="Enable v25 outlier-view detector")
    parser.add_argument("--v25_outlier_z_thresh", type=float, default=3.0, help="Robust z-score threshold for v25 outlier-view detector")
    parser.add_argument("--v25_outlier_soft_beta", type=float, default=1.0, help="Softness of exponential down-weighting for v25 outlier-view detector")
    parser.add_argument("--use_temporal_geometry_fusion_v26", action="store_true", default=None, help="Enable v26 temporal geometry fusion")
    parser.add_argument("--v26_temporal_window", type=int, default=3, help="Temporal window size for v26")
    parser.add_argument("--use_uncertainty_depth_proposals_v27", action="store_true", default=None, help="Enable v27 uncertainty-aware depth-proposal triangulation")
    parser.add_argument("--v27_uncertainty_loss_weight", type=float, default=0.01, help="Weight for v27 uncertainty regularisation loss")
    parser.add_argument("--v27_udp_n_mixtures", type=int, default=1, help="Number of Gaussian mixture components for v27 depth proposals")
    parser.add_argument("--use_physical_space_alignment_v28", action="store_true", default=None, help="Enable v28 physical-space alignment refiner")
    parser.add_argument("--v28_floor_loss_weight", type=float, default=0.0, help="Weight for v28 floor consistency loss")
    parser.add_argument("--v28_bone_temporal_weight", type=float, default=0.0, help="Weight for v28 bone-length temporal consistency loss")
    # v27 toggles
    parser.add_argument("--use_test_time_self_evolution_v27", action="store_true", default=False, help="Enable v27 test-time self-evolution at inference")
    parser.add_argument("--v27_tte_n_iters", type=int, default=3, help="Number of iterations for v27 self-evolution")
    parser.add_argument("--v27_tte_sigma_reproj", type=float, default=5.0, help="Cauchy kernel scale (pixels) for v27 self-evolution")
    parser.add_argument("--v27_tte_residual_thresh_mm", type=float, default=0.5, help="Early-stop threshold (mm) for v27 self-evolution")
    parser.add_argument("--v25_geom_loss_weight", type=float, default=0.1, help="Weight for v25 geometry loss during training")
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
    parser.add_argument("--out_json", type=str, default="outputs/eval_omniview_fusion_v5_h36m.json", help="JSON output path")
    parser.add_argument("--out_csv", type=str, default="outputs/eval_omniview_fusion_v5_h36m.csv", help="CSV output path")
    args = parser.parse_args()

    if args.smoke:
        args.run_robustness = True
        args.run_variable_views = True
        args.clip_len = 9
        args.batch_size = 2
        args.num_subsets_per_k = 2
        if args.checkpoint is None:
            args.checkpoint = "__smoke__"
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

    if args.checkpoint and args.checkpoint != "__smoke__":
        # The training script saves architecture flags in a side-car
        # ``<checkpoint>.config.json``.  Prefer that over any config embedded in
        # the checkpoint state dict.
        saved_config = load_training_config(args.checkpoint, args.config_json)
        if saved_config is not None:
            restore_v25_flags(args, saved_config)
            print("Restored v25 flags from training config.")

    normalise_v25_flags(args)

    model = build_model(args, n_views=n_views, j=n_joints).to(device)
    if args.checkpoint and args.checkpoint != "__smoke__":
        load_checkpoint(model, args.checkpoint)
    else:
        print("No checkpoint provided; using freshly initialised model for smoke test")
    model.eval()

    print("Clean evaluation...")
    clean_report = evaluate_clean(model, loader, device)
    clean_summary = _scalar_summary(clean_report)
    print(
        f"Clean: MPJPE={clean_summary['mpjpe']:.2f}mm "
        f"PA-MPJPE={clean_summary['pa_mpjpe']:.2f}mm "
        f"VisAcc={clean_summary.get('visibility_accuracy', float('nan')):.3f}"
    )

    per_joint_keys = [k for k in clean_report if "per_joint" in k or k == "visibility_accuracy"]
    for k in per_joint_keys:
        clean_summary[k] = clean_report[k]

    results: Dict[str, Any] = {"clean": clean_summary}

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

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    write_json(out_json, results)
    write_csv(out_csv, results)
    print(f"Saved JSON -> {out_json}")
    print(f"Saved CSV  -> {out_csv}")


if __name__ == "__main__":
    main()
