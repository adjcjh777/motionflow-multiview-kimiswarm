#!/usr/bin/env python3
"""Failure-mode analysis for the v25 Multi-View Geometry Fusion module.

Loads a v25 checkpoint (standalone or extracted from ``OmniMultiViewFusionV5``)
and a validation ``.npz`` dataset, evaluates every frame, and produces a
Markdown + JSON report that bins failures by both generic pose-estimation
covariates (views, occlusion, baseline, depth, motion) and v25-specific
geometry cues (reprojection error, depth-proposal disagreement, ray-intersection
quality, outlier views).

Intended for off-line diagnostics on the local RTX 4090 / CPU while the A800-D
runs are still training.

Examples
--------
# Run on a real validation npz (CPU is fine for a small set)
python scripts/analyze_v25_failures.py \
    --checkpoint outputs/omniview_fusion_v25_geometry_fusion_small.pth \
    --dataset data/webbridge/h36m_meters/s_11_acts_02_multiview_m.npz \
    --out_dir outputs/v25_failure_analysis

# Smoke-test with synthetic data, no checkpoint (uses init weights)
python scripts/analyze_v25_failures.py --synthetic --out_dir outputs/v25_failure_analysis_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Allow this script to be run from any cwd when repo root is the parent.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motionflow_mv.eval.metrics import mpjpe as mpjpe_metric
from motionflow_mv.eval.metrics import pa_mpjpe as pa_mpjpe_metric
from motionflow_mv.fusion.multiview_geometry_fusion_v25 import (
    MultiViewGeometryFusionV25,
    compute_rays,
    ray_intersection_logit,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    """Frame-level multi-view pose dataset compatible with the v25 module."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.points_2d = data["points_2d"].astype(np.float32)  # (F, V, J, 2)
        self.confidences = data["confidences"].astype(np.float32)  # (F, V, J)
        self.joints_3d = data["joints_3d"].astype(np.float32)  # (F, J, 3)
        self.K = data["camera_K"].astype(np.float32)
        self.R = data["camera_R"].astype(np.float32)
        self.t = data["camera_t"].astype(np.float32)
        self.n_frames = self.points_2d.shape[0]
        self.n_views = self.points_2d.shape[1]

        # Broadcast static camera parameters if they are not frame-dependent.
        if self.K.ndim == 3:
            self.K = np.broadcast_to(self.K, (self.n_frames, self.n_views, 3, 3)).copy()
        if self.R.ndim == 3:
            self.R = np.broadcast_to(self.R, (self.n_frames, self.n_views, 3, 3)).copy()
        if self.t.ndim == 2:
            self.t = np.broadcast_to(self.t, (self.n_frames, self.n_views, 3)).copy()

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # v25 expects a temporal dimension of size 1; we add it in collate.
        x = np.concatenate(
            [self.points_2d[idx], self.confidences[idx][..., None]],
            axis=-1,
        )  # (V, J, 3)
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.joints_3d[idx]),
            torch.from_numpy(self.K[idx]),
            torch.from_numpy(self.R[idx]),
            torch.from_numpy(self.t[idx]),
            idx,
        )


def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    """Stack frames batch and insert a singleton temporal dimension."""
    x = torch.stack([b[0] for b in batch], dim=0).unsqueeze(1)  # (B, 1, V, J, 3)
    y = torch.stack([b[1] for b in batch], dim=0).unsqueeze(1)  # (B, 1, J, 3)
    K = torch.stack([b[2] for b in batch], dim=0).unsqueeze(1)  # (B, 1, V, 3, 3)
    R = torch.stack([b[3] for b in batch], dim=0).unsqueeze(1)
    t = torch.stack([b[4] for b in batch], dim=0).unsqueeze(1)
    idx = torch.tensor([b[5] for b in batch], dtype=torch.long)
    return x, y, K, R, t, idx


# ---------------------------------------------------------------------------
# Synthetic data generator (for smoke tests)
# ---------------------------------------------------------------------------

def _make_cameras(n_views: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple circular camera rig."""
    Ks, Rs, ts = [], [], []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t_vec = -R @ c
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        Ks.append(K)
        Rs.append(R.astype(np.float32))
        ts.append(t_vec.astype(np.float32))
    return np.stack(Ks), np.stack(Rs), np.stack(ts)


def _project_points(
    joints_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Project world points into all views; returns (F, V, J, 2)."""
    X_cam = np.einsum("vab,fjb->vfja", R, joints_3d) + t[:, None, None, :]
    z = np.maximum(X_cam[..., 2:3], 1e-6)
    uv = np.matmul(K[:, None, None, :, :], (X_cam / z)[..., None])
    return uv[..., :2, 0] / uv[..., 2:3, 0]


def make_synthetic_npz(path: Path, F: int = 128, V: int = 4, J: int = 17) -> None:
    """Write a small synthetic multi-view npz to *path* for smoke testing."""
    np.random.seed(0)
    K, R, t = _make_cameras(V)

    # A simple moving subject.
    t_vec = np.linspace(0, 4 * np.pi, F)
    joints_3d = np.zeros((F, J, 3), dtype=np.float32)
    for j in range(J):
        phase = 2 * np.pi * j / max(J, 1)
        joints_3d[:, j, 0] = np.sin(t_vec + phase) * (0.3 + 0.1 * (j % 3))
        joints_3d[:, j, 1] = np.cos(t_vec + phase) * 0.3
        joints_3d[:, j, 2] = 1.5 + 0.2 * np.sin(2 * t_vec + phase)

    points_2d = _project_points(joints_3d, K, R, t).transpose(1, 0, 2, 3)
    confidences = np.ones((F, V, J), dtype=np.float32)

    # Drop a few views / joints to simulate occlusion.
    for f in range(F):
        if f % 7 == 0:
            confidences[f, f % V, j % J] = 0.0
        if f % 13 == 0:
            confidences[f, 1:, j % J] = 0.0

    np.savez(
        path,
        points_2d=points_2d.astype(np.float32),
        confidences=confidences,
        joints_3d=joints_3d.astype(np.float32),
        camera_K=K.astype(np.float32),
        camera_R=R.astype(np.float32),
        camera_t=t.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_v25_model(
    checkpoint_path: Optional[str],
    n_views: int,
    d: int,
    n_heads: int,
    n_geometry_layers: int,
    n_ray_samples: int,
    use_geometry_attention: bool,
    use_learned_depth_triangulation: bool,
    use_geometry_bundle_adjustment: bool,
    device: torch.device,
    state_dict_prefix: Optional[str] = None,
) -> nn.Module:
    """Build a v25 module and optionally load a checkpoint into it."""
    model = MultiViewGeometryFusionV25(
        d=d,
        n_heads=n_heads,
        n_views=n_views,
        n_geometry_layers=n_geometry_layers,
        n_ray_samples=n_ray_samples,
        use_geometry_attention=use_geometry_attention,
        use_learned_depth_triangulation=use_learned_depth_triangulation,
        use_geometry_bundle_adjustment=use_geometry_bundle_adjustment,
        use_camera_joint_graph=False,
    )
    model.to(device)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict):
            keys = list(state.keys())
            # If the checkpoint is from the full OmniMultiViewFusionV5 model,
            # strip the v25 submodule prefix.
            prefix = state_dict_prefix or "multiview_geometry_fusion_v25."
            if all(k.startswith(prefix) for k in keys):
                state = {k[len(prefix):]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint into v25 module from {checkpoint_path}")

    else:
        print("No checkpoint provided; using initialised (identity-at-init) v25 module.")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Geometry / analysis helpers
# ---------------------------------------------------------------------------

def camera_span_deg(R: torch.Tensor) -> float:
    """Largest pairwise angle between camera optical axes."""
    forwards = R[:, :, 2]  # (V, 3)
    forwards = forwards / (torch.linalg.norm(forwards, dim=-1, keepdim=True) + 1e-8)
    max_angle = 0.0
    V = forwards.shape[0]
    for i in range(V):
        for j in range(i + 1, V):
            cos = torch.clamp(torch.dot(forwards[i], forwards[j]), -1.0, 1.0)
            angle = math.degrees(torch.acos(cos).item())
            if angle > max_angle:
                max_angle = angle
    return max_angle


def per_sample_reprojection_error(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    confidence: torch.Tensor,
    view_mask: torch.Tensor,
) -> float:
    """Mean reprojection error in pixels for a single frame, weighted by confidence."""
    # pred_3d: (1, 1, J, 3); points_2d: (1, 1, V, J, 2)
    V = points_2d.shape[2]
    X_exp = pred_3d.unsqueeze(2).expand(-1, -1, V, -1, -1).permute(0, 1, 2, 4, 3)
    X_cam = torch.matmul(R, X_exp)  # (1, 1, V, 3, J)
    X_cam = X_cam.permute(0, 1, 2, 4, 3) + t[..., None, :]
    Z = X_cam[..., 2:3]
    Z_safe = Z.sign() * (Z.abs() + 1e-6)
    X_norm = X_cam / Z_safe
    proj = torch.matmul(K[..., None, :, :], X_norm[..., None]).squeeze(-1)
    proj_2d = proj[..., :2] / proj[..., 2:3]
    diff = (proj_2d - points_2d).norm(dim=-1)  # (1, 1, V, J)
    weights = confidence * view_mask[:, :, :, None].float()
    if weights.sum() == 0:
        return 0.0
    return float((diff * weights).sum() / weights.sum().clamp(min=1e-6))


def depth_uncertainty(
    pred_3d: torch.Tensor,
    centre: torch.Tensor,
    direction: torch.Tensor,
    view_mask: torch.Tensor,
) -> float:
    """Mean per-joint std of per-view depth along rays; higher = more disagreement."""
    # pred_3d: (1, 1, J, 3); centre/direction: (1, 1, V, 3)
    delta = pred_3d.unsqueeze(2) - centre.unsqueeze(3)  # (1, 1, V, J, 3)
    depths = (delta * direction).sum(dim=-1)  # (1, 1, V, J)
    mask = view_mask[:, :, :, None]  # (1, 1, V, 1)
    # Compute per-joint std ignoring masked views.
    count = mask.sum(dim=2).squeeze()  # (J,)
    mean = (depths * mask).sum(dim=2).squeeze() / count.clamp(min=1e-6)
    mean_sq = ((depths ** 2) * mask).sum(dim=2).squeeze() / count.clamp(min=1e-6)
    std = (mean_sq - mean ** 2).clamp(min=0.0).sqrt()
    valid = count > 1
    if not valid.any():
        return 0.0
    return float(std[valid].mean())


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

@dataclass
class SampleRecord:
    frame_id: int
    mpjpe: float
    pa_mpjpe: float
    active_views: int
    missing_ratio: float
    mean_conf: float
    camera_span_deg: float
    root_depth: float
    root_velocity: float
    reproj_error: float
    depth_uncertainty: float
    ray_quality: float
    geom_loss: float
    outlier_view: bool
    categories: List[str] = field(default_factory=list)


def compute_thresholds(records: List[SampleRecord]) -> Dict[str, float]:
    mpjpes = np.array([r.mpjpe for r in records])
    confs = np.array([r.mean_conf for r in records])
    spans = np.array([r.camera_span_deg for r in records])
    depths = np.array([r.root_depth for r in records])
    velocities = np.array([r.root_velocity for r in records])
    reprojs = np.array([r.reproj_error for r in records])
    dus = np.array([r.depth_uncertainty for r in records])
    rays = np.array([r.ray_quality for r in records])

    def _p(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if len(arr) else 0.0

    return {
        "few_views": 3,
        "occlusion": 0.2,
        "low_confidence": max(0.7, _p(confs, 20) - 1e-6),
        "large_baseline": max(120.0, _p(spans, 80)),
        "far_subject": _p(depths, 80),
        "high_motion": _p(velocities, 80),
        "high_mpjpe": _p(mpjpes, 80),
        "high_reprojection": _p(reprojs, 80),
        "high_depth_uncertainty": _p(dus, 80),
        "low_ray_quality": _p(rays, 20),
    }


def categorise(record: SampleRecord, thresholds: Dict[str, float]) -> List[str]:
    cats: List[str] = []
    if record.active_views <= thresholds["few_views"]:
        cats.append("few_views")
    if record.missing_ratio >= thresholds["occlusion"]:
        cats.append("occlusion")
    if record.mean_conf <= thresholds["low_confidence"]:
        cats.append("low_confidence")
    if record.camera_span_deg >= thresholds["large_baseline"]:
        cats.append("large_baseline")
    if record.root_depth >= thresholds["far_subject"]:
        cats.append("far_subject")
    if record.root_velocity >= thresholds["high_motion"]:
        cats.append("high_motion")
    if record.reproj_error >= thresholds["high_reprojection"]:
        cats.append("high_reprojection")
    if record.depth_uncertainty >= thresholds["high_depth_uncertainty"]:
        cats.append("high_depth_uncertainty")
    if record.ray_quality <= thresholds["low_ray_quality"]:
        cats.append("low_ray_quality")
    if record.outlier_view:
        cats.append("outlier_view")
    return cats


def run_v25_failure_analysis(
    checkpoint_path: Optional[str],
    dataset_path: str,
    out_dir: Path,
    d: int,
    n_heads: int,
    n_geometry_layers: int,
    n_ray_samples: int,
    use_geometry_attention: bool,
    use_learned_depth_triangulation: bool,
    use_geometry_bundle_adjustment: bool,
    state_dict_prefix: Optional[str],
    batch_size: int,
    top_k: int,
    device: torch.device,
) -> Tuple[List[SampleRecord], Dict[str, Any]]:
    dataset = FrameDataset(dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = load_v25_model(
        checkpoint_path,
        n_views=dataset.n_views,
        d=d,
        n_heads=n_heads,
        n_geometry_layers=n_geometry_layers,
        n_ray_samples=n_ray_samples,
        use_geometry_attention=use_geometry_attention,
        use_learned_depth_triangulation=use_learned_depth_triangulation,
        use_geometry_bundle_adjustment=use_geometry_bundle_adjustment,
        device=device,
        state_dict_prefix=state_dict_prefix,
    )

    all_records: List[SampleRecord] = []
    prev_root: Optional[np.ndarray] = None

    with torch.no_grad():
        for xb, yb, K, R, t, idxs in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)

            confidence = xb[..., 2]
            points_2d = xb[..., :2]
            view_mask = (confidence.max(dim=-1).values > 0.0)  # (B, 1, V)

            pred_3d, geom_loss = model(
                xb,
                K=K,
                R=R,
                t=t,
                view_mask=view_mask,
            )

            pred_np = pred_3d.cpu().numpy()
            gt_np = yb.cpu().numpy()
            conf_np = confidence.cpu().numpy()

            # Ray geometry for diagnostics.
            centre, direction = compute_rays(points_2d, K, R, t)
            sigma_d = model.geom_attn_layers[0].sigma_d if model.geom_attn_layers else torch.tensor(0.5, device=device)
            sigma_a = model.geom_attn_layers[0].sigma_a if model.geom_attn_layers else torch.tensor(0.5, device=device)
            ray_logits = ray_intersection_logit(centre, direction, sigma_d, sigma_a)

            for i, frame_id in enumerate(idxs.tolist()):
                p = pred_np[i, 0]
                g = gt_np[i, 0]
                c = conf_np[i, 0]
                vm = view_mask[i, 0]

                mpjpe = mpjpe_metric(p * 1000.0, g * 1000.0)
                pa = pa_mpjpe_metric(p * 1000.0, g * 1000.0)

                missing = (c <= 0.0).sum()
                total = c.size
                missing_ratio = float(missing / total)
                mean_conf = float(c.mean())
                active_views = int((c.max(axis=-1) > 0.0).sum())
                span = camera_span_deg(R[i, 0])
                root = g[0]
                root_depth = float(root[2])
                velocity = 0.0
                if prev_root is not None:
                    velocity = float(np.linalg.norm(root - prev_root))

                reproj = per_sample_reprojection_error(
                    pred_3d[i : i + 1],
                    points_2d[i : i + 1],
                    K[i : i + 1],
                    R[i : i + 1],
                    t[i : i + 1],
                    confidence[i : i + 1],
                    vm[None, None, :],
                )

                du = depth_uncertainty(
                    pred_3d[i : i + 1],
                    centre[i : i + 1],
                    direction[i : i + 1],
                    vm[None, None, :],
                )

                # Average ray-intersection logit over joint/view-pairs, ignoring self-pairs and masked views.
                logit = ray_logits[i, 0]  # (V, V, J)
                vm_np = vm.cpu().numpy()
                mask_2d = vm_np[:, None] & vm_np[None, :]
                logit_np = logit.cpu().numpy()
                vals = []
                V = logit.shape[0]
                for vi in range(V):
                    for vj in range(V):
                        if vi != vj and mask_2d[vi, vj]:
                            vals.append(logit_np[vi, vj].mean())
                ray_quality = float(np.mean(vals)) if vals else 0.0

                # Detect outlier views by per-view reprojection error vs median.
                outlier_present = False
                if active_views >= 3:
                    view_errors = []
                    for v in range(dataset.n_views):
                        if not vm[v]:
                            continue
                        single_view_mask = torch.zeros_like(vm)
                        single_view_mask[v] = True
                        err = per_sample_reprojection_error(
                            pred_3d[i : i + 1],
                            points_2d[i : i + 1],
                            K[i : i + 1],
                            R[i : i + 1],
                            t[i : i + 1],
                            confidence[i : i + 1],
                            single_view_mask[None, None, :],
                        )
                        view_errors.append(err)
                    if len(view_errors) > 1:
                        active_err = np.array(view_errors)
                        median_err = np.median(active_err)
                        outlier_present = bool(np.any(active_err[active_err > 0] > 2.0 * median_err))

                all_records.append(
                    SampleRecord(
                        frame_id=frame_id,
                        mpjpe=mpjpe,
                        pa_mpjpe=pa,
                        active_views=active_views,
                        missing_ratio=missing_ratio,
                        mean_conf=mean_conf,
                        camera_span_deg=span,
                        root_depth=root_depth,
                        root_velocity=velocity,
                        reproj_error=reproj,
                        depth_uncertainty=du,
                        ray_quality=ray_quality,
                        geom_loss=float(geom_loss.item()) if torch.is_tensor(geom_loss) else float(geom_loss),
                        outlier_view=outlier_present,
                    )
                )
                prev_root = root

    thresholds = compute_thresholds(all_records)
    for r in all_records:
        r.categories = categorise(r, thresholds)

    metadata = {
        "n_frames": len(dataset),
        "dataset_name": Path(dataset_path).as_posix(),
        "thresholds": {k: float(v) for k, v in thresholds.items()},
    }
    return all_records, metadata


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(records: List[SampleRecord], metadata: Dict[str, Any], top_k: int) -> str:
    records_sorted = sorted(records, key=lambda r: r.mpjpe, reverse=True)
    top_records = records_sorted[:top_k]
    mean_mpjpe = np.mean([r.mpjpe for r in records])
    mean_pa = np.mean([r.pa_mpjpe for r in records])

    cat_counts: Counter[str] = Counter()
    cat_mpjpe: Dict[str, List[float]] = {}
    for r in records:
        for cat in r.categories:
            cat_counts[cat] += 1
            cat_mpjpe.setdefault(cat, []).append(r.mpjpe)

    lines: List[str] = []
    lines.append("# v25 Failure-Mode Analysis Report\n")
    lines.append(f"*Generated:* {datetime.now(timezone.utc).isoformat()}Z  ")
    lines.append(f"*Dataset:* `{metadata['dataset_name']}`  ")
    lines.append(f"*Frames analysed:* {metadata['n_frames']}\n")

    lines.append("## Overall Metrics\n")
    lines.append(f"- Mean MPJPE: **{mean_mpjpe:.2f} mm**")
    lines.append(f"- Mean PA-MPJPE: **{mean_pa:.2f} mm**")
    lines.append(f"- Worst MPJPE: **{records_sorted[0].mpjpe:.2f} mm** (frame {records_sorted[0].frame_id})")
    lines.append(f"- 90th percentile MPJPE: **{np.percentile([r.mpjpe for r in records], 90):.2f} mm**\n")

    lines.append("## Category Thresholds\n")
    lines.append("| Category | Threshold |")
    lines.append("| --- | --- |")
    for k, v in metadata["thresholds"].items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")

    lines.append("## Failure-Mode Breakdown\n")
    lines.append("| Category | Count | % Frames | Mean MPJPE (mm) | Median MPJPE (mm) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for cat, count in cat_counts.most_common():
        errors = cat_mpjpe[cat]
        pct = 100.0 * count / len(records)
        mean_err = np.mean(errors)
        median_err = np.median(errors)
        lines.append(f"| {cat} | {count} | {pct:.1f}% | {mean_err:.2f} | {median_err:.2f} |")
    lines.append("")

    lines.append(f"## Top {top_k} Worst Samples\n")
    lines.append(
        "| Rank | Frame | MPJPE (mm) | PA-MPJPE (mm) | Active views | Missing ratio | "
        "Mean conf | Cam span (deg) | Root depth (m) | Root vel (m) | Reproj (px) | "
        "Depth unc | Ray quality | Geom loss | Categories |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for rank, r in enumerate(top_records, start=1):
        cats = ", ".join(r.categories) if r.categories else "none"
        lines.append(
            f"| {rank} | {r.frame_id} | {r.mpjpe:.2f} | {r.pa_mpjpe:.2f} | "
            f"{r.active_views} | {r.missing_ratio:.3f} | {r.mean_conf:.3f} | "
            f"{r.camera_span_deg:.1f} | {r.root_depth:.3f} | {r.root_velocity:.3f} | "
            f"{r.reproj_error:.2f} | {r.depth_uncertainty:.4f} | {r.ray_quality:.4f} | "
            f"{r.geom_loss:.4f} | {cats} |"
        )
    lines.append("")

    lines.append("## Methodology\n")
    lines.append(
        "Frames are evaluated independently with the v25 ``MultiViewGeometryFusionV25`` "
        "module. Failure categories are heuristics: generic ones (few_views, occlusion, "
        "low_confidence, large_baseline, far_subject, high_motion) plus v25-specific ones "
        "(high_reprojection, high_depth_uncertainty, low_ray_quality, outlier_view).\n"
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Failure-mode analysis for v25 Multi-View Geometry Fusion."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint (optional).")
    parser.add_argument("--dataset", type=str, default=None, help="Path to .npz validation dataset.")
    parser.add_argument("--synthetic", action="store_true", help="Generate and analyse a synthetic dataset.")
    parser.add_argument("--out_dir", type=str, default="outputs/v25_failure_analysis")
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_geometry_layers", type=int, default=2)
    parser.add_argument("--n_ray_samples", type=int, default=4)
    parser.add_argument("--use_geometry_attention", action="store_true", default=True)
    parser.add_argument("--no_geometry_attention", action="store_true")
    parser.add_argument("--use_learned_depth_triangulation", action="store_true", default=True)
    parser.add_argument("--no_learned_depth_triangulation", action="store_true")
    parser.add_argument("--use_geometry_bundle_adjustment", action="store_true", default=True)
    parser.add_argument("--no_geometry_bundle_adjustment", action="store_true")
    parser.add_argument("--state_dict_prefix", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        dataset_path = out_dir / "synthetic_val.npz"
        make_synthetic_npz(dataset_path)
    elif args.dataset:
        dataset_path = Path(args.dataset)
    else:
        parser.error("Provide --dataset or use --synthetic.")

    device = torch.device(args.device)

    use_geom_attn = args.use_geometry_attention and not args.no_geometry_attention
    use_depth = args.use_learned_depth_triangulation and not args.no_learned_depth_triangulation
    use_ba = args.use_geometry_bundle_adjustment and not args.no_geometry_bundle_adjustment

    records, metadata = run_v25_failure_analysis(
        checkpoint_path=args.checkpoint,
        dataset_path=str(dataset_path),
        out_dir=out_dir,
        d=args.d,
        n_heads=args.n_heads,
        n_geometry_layers=args.n_geometry_layers,
        n_ray_samples=args.n_ray_samples,
        use_geometry_attention=use_geom_attn,
        use_learned_depth_triangulation=use_depth,
        use_geometry_bundle_adjustment=use_ba,
        state_dict_prefix=args.state_dict_prefix,
        batch_size=args.batch_size,
        top_k=args.top_k,
        device=device,
    )
    metadata["checkpoint"] = args.checkpoint
    metadata["out_dir"] = str(out_dir)

    report = build_report(records, metadata, args.top_k)
    md_path = out_dir / "v25_failure_analysis_report.md"
    with open(md_path, "w") as f:
        f.write(report)

    json_path = out_dir / "v25_failure_analysis_records.json"
    with open(json_path, "w") as f:
        serialisable = []
        for r in records:
            d = r.__dict__.copy()
            d["categories"] = r.categories
            serialisable.append(d)
        json.dump(
            {
                "metadata": metadata,
                "records": serialisable,
            },
            f,
            indent=2,
        )

    print(f"Report written to {md_path}")
    print(f"JSON records written to {json_path}")
    print(f"Mean MPJPE: {np.mean([r.mpjpe for r in records]):.2f} mm")


if __name__ == "__main__":
    main()
