#!/usr/bin/env python3
"""Failure analysis tool for MotionFlow-MultiView checkpoints.

Loads a trained checkpoint and a validation ``.npz`` dataset, evaluates every
sample, ranks the worst ones by MPJPE, and categorises likely failure modes
(few views, occlusion, large baseline, far subject, high motion, low
confidence).  The report is written as Markdown to ``outputs/failure_analysis_report.md``
by default.

Example
-------
    python scripts/failure_analysis.py \
        --checkpoint outputs/omniview_fusion_v4_noskel_mpi.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --model_module motionflow_mv.fusion.omniview_fusion_v4 \
        --model_class OmniMultiViewFusionV4 \
        --model_kwargs_file scripts/failure_analysis_v4_noskel_kwargs.json \
        --dataset_name mpi \
        --batch_size 32
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import mpjpe as mpjpe_metric
from motionflow_mv.eval.metrics import pa_mpjpe as pa_mpjpe_metric


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        value = value.item() if value.size == 1 else float(value)
    return float(value)


def _parse_kwargs(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    # JSON booleans become Python bools; numbers are fine.
    return data


def _camera_angle_span(R: np.ndarray, active: np.ndarray) -> float:
    """Return the largest pairwise angle (degrees) among active camera centers.

    Camera centres are approximated by ``-R^T t`` and the angle is computed
    from the origin.
    """
    active = np.asarray(active, dtype=bool)
    if active.sum() < 2:
        return 0.0
    R_active = R[active]
    centers = np.einsum("vij,vj->vi", np.linalg.inv(R_active), -np.zeros((R_active.shape[0], 3)))
    # Actually -R^T t: R is world->cam, t = -R c => c = -R^T t
    centers = np.einsum("vji,vj->vi", R_active, np.zeros((R_active.shape[0], 3)))
    # Recompute correctly
    centers = []
    for Rv in R_active:
        # world->cam rotation Rv, translation tv not available; use Rv.T @ (-0)??
        # We only have R. For camera centre c = -R^T t, but t is not passed here.
        # Instead, use camera forward direction as proxy: forward = R[:, 2]
        centers.append(Rv[:, 2])
    centers = np.stack(centers, axis=0)
    centers = centers / (np.linalg.norm(centers, axis=-1, keepdims=True) + 1e-8)
    max_angle = 0.0
    for i in range(len(centers)):
        for j_ in range(i + 1, len(centers)):
            cos = np.clip(np.dot(centers[i], centers[j_]), -1.0, 1.0)
            angle = math.degrees(math.acos(cos))
            if angle > max_angle:
                max_angle = angle
    return max_angle


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    """Frame-level multi-view pose dataset."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.points_2d = data["points_2d"].astype(np.float32)  # (F, V, J, 2)
        self.confidences = data["confidences"].astype(np.float32)  # (F, V, J)
        self.joints_3d = data["joints_3d"].astype(np.float32)  # (F, J, 3)
        self.K = data["camera_K"].astype(np.float32)
        self.R = data["camera_R"].astype(np.float32)
        self.t = data["camera_t"].astype(np.float32)
        self.n_frames = self.points_2d.shape[0]

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int):
        x = np.concatenate(
            [self.points_2d[idx], self.confidences[idx][..., None]],
            axis=-1,
        )
        y = self.joints_3d[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(self.K),
            torch.from_numpy(self.R),
            torch.from_numpy(self.t),
            idx,
        )


def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    idx = torch.tensor([b[5] for b in batch], dtype=torch.long)
    return x, y, K, R, t, idx


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, model: nn.Module) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=False)
    return model


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
    categories: List[str] = field(default_factory=list)


def compute_thresholds(records: List[SampleRecord]) -> Dict[str, float]:
    """Return dataset-dependent thresholds for failure categories."""
    mpjpes = [r.mpjpe for r in records]
    mean_confs = [r.mean_conf for r in records]
    spans = [r.camera_span_deg for r in records]
    depths = [r.root_depth for r in records]
    velocities = [r.root_velocity for r in records]

    return {
        "few_views": 3,  # absolute: ≤3 active views
        "occlusion": 0.2,  # absolute: ≥20% missing keypoints
        "low_confidence": max(0.7, np.percentile(mean_confs, 20) - 1e-6),  # bottom 20% confidence
        "large_baseline": max(120.0, np.percentile(spans, 80)),  # wide arrangement
        "far_subject": np.percentile(depths, 80),  # top 20% depth
        "high_motion": np.percentile(velocities, 80),  # top 20% velocity
        "high_mpjpe": np.percentile(mpjpes, 80),  # top 20% errors (used as a tag)
    }


def categorise(
    active_views: int,
    missing_ratio: float,
    mean_conf: float,
    camera_span_deg: float,
    root_depth: float,
    root_velocity: float,
    thresholds: Dict[str, float],
) -> List[str]:
    cats: List[str] = []
    if active_views <= thresholds["few_views"]:
        cats.append("few_views")
    if missing_ratio >= thresholds["occlusion"]:
        cats.append("occlusion")
    if mean_conf <= thresholds["low_confidence"]:
        cats.append("low_confidence")
    if camera_span_deg >= thresholds["large_baseline"]:
        cats.append("large_baseline")
    if root_depth >= thresholds["far_subject"]:
        cats.append("far_subject")
    if root_velocity >= thresholds["high_motion"]:
        cats.append("high_motion")
    return cats


def compute_camera_span(R: torch.Tensor) -> float:
    """Largest pairwise angle between camera optical axes."""
    R_np = R.cpu().numpy()
    forwards = R_np[:, :, 2]
    norms = np.linalg.norm(forwards, axis=-1, keepdims=True)
    forwards = forwards / (norms + 1e-8)
    max_angle = 0.0
    for i in range(forwards.shape[0]):
        for j_ in range(i + 1, forwards.shape[0]):
            cos = np.clip(np.dot(forwards[i], forwards[j_]), -1.0, 1.0)
            angle = math.degrees(math.acos(cos))
            if angle > max_angle:
                max_angle = angle
    return max_angle


def run_failure_analysis(
    checkpoint_path: str,
    dataset_path: str,
    model_module: str,
    model_class: str,
    model_kwargs: Dict[str, Any],
    dataset_name: str,
    batch_size: int,
    top_k: int,
    device: torch.device,
) -> Tuple[List[SampleRecord], Dict[str, Any]]:
    # Load dataset.
    dataset = FrameDataset(dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Build model.
    mod = importlib.import_module(model_module)
    ModelCls = getattr(mod, model_class)
    model = ModelCls(**model_kwargs)
    if hasattr(model, "rebuild_graph"):
        graph_dataset = "mpiinf3dhp" if dataset_name.lower() in ("mpi", "mpiinf3dhp") else "h36m"
        # Infer number of joints from dataset.
        n_joints = dataset.joints_3d.shape[1]
        model.rebuild_graph(n_joints, dataset=graph_dataset)
    load_model(checkpoint_path, model)
    model.to(device)
    model.eval()

    all_records: List[SampleRecord] = []
    prev_root: Optional[np.ndarray] = None

    with torch.no_grad():
        for xb, yb, K, R, t, idxs in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)

            out = model(xb, K=K, R=R, t=t)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            pred_np = pred.cpu().numpy()
            gt_np = yb.cpu().numpy()
            confs = xb[..., 2].cpu().numpy()

            for i, frame_id in enumerate(idxs.tolist()):
                p = pred_np[i]
                g = gt_np[i]
                c = confs[i]
                mpjpe = mpjpe_metric(p * 1000.0, g * 1000.0)
                pa = pa_mpjpe_metric(p * 1000.0, g * 1000.0)

                missing = (c <= 0.0).sum()
                total = c.size
                missing_ratio = float(missing / total)
                mean_conf = float(c.mean())
                active_views = int((c.max(axis=-1) > 0.0).sum())
                camera_span = compute_camera_span(R[i])
                root = g[0]
                root_depth = float(root[2])
                velocity = 0.0
                if prev_root is not None:
                    velocity = float(np.linalg.norm(root - prev_root))
                # Approximate for the first frame by using the next frame later is fine;
                # we store it as 0.0.

                all_records.append(
                    SampleRecord(
                        frame_id=frame_id,
                        mpjpe=mpjpe,
                        pa_mpjpe=pa,
                        active_views=active_views,
                        missing_ratio=missing_ratio,
                        mean_conf=mean_conf,
                        camera_span_deg=camera_span,
                        root_depth=root_depth,
                        root_velocity=velocity,
                        categories=[],
                    )
                )
                prev_root = root

    # Compute dataset-dependent thresholds and recategorise samples.
    thresholds = compute_thresholds(all_records)
    for r in all_records:
        r.categories = categorise(
            r.active_views,
            r.missing_ratio,
            r.mean_conf,
            r.camera_span_deg,
            r.root_depth,
            r.root_velocity,
            thresholds,
        )

    return all_records, {
        "n_frames": len(dataset),
        "dataset_name": dataset_name,
        "thresholds": {k: float(v) for k, v in thresholds.items()},
    }


def build_report(records: List[SampleRecord], metadata: Dict[str, Any], top_k: int) -> str:
    records_sorted = sorted(records, key=lambda r: r.mpjpe, reverse=True)
    top_records = records_sorted[:top_k]
    mean_mpjpe = np.mean([r.mpjpe for r in records])
    mean_pa = np.mean([r.pa_mpjpe for r in records])

    # Category breakdown.
    cat_counts: Counter[str] = Counter()
    cat_mpjpe: Dict[str, List[float]] = {}
    for r in records:
        for cat in r.categories:
            cat_counts[cat] += 1
            cat_mpjpe.setdefault(cat, []).append(r.mpjpe)

    lines: List[str] = []
    lines.append("# Failure Analysis Report\n")
    lines.append(f"*Checkpoint:* `{metadata['checkpoint']}`  ")
    lines.append(f"*Dataset:* `{metadata['dataset']}`  ")
    lines.append(f"*Generated:* {datetime.utcnow().isoformat()}Z  ")
    lines.append(f"*Frames analysed:* {metadata['n_frames']}\n")

    lines.append("## Overall Metrics\n")
    lines.append(f"- Mean MPJPE: **{mean_mpjpe:.2f} mm**")
    lines.append(f"- Mean PA-MPJPE: **{mean_pa:.2f} mm**")
    lines.append(f"- Worst MPJPE: **{records_sorted[0].mpjpe:.2f} mm** (frame {records_sorted[0].frame_id})")
    lines.append(f"- 90th percentile MPJPE: **{np.percentile([r.mpjpe for r in records], 90):.2f} mm**\n")

    lines.append("## Category Thresholds\n")
    thresholds = metadata.get("thresholds", {})
    if thresholds:
        lines.append("| Category | Threshold |")
        lines.append("| --- | --- |")
        for k, v in thresholds.items():
            lines.append(f"| {k} | {v:.4f} |")
        lines.append("\n")

    lines.append("## Failure-Mode Breakdown\n")
    lines.append("| Category | Count | % Frames | Mean MPJPE (mm) | Median MPJPE (mm) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for cat, count in cat_counts.most_common():
        errors = cat_mpjpe[cat]
        pct = 100.0 * count / len(records)
        mean_err = np.mean(errors)
        median_err = np.median(errors)
        lines.append(f"| {cat} | {count} | {pct:.1f}% | {mean_err:.2f} | {median_err:.2f} |")
    lines.append("\n")

    lines.append(f"## Top {top_k} Worst Samples\n")
    lines.append("| Rank | Frame | MPJPE (mm) | PA-MPJPE (mm) | Active views | Missing ratio | Mean conf | Cam span (deg) | Root depth (m) | Root velocity (m) | Categories |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for rank, r in enumerate(top_records, start=1):
        cats = ", ".join(r.categories) if r.categories else "none"
        lines.append(
            f"| {rank} | {r.frame_id} | {r.mpjpe:.2f} | {r.pa_mpjpe:.2f} | "
            f"{r.active_views} | {r.missing_ratio:.3f} | {r.mean_conf:.3f} | "
            f"{r.camera_span_deg:.1f} | {r.root_depth:.3f} | {r.root_velocity:.3f} | {cats} |"
        )
    lines.append("\n")

    lines.append("## Methodology\n")
    lines.append(
        "Each frame is evaluated independently. Failure categories are heuristics: "
        "`few_views` (≤3 active views), `occlusion` (≥20% missing keypoints), "
        "`low_confidence` (bottom-20% mean confidence), "
        "`large_baseline` (wide camera arrangement), "
        "`far_subject` (top-20% root depth), `high_motion` (top-20% root velocity).\n"
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure analysis for MotionFlow-MultiView checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz validation dataset.")
    parser.add_argument("--model_module", type=str, default="motionflow_mv.fusion.omniview_fusion_v4")
    parser.add_argument("--model_class", type=str, default="OmniMultiViewFusionV4")
    parser.add_argument("--model_kwargs_file", type=str, default=None, help="JSON file of model kwargs.")
    parser.add_argument("--model_kwargs_json", type=str, default="{}", help="Inline JSON model kwargs.")
    parser.add_argument("--dataset_name", type=str, default="mpi", choices=["h36m", "mpi"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--out_md", type=str, default="outputs/failure_analysis_report.md")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.model_kwargs_file:
        model_kwargs = _parse_kwargs(args.model_kwargs_file)
    else:
        model_kwargs = json.loads(args.model_kwargs_json)

    records, metadata = run_failure_analysis(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        model_module=args.model_module,
        model_class=args.model_class,
        model_kwargs=model_kwargs,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        top_k=args.top_k,
        device=device,
    )
    metadata["checkpoint"] = args.checkpoint
    metadata["dataset"] = args.dataset

    report = build_report(records, metadata, args.top_k)
    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Report written to {out_path}")
    print(f"Mean MPJPE: {np.mean([r.mpjpe for r in records]):.2f} mm")


if __name__ == "__main__":
    main()
