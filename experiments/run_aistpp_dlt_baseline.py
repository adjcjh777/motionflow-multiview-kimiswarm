"""DLT baseline on the AIST++ validation split.

Reads the canonical AIST++ train/val split from
``configs/splits/webbridge_aistpp_train_val.yaml``, triangulates the 2D
keypoints with confidence-weighted DLT, and writes a Markdown report.

The script is CPU-only and does not touch the GPU, so it can run while
another training job is active on the local RTX 4090.

Usage
-----
    python experiments/run_aistpp_dlt_baseline.py

Outputs
-------
    outputs/aistpp_dlt_baseline.json
    docs/results_aistpp_dlt_baseline.md
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Windows/conda OpenMP runtime fix required by NumPy before any numpy import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Make this script runnable from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

# experiments/baselines.py is self-contained and NumPy-only.
sys.path.insert(0, str(Path(__file__).parent))
from baselines import (
    build_projection_matrices,
    reprojection_error,
)
from motionflow_mv.eval.metrics import pa_mpjpe
import numpy as np
import torch as _torch


def load_val_paths(split_path: Path) -> list[str]:
    with open(split_path) as f:
        split = yaml.safe_load(f)
    return list(split.get("val", []))


def evaluate_npz(path: Path) -> dict:
    data = np.load(path)
    points_2d = np.asarray(data["points_2d"], dtype=np.float64)
    confidences = np.asarray(data["confidences"], dtype=np.float64)
    joints_3d = np.asarray(data["joints_3d"], dtype=np.float64)
    K = np.asarray(data["camera_K"], dtype=np.float64)
    R = np.asarray(data["camera_R"], dtype=np.float64)
    t = np.asarray(data["camera_t"], dtype=np.float64)

    P = build_projection_matrices(K, R, t)
    pred = dlt_from_views(points_2d, confidences, P)
    # Replace any non-finite values so metric backends do not fail.
    pred = np.where(np.isfinite(pred), pred, 0.0)
    joints_3d = np.where(np.isfinite(joints_3d), joints_3d, 0.0)

    e = np.linalg.norm(pred - joints_3d, axis=-1)
    reproj = reprojection_error(pred, points_2d, P, confidences)
    metrics = {
        "mpjpe": float(e.mean()),
        "pa_mpjpe": pa_mpjpe(pred, joints_3d),
        "reproj_px": float(np.nanmean(reproj)),
    }
    return {"shape": {"T": points_2d.shape[0], "V": points_2d.shape[1], "J": points_2d.shape[2]}, "metrics": metrics}


def dlt_from_views(points_2d: np.ndarray, confidences: np.ndarray,
                   P: np.ndarray) -> np.ndarray:
    """Confidence-weighted DLT using batched torch SVD."""
    T, V, J, _ = points_2d.shape

    # Build per-sample weights from confidences: (T, V, J).
    conf = _torch.from_numpy(confidences).permute(0, 2, 1)  # (T, J, V)
    zero_frames = conf.sum(dim=-1, keepdim=True) == 0
    conf = conf.clone()
    conf[zero_frames.expand_as(conf)] = 1.0

    p2d = _torch.from_numpy(points_2d).permute(0, 2, 1, 3)  # (T, J, V, 2)
    P_t = _torch.from_numpy(P)  # (V, 3, 4)

    # Mask out NaN 2D observations so DLT uses only valid views.
    nan_mask = _torch.isnan(p2d).any(dim=-1)  # (T, J, V)
    p2d = _torch.where(_torch.isnan(p2d), _torch.zeros_like(p2d), p2d)
    conf = conf.clone()
    conf[nan_mask] = 0.0

    # Build DLT rows: for each view, x*P2 - P0 and y*P2 - P1.
    x = p2d[..., 0]  # (T, J, V)
    y = p2d[..., 1]
    P0 = P_t[:, 0, :].unsqueeze(0).unsqueeze(0)  # (1, 1, V, 4)
    P1 = P_t[:, 1, :].unsqueeze(0).unsqueeze(0)
    P2 = P_t[:, 2, :].unsqueeze(0).unsqueeze(0)
    row_x = x.unsqueeze(-1) * P2 - P0  # (T, J, V, 4)
    row_y = y.unsqueeze(-1) * P2 - P1

    # Stack x/y rows and apply confidence weights.
    A = _torch.stack([row_x, row_y], dim=-2)  # (T, J, V, 2, 4)
    w_sqrt = conf.unsqueeze(-1).unsqueeze(-1).sqrt()  # (T, J, V, 1, 1)
    A = A * w_sqrt
    A = A.reshape(T, J, 2 * V, 4)

    # Solve via SVD of the homogeneous system.
    _, _, Vh = _torch.linalg.svd(A)
    X = Vh[..., -1, :]  # (T, J, 4)
    pred = X[..., :3] / X[..., 3:4]
    return pred.numpy()


def weighted_aggregate(records: list[dict]) -> dict:
    total_frames = sum(r["shape"]["T"] for r in records)
    if total_frames == 0:
        return {}

    def weighted(key: str) -> float:
        return sum(r["metrics"][key] * r["shape"]["T"] for r in records) / total_frames

    return {
        "mpjpe_m": weighted("mpjpe"),
        "pa_mpjpe_m": weighted("pa_mpjpe"),
        "reproj_px": weighted("reproj_px"),
        "total_frames": total_frames,
        "n_clips": len(records),
    }


def genre_of(name: str) -> str:
    # e.g. "gWA_sBM_cAll_d25_mWA1_ch03_multiview.npz" -> "gWA"
    return name.split("_")[0]


def style_of(name: str) -> str:
    # e.g. "gWA_sBM_..." -> "sBM"
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def main() -> None:
    split_path = Path("configs/splits/webbridge_aistpp_train_val.yaml")
    out_json = Path("outputs/aistpp_dlt_baseline.json")
    out_md = Path("docs/results_aistpp_dlt_baseline.md")

    val_paths = load_val_paths(split_path)
    if not val_paths:
        raise RuntimeError(f"No validation paths found in {split_path}")

    records = []
    missing = []
    for p in val_paths:
        npz = Path(p)
        if not npz.exists():
            missing.append(str(npz))
            continue
        rec = evaluate_npz(npz)
        rec["path"] = str(npz)
        rec["clip"] = npz.name
        records.append(rec)

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} validation .npz files:\n" + "\n".join(missing[:10]))

    overall = weighted_aggregate(records)

    # Per-genre and per-style breakdowns.
    genre_groups = defaultdict(list)
    style_groups = defaultdict(list)
    for r in records:
        genre_groups[genre_of(r["clip"])].append(r)
        style_groups[style_of(r["clip"])].append(r)

    genre_agg = {g: weighted_aggregate(recs) for g, recs in sorted(genre_groups.items())}
    style_agg = {s: weighted_aggregate(recs) for s, recs in sorted(style_groups.items())}

    # Save raw JSON.
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "overall": overall,
                "per_genre": genre_agg,
                "per_style": style_agg,
                "per_clip": records,
            },
            f,
            indent=2,
        )

    # Write Markdown report.
    def row(label: str, agg: dict) -> str:
        return (
            f"| {label} | {agg['n_clips']} | {agg['total_frames']} | "
            f"{agg['mpjpe_m']*1000:.2f} | {agg['pa_mpjpe_m']*1000:.2f} | {agg['reproj_px']:.2f} |\n"
        )

    md = []
    md.append("# AIST++ Validation – DLT Baseline\n")
    md.append("\n")
    md.append("*Report generated by `experiments/run_aistpp_dlt_baseline.py`*\n")
    md.append("\n")
    md.append("## Overall Results\n")
    md.append("\n")
    md.append("| Metric | Value |\n")
    md.append("|---|---|\n")
    md.append(f"| Clips | {overall['n_clips']} |\n")
    md.append(f"| Total frames | {overall['total_frames']} |\n")
    md.append(f"| MPJPE (mm) | {overall['mpjpe_m']*1000:.2f} |\n")
    md.append(f"| PA-MPJPE (mm) | {overall['pa_mpjpe_m']*1000:.2f} |\n")
    md.append(f"| Reprojection error (px) | {overall['reproj_px']:.2f} |\n")
    md.append("\n")
    md.append("### Notes\n")
    md.append("\n")
    md.append("- MPJPE/PA-MPJPE are reported in millimetres.\n")
    md.append("- AIST++ canonical files are scaled to metres (`scale_factor=0.01`).\n")
    md.append("- DLT is confidence-weighted per joint and per view.\n")
    md.append("- Aggregates are weighted by clip length in frames.\n")
    md.append("\n")
    md.append("## Per-Genre Breakdown\n")
    md.append("\n")
    md.append("| Genre | Clips | Frames | MPJPE (mm) | PA-MPJPE (mm) | Reproj (px) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for g, agg in genre_agg.items():
        md.append(row(g, agg))
    md.append("\n")
    md.append("## Per-Style (BM / FM) Breakdown\n")
    md.append("\n")
    md.append("| Style | Clips | Frames | MPJPE (mm) | PA-MPJPE (mm) | Reproj (px) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for s, agg in style_agg.items():
        md.append(row(s, agg))
    md.append("\n")
    md.append("## Configuration\n")
    md.append("\n")
    md.append(f"- Split file: `{split_path}`\n")
    md.append(f"- Number of validation clips: {len(val_paths)}\n")
    md.append(f"- Raw JSON: `{out_json}`\n")
    md.append("\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(md)

    print(f"Overall MPJPE: {overall['mpjpe_m']*1000:.2f} mm")
    print(f"Overall PA-MPJPE: {overall['pa_mpjpe_m']*1000:.2f} mm")
    print(f"Results saved to {out_json} and {out_md}")


if __name__ == "__main__":
    main()
