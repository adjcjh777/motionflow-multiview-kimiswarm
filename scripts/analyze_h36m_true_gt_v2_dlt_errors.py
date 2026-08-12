"""Detailed error analysis of the H36M true-GT v2 DLT baseline.

Loads the confidence-weighted DLT result JSON and the underlying v2 test
``.npz`` files (S9, S11), then computes per-joint MPJPE, per-camera
reprojection error, per-frame MPJPE, and confidence-error correlations.

Usage
-----
    python scripts/analyze_h36m_true_gt_v2_dlt_errors.py
    python scripts/analyze_h36m_true_gt_v2_dlt_errors.py --device cuda

Output
------
    outputs/h36m_true_gt_v2/dlt_error_analysis/*.png
    outputs/h36m_true_gt_v2/dlt_error_analysis/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Use a non-interactive matplotlib backend so the script can run headless.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq

def _subject_label(name: str) -> str:
    """Map 's_09_...' / 's_11_...' names to 'S9' / 'S11'."""
    try:
        return f"S{int(name.split('_')[1])}"
    except Exception:
        return name


CANONICAL_17_JOINTS = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "head_top",
]


def build_projection_matrices(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return (V, 3, 4) projection matrices P = K [R | t]."""
    V = K.shape[0]
    P = np.zeros((V, 3, 4), dtype=np.float64)
    for v in range(V):
        Rt = np.concatenate([R[v], t[v].reshape(3, 1)], axis=1)
        P[v] = K[v] @ Rt
    return P


def triangulate_file(path: Path, P: np.ndarray, device: str, chunk_size: int = 8192) -> np.ndarray:
    """Return (T, J, 3) DLT triangulation in metres for a canonical .npz file."""
    data = np.load(path)
    points_2d = np.asarray(data["points_2d"], dtype=np.float64)
    confidences = np.asarray(data["confidences"], dtype=np.float64)

    T, V, J, _ = points_2d.shape
    P_t = torch.from_numpy(P).to(device=device, dtype=torch.float64)

    pred_chunks = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        p2d_chunk = torch.from_numpy(points_2d[start:end]).to(device=device, dtype=torch.float64)
        w_chunk = torch.from_numpy(confidences[start:end]).to(device=device, dtype=torch.float64)
        X_t = triangulate_dlt_batched_lstsq(p2d_chunk, P_t, weights=w_chunk)
        pred_chunks.append(X_t.detach().cpu().numpy())

    return np.concatenate(pred_chunks, axis=0)


def compute_per_camera_reproj_error(
    pred_3d: np.ndarray,
    points_2d: np.ndarray,
    confidences: np.ndarray,
    P: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (per_view_weighted_px, per_view_unweighted_px, per_obs_error, per_obs_conf).

    Shapes:
        per_view_* : (V,)
        per_obs_error / per_obs_conf : (V, T*J)
    """
    T, V, J, _ = points_2d.shape
    # Homogeneous 3D points.
    Xh = np.concatenate([pred_3d, np.ones((T, J, 1), dtype=np.float64)], axis=-1)  # (T,J,4)
    # Project to every view: (V, T*J, 3)
    Xh_flat = Xh.reshape(T * J, 4)  # (N, 4)
    proj = np.matmul(P, Xh_flat.T[None, ...])  # (V, 3, N)
    proj = np.transpose(proj, (0, 2, 1))  # (V, N, 3)
    proj_2d = proj[..., :2] / proj[..., 2:3]  # (V, T*J, 2)

    # points_2d (T,V,J,2) -> flatten each view to (T*J, 2) in frame-major, joint-minor order.
    p2d_flat = points_2d.transpose(1, 0, 2, 3).reshape(V, T * J, 2)
    diffs = np.linalg.norm(proj_2d - p2d_flat, axis=-1)  # (V, T*J)

    # Per-view confidence in the same (V, T*J) layout.
    conf_per_view = np.stack([confidences[:, v, :].reshape(-1) for v in range(V)], axis=0)  # (V, T*J)

    per_view_weighted = np.sum(conf_per_view * diffs, axis=1) / np.maximum(np.sum(conf_per_view, axis=1), 1e-8)
    per_view_unweighted = np.mean(diffs, axis=1)

    return per_view_weighted, per_view_unweighted, diffs.ravel(), conf_per_view.ravel()


def weighted_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, ignoring NaNs."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def smooth_line(x: np.ndarray, window: int = 101) -> np.ndarray:
    """1D boxcar smoothing with reflection padding; preserves input length."""
    if x.size < window:
        return x
    pad = window // 2
    padded = np.pad(x, pad, mode="reflect")
    cum = np.cumsum(np.concatenate(([0.0], padded)))
    return (cum[window:] - cum[:-window]) / window


def analyze_subject(path: Path, P: np.ndarray, device: str) -> dict:
    """Compute all per-subject diagnostics."""
    print(f"  Analyzing {path.name} ...", flush=True)
    data = np.load(path)
    points_2d = np.asarray(data["points_2d"], dtype=np.float64)
    confidences = np.asarray(data["confidences"], dtype=np.float64)
    joints_3d = np.asarray(data["joints_3d"], dtype=np.float64)

    pred_3d = triangulate_file(path, P, device)
    T, V, J, _ = points_2d.shape

    # 3D errors (metres -> mm).
    diff_3d = np.linalg.norm(pred_3d - joints_3d, axis=-1) * 1000.0  # (T, J)
    per_joint_mpjpe = diff_3d.mean(axis=0)  # (J,)
    per_frame_mpjpe = diff_3d.mean(axis=1)  # (T,)

    # Reprojection errors.
    per_view_weighted, per_view_unweighted, per_obs_error, per_obs_conf = compute_per_camera_reproj_error(
        pred_3d, points_2d, confidences, P
    )

    # Confidence-error correlations.
    frame_conf = confidences.reshape(T, -1).mean(axis=1)  # (T,)
    frame_err = per_frame_mpjpe
    frame_corr = weighted_pearson(frame_conf, frame_err)

    obs_corr = weighted_pearson(per_obs_conf, per_obs_error)

    return {
        "name": path.stem,
        "T": int(T),
        "V": int(V),
        "J": int(J),
        "per_joint_mpjpe": per_joint_mpjpe,  # (J,)
        "per_frame_mpjpe": per_frame_mpjpe,  # (T,)
        "per_view_weighted_reproj_px": per_view_weighted,  # (V,)
        "per_view_unweighted_reproj_px": per_view_unweighted,  # (V,)
        "per_obs_error": per_obs_error,
        "per_obs_conf": per_obs_conf,
        "frame_conf": frame_conf,
        "frame_err": frame_err,
        "frame_conf_corr": frame_corr,
        "obs_conf_corr": obs_corr,
        "mean_mpjpe_mm": float(per_frame_mpjpe.mean()),
        "mean_weighted_reproj_px": float(per_view_weighted.mean()),
        "mean_unweighted_reproj_px": float(per_view_unweighted.mean()),
    }


def plot_per_joint_mpjpe(results: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(17)
    width = 0.25
    names = [_subject_label(r["name"]) for r in results]
    for i, r in enumerate(results):
        ax.bar(x + (i - len(results) / 2 + 0.5) * width, r["per_joint_mpjpe"], width, label=names[i])
    # combined
    if len(results) == 2:
        combined = (
            results[0]["per_joint_mpjpe"] * results[0]["T"] + results[1]["per_joint_mpjpe"] * results[1]["T"]
        ) / (results[0]["T"] + results[1]["T"])
        ax.bar(x + width, combined, width, label="combined")
    ax.set_xticks(x)
    ax.set_xticklabels(CANONICAL_17_JOINTS, rotation=60, ha="right")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Per-joint DLT baseline MPJPE — H36M true-GT v2 test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_joint_mpjpe.png", dpi=200)
    plt.close(fig)


def plot_per_camera_reproj(results: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(results[0]["V"])
    width = 0.25
    names = [_subject_label(r["name"]) for r in results]
    for i, r in enumerate(results):
        ax.bar(x + (i - len(results) / 2 + 0.5) * width, r["per_view_weighted_reproj_px"], width, label=names[i])
    if len(results) == 2:
        combined = (
            results[0]["per_view_weighted_reproj_px"] * results[0]["T"]
            + results[1]["per_view_weighted_reproj_px"] * results[1]["T"]
        ) / (results[0]["T"] + results[1]["T"])
        ax.bar(x + width, combined, width, label="combined")
    ax.set_xticks(x)
    ax.set_xticklabels([f"cam {v}" for v in x])
    ax.set_ylabel("Reprojection error (pixels)")
    ax.set_title("Per-camera reprojection error (confidence-weighted) — DLT v2 baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_camera_reproj_error.png", dpi=200)
    plt.close(fig)


def plot_per_frame_mpjpe(results: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for r in results:
        t = np.arange(r["T"])
        label = _subject_label(r["name"])
        if r["T"] > 1000:
            stride = max(1, r["T"] // 2000)
            ax.plot(t[::stride], r["per_frame_mpjpe"][::stride], alpha=0.6, label=f"{label} (subsampled)")
        else:
            ax.plot(t, r["per_frame_mpjpe"], alpha=0.6, label=label)
        # Smoothed trend.
        smooth = smooth_line(r["per_frame_mpjpe"], window=501)
        ax.plot(t, smooth, lw=2, label=f"{label} smoothed")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Per-frame MPJPE over time — DLT v2 baseline")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_frame_mpjpe.png", dpi=200)
    plt.close(fig)


def plot_confidence_vs_error(results: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["tab:blue", "tab:orange"]
    for i, r in enumerate(results):
        ax.scatter(r["frame_conf"][::100], r["frame_err"][::100], s=4, alpha=0.4, color=colors[i % 2], label=_subject_label(r["name"]))
    ax.set_xlabel("Mean per-frame confidence")
    ax.set_ylabel("Per-frame MPJPE (mm)")
    ax.set_title("Confidence vs. 3D error (frame-level; subsampled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_vs_error.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze H36M true-GT v2 DLT baseline errors.")
    parser.add_argument("--json", type=str, default="outputs/h36m_true_gt_v2/dlt_baseline_h36m_true_gt_v2.json")
    parser.add_argument("--out_dir", type=str, default="outputs/h36m_true_gt_v2/dlt_error_analysis")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device (cpu or cuda).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    json_path = repo_root / args.json
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as fh:
        dlt_summary = json.load(fh)

    test_files = [e for e in dlt_summary["confidence_weighted"]["per_file"] if e["split"] == "test"]
    print(f"Loaded {len(test_files)} test files from {json_path}")

    results = []
    for entry in test_files:
        path = Path(entry["path"])
        data = np.load(path)
        K = np.asarray(data["camera_K"], dtype=np.float64)
        R = np.asarray(data["camera_R"], dtype=np.float64)
        t = np.asarray(data["camera_t"], dtype=np.float64)
        P = build_projection_matrices(K, R, t)
        results.append(analyze_subject(path, P, args.device))

    # Combined statistics.
    total_T = sum(r["T"] for r in results)
    combined_per_joint = sum(r["per_joint_mpjpe"] * r["T"] for r in results) / total_T
    combined_per_view = sum(r["per_view_weighted_reproj_px"] * r["T"] for r in results) / total_T

    print("\n=== Per-joint MPJPE (mm) ===")
    for j, name in enumerate(CANONICAL_17_JOINTS):
        line = f"  {name:<15}"
        for r in results:
            line += f" {r['name'].split('_')[0].upper()}={r['per_joint_mpjpe'][j]:.2f}"
        line += f" combined={combined_per_joint[j]:.2f}"
        print(line)

    print("\n=== Per-camera reprojection error (px) ===")
    for v in range(results[0]["V"]):
        line = f"  cam {v}"
        for r in results:
            line += f" {r['name'].split('_')[0].upper()}={r['per_view_weighted_reproj_px'][v]:.2f}"
        line += f" combined={combined_per_view[v]:.2f}"
        print(line)

    print("\n=== Subject-level MPJPE (mm) ===")
    for r in results:
        print(f"  {r['name']}: mean={r['mean_mpjpe_mm']:.2f}, weighted_reproj={r['mean_weighted_reproj_px']:.2f}px")
    print(f"  combined mean MPJPE: {sum(r['mean_mpjpe_mm'] * r['T'] for r in results) / total_T:.2f} mm")

    print("\n=== Confidence vs. error correlation ===")
    for r in results:
        print(f"  {r['name']}: frame-level r={r['frame_conf_corr']:.3f}, obs-level r={r['obs_conf_corr']:.3f}")

    # Plots.
    plot_per_joint_mpjpe(results, out_dir)
    plot_per_camera_reproj(results, out_dir)
    plot_per_frame_mpjpe(results, out_dir)
    plot_confidence_vs_error(results, out_dir)

    # Summary JSON.
    summary = {
        "protocol": "S1,S5,S6,S7,S8 train -> S9,S11 test",
        "unit": "mm",
        "per_joint_mpjpe": {
            "joints": CANONICAL_17_JOINTS,
            "combined": combined_per_joint.tolist(),
        },
        "per_camera_reproj_px": {
            "combined": combined_per_view.tolist(),
        },
        "subjects": [],
    }
    for r in results:
        summary["subjects"].append(
            {
                "name": r["name"],
                "T": r["T"],
                "mean_mpjpe_mm": r["mean_mpjpe_mm"],
                "per_joint_mpjpe": r["per_joint_mpjpe"].tolist(),
                "per_view_weighted_reproj_px": r["per_view_weighted_reproj_px"].tolist(),
                "frame_conf_corr": r["frame_conf_corr"],
                "obs_conf_corr": r["obs_conf_corr"],
            }
        )

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
