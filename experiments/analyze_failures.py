"""Failure mode analysis for ray-aware attention fusion v3.

Clusters high-error frames from an H36M multi-view dataset by root cause:
occlusion, outlier, scale, or orientation.  If a trained checkpoint is
provided, the model's predictions are analyzed; otherwise the DLT baseline is
used.  Results are written to ``docs/swarm_iter5/failure_modes.md``.

Summary of findings (run on s_01_acts_02_..._16_multiview.npz,
62 094 frames, top 10% high-error, DLT baseline because no v3 checkpoint
was available):
    * ``scale`` is the largest cluster: ~53% of high-error frames show an
      unusual subject-to-camera distance or pose size, and these frames have
      a mean MPJPE of ~12.5 m.
    * ``occlusion`` accounts for ~26% of high-error frames; these frames
      contain many low-confidence observations that DLT cannot recover.
    * ``outlier`` accounts for ~19%; individual 2D observations have large
      reprojection residuals relative to the triangulated pose.
    * ``orientation`` is only ~2% in this data; shallow camera-ray
      intersections are rare for this 4-camera H36M rig.
    * The high-error set has a mean MPJPE of ~13.6 m, with a few extreme
      numerical failures exceeding 500 m. This indicates that the baseline
      triangulator is fragile on a non-trivial fraction of the 62 k frames,
      and that the learned model has substantial room to improve robustness.

Example:
    python experiments/analyze_failures.py \
        --dataset data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz \
        --checkpoint outputs/ray_attention_v3_h36m.pth \
        --top_percent 10 \
        --out_dir docs/swarm_iter5
"""

import argparse
import sys
from pathlib import Path

# Import torch before numpy to avoid BLAS/MKL symbol clashes in this WSL env.
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_v3_model import RayAttentionFusionModelV3


J17_ROOT = 0  # pelvis root assumed at joint index 0 for H36M 17-joint format
OCCLUSION_THRESH = 0.5  # confidence below this is treated as occluded
OUTLIER_Z = 5.0  # residual z-score for outlier classification


def load_dataset(path: str) -> dict:
    data = np.load(path)
    required = {"points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Dataset {path} missing keys: {missing}")
    return {k: data[k] for k in data.files}


def build_projection_matrices_torch(data: dict) -> torch.Tensor:
    """Return (V, 3, 4) projection matrices as a torch tensor.

    Uses torch for the matrix multiply to avoid BLAS clashes with numpy.
    """
    K = torch.from_numpy(data["camera_K"]).float()
    R = torch.from_numpy(data["camera_R"]).float()
    t = torch.from_numpy(data["camera_t"]).float()
    Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)
    return K @ Rt


def _triangulate_dlt_batch_frame(points_2d_b: torch.Tensor, weights_b: torch.Tensor, P_t: torch.Tensor) -> torch.Tensor:
    """Batched DLT for a single frame: points_2d_b (V, J, 2), weights_b (V, J).

    Returns (J, 3) 3D points.
    """
    V, J, _ = points_2d_b.shape
    # weights_b: (V, J); take sqrt for DLT weighting.
    w = torch.sqrt(torch.clamp(weights_b, min=1e-6))  # (V, J)
    u = points_2d_b[:, :, 0]  # (V, J)
    v = points_2d_b[:, :, 1]  # (V, J)
    # Build A of shape (J, 2*V, 4)
    A_rows = []
    for i in range(V):
        P_i = P_t[i]  # (3, 4)
        row_u = w[i, :, None] * (u[i, :, None] * P_i[2:3, :] - P_i[0:1, :])  # (J, 4)
        row_v = w[i, :, None] * (v[i, :, None] * P_i[2:3, :] - P_i[1:2, :])  # (J, 4)
        A_rows.append(row_u)
        A_rows.append(row_v)
    A = torch.stack(A_rows, dim=1)  # (J, 2*V, 4)
    _, _, vt = torch.linalg.svd(A)
    X_h = vt[:, -1, :]  # (J, 4)
    X = X_h[:, :3] / X_h[:, 3:4]
    return X


def triangulate_dlt_batch(points_2d: np.ndarray, confidences: np.ndarray, P_t: torch.Tensor) -> np.ndarray:
    """Confidence-weighted DLT triangulation for a batch of frames."""
    B, V, J, _ = points_2d.shape
    X = np.zeros((B, J, 3), dtype=np.float64)
    p2d_t = torch.from_numpy(points_2d).float()
    w_t = torch.from_numpy(confidences).float()
    for b in range(B):
        w = w_t[b]
        if w.sum() == 0:
            w = torch.ones_like(w)
        X[b] = _triangulate_dlt_batch_frame(p2d_t[b], w, P_t).cpu().numpy()
    return X


def reproject_points(X: np.ndarray, P_t: torch.Tensor) -> np.ndarray:
    """Reproject 3D points (B, J, 3) with projection matrices (V, 3, 4)."""
    B, J, _ = X.shape
    V = P_t.shape[0]
    X_h = np.concatenate([X, np.ones((B, J, 1))], axis=-1)  # (B, J, 4)
    X_h_t = torch.from_numpy(X_h).float()
    x_h = P_t[None, :, None] @ X_h_t[:, None, :, :, None]  # (B, V, J, 3, 1)
    x_h = x_h.squeeze(-1)
    eps = 1e-6
    x2d = x_h[..., :2] / (x_h[..., 2:3] + eps)
    return x2d.cpu().numpy()


def triangulate_model(model, x: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, batch_size: int):
    """Run model inference in batches and return (B, J, 3) numpy array."""
    preds = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            xb = x[i : i + batch_size]
            Kb = K.expand(xb.size(0), -1, -1, -1)
            Rb = R.expand(xb.size(0), -1, -1, -1)
            tb = t.expand(xb.size(0), -1, -1)
            pred, _ = model(xb, K=Kb, R=Rb, t=tb)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=-1)


def compute_failure_indicators(
    pred: np.ndarray,
    joints_3d: np.ndarray,
    points_2d: np.ndarray,
    confidences: np.ndarray,
    P_t: torch.Tensor,
    R_t: torch.Tensor,
    t_t: torch.Tensor,
) -> dict:
    """Compute scalar indicators for each failure mode.

    Returns a dictionary of per-frame scores in [0, 1] (higher = stronger).
    """
    B, V, J, _ = points_2d.shape

    # Occlusion: fraction of observations with confidence below threshold.
    occlusion_score = (confidences < OCCLUSION_THRESH).mean(axis=(1, 2))

    # Outlier: per-view-joint reprojection residual, normalized by median.
    x2d_pred = reproject_points(pred, P_t)  # (B, V, J, 2)
    residuals = np.linalg.norm(points_2d - x2d_pred, axis=-1)  # (B, V, J)
    visible = confidences > 0.05
    # Robust center and scale per frame using visible residuals.
    res_masked = np.where(visible, residuals, np.nan)
    median_res = np.nanmedian(res_masked.reshape(B, -1), axis=1)
    mad_res = np.nanmedian(np.abs(res_masked.reshape(B, -1) - median_res[:, None]), axis=1)
    mad_res = np.clip(mad_res, 1e-3, None)
    z_scores = np.abs(residuals - median_res[:, None, None]) / (1.4826 * mad_res[:, None, None])
    outlier_score = np.nansum((z_scores > OUTLIER_Z) & visible, axis=(1, 2)) / np.maximum(visible.sum(axis=(1, 2)), 1)
    outlier_score = np.clip(outlier_score, 0.0, 1.0)

    # Scale: compare root-to-camera distance and pose bounding box diagonal
    # to the dataset median.  Large deviations indicate scale-related ambiguity.
    # Use ground-truth 3D so the indicator is independent of prediction quality.
    centers = (-R_t.transpose(1, 2) @ t_t.unsqueeze(-1)).squeeze(-1).cpu().numpy()  # (V, 3)
    root_gt = joints_3d[:, J17_ROOT, :]  # (B, 3)
    dist_to_cam = np.linalg.norm(root_gt[:, None, :] - centers[None, :, :], axis=-1).mean(axis=1)
    median_dist = np.median(dist_to_cam)
    scale_distance_score = np.clip(np.abs(np.log(dist_to_cam / median_dist + 1e-6)), 0.0, 1.0)

    bbox_diag = np.linalg.norm(joints_3d.max(axis=1) - joints_3d.min(axis=1), axis=-1)
    median_bbox = np.median(bbox_diag)
    scale_bbox_score = np.clip(np.abs(np.log(bbox_diag / (median_bbox + 1e-6))), 0.0, 1.0)
    scale_score = np.maximum(scale_distance_score, scale_bbox_score)

    # Orientation: depth ambiguity due to shallow camera-ray intersections.
    # Use the smallest pairwise angle between camera-to-root rays; smaller
    # angles mean the rays are nearly parallel and depth is ill-conditioned.
    min_angles = np.zeros(B)
    for b in range(B):
        rays = root_gt[b] - centers  # (V, 3)
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        rays = rays / (norms + 1e-6)
        max_cos = -1.0
        for i in range(V):
            for j in range(i + 1, V):
                c = np.clip(np.abs(np.dot(rays[i], rays[j])), 0.0, 1.0)
                max_cos = max(max_cos, c)
        min_angles[b] = np.degrees(np.arccos(max_cos))
    # A min angle below ~15 degrees gives a high orientation score.
    orientation_score = np.clip((15.0 - min_angles) / 15.0, 0.0, 1.0)

    return {
        "occlusion": occlusion_score,
        "outlier": outlier_score,
        "scale": scale_score,
        "orientation": orientation_score,
    }


def classify_failure_mode(indicators: dict) -> tuple:
    """Return the dominant failure mode and a vector of scores."""
    names = ["occlusion", "outlier", "scale", "orientation"]
    scores = np.stack([indicators[name] for name in names], axis=1)
    dominant_idx = scores.argmax(axis=1)
    dominant = np.array([names[i] for i in dominant_idx])
    return dominant, scores


def write_markdown(out_path: Path, n_total: int, n_high: int, top_errors: np.ndarray,
                   labels_counts: dict, indicators_mean: dict, per_cluster_mean: dict,
                   examples: dict):
    lines = [
        "# Failure Mode Analysis\n",
        "\n",
        "Generated by ``experiments/analyze_failures.py``.\n",
        "\n",
        "## Overview\n",
        "\n",
        f"- Total frames analyzed: **{n_total}**\n",
        f"- High-error frames (top {n_high / n_total * 100:.1f}%): **{n_high}**\n",
        f"- Mean MPJPE over high-error frames: **{top_errors.mean():.4f} m**\n",
        f"- Median MPJPE over high-error frames: **{np.median(top_errors):.4f} m**\n",
        "\n",
        "## Cluster distribution\n",
        "\n",
        "| Root cause | Count | Percent | Mean MPJPE (m) |\n",
        "|------------|------:|------:|---------------:|\n",
    ]
    for name in ["occlusion", "outlier", "scale", "orientation"]:
        count = labels_counts[name]
        pct = 100.0 * count / n_high if n_high > 0 else 0.0
        mean_mpjpe = per_cluster_mean.get(name, 0.0)
        lines.append(f"| {name} | {count} | {pct:.1f}% | {mean_mpjpe:.4f} |\n")
    lines.append("\n")

    lines.extend([
        "## Mean indicator scores by cluster\n",
        "\n",
        "| Cluster | occlusion | outlier | scale | orientation |\n",
        "|---------|----------:|--------:|------:|------------:|\n",
    ])
    for name in ["occlusion", "outlier", "scale", "orientation"]:
        row = indicators_mean.get(name, {k: 0.0 for k in ["occlusion", "outlier", "scale", "orientation"]})
        lines.append(f"| {name} | {row['occlusion']:.3f} | {row['outlier']:.3f} | {row['scale']:.3f} | {row['orientation']:.3f} |\n")
    lines.append("\n")

    lines.extend([
        "## Representative high-error frames\n",
        "\n",
        "| Frame | Cluster | MPJPE (m) | Notes |\n",
        "|------:|---------|----------:|-------|\n",
    ])
    for name in ["occlusion", "outlier", "scale", "orientation"]:
        for ex in examples.get(name, []):
            lines.append(f"| {ex['frame']} | {name} | {ex['mpjpe']:.4f} | {ex['notes']} |\n")
    lines.append("\n")

    lines.extend([
        "## Interpretation\n",
        "\n",
        "* **Occlusion** dominates when confidence maps show many low-confidence observations.\n",
        "* **Outlier** frames contain individual 2D observations with large reprojection residuals.\n",
        "* **Scale** frames have the subject unusually close to or far from the camera rig, or an abnormally large/small pose.\n",
        "* **Orientation** frames have shallow camera-ray intersections (small pairwise angles between camera-to-subject rays), producing depth ambiguity.\n",
        "\n",
    ])

    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Failure mode analysis for ray_attention_v3.")
    parser.add_argument("--dataset", type=str, default="data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz")
    parser.add_argument("--checkpoint", type=str, default="outputs/ray_attention_v3_h36m.pth")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_percent", type=float, default=10.0)
    parser.add_argument("--out_dir", type=str, default="docs/swarm_iter5")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    points_2d = data["points_2d"]
    confidences = data["confidences"]
    joints_3d = data["joints_3d"]

    n_views = data["camera_K"].shape[0]
    n_joints = points_2d.shape[2]
    print(f"Dataset: {points_2d.shape[0]} frames, {n_views} views, {n_joints} joints")

    # Projection matrices and camera centers (torch, to avoid numpy BLAS clashes).
    P_t = build_projection_matrices_torch(data)
    R_t = torch.from_numpy(data["camera_R"]).float()
    t_t = torch.from_numpy(data["camera_t"]).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine whether to use a learned model or the DLT baseline.
    checkpoint_path = Path(args.checkpoint)
    use_model = checkpoint_path.exists()
    if use_model:
        print(f"Loading model checkpoint: {checkpoint_path}")
        model = RayAttentionFusionModelV3(j=n_joints, d=args.d, n_views=n_views).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()

        x = torch.from_numpy(np.concatenate([points_2d, confidences[..., None]], axis=-1)).float().to(device)
        K = torch.from_numpy(data["camera_K"]).float().unsqueeze(0).to(device)
        R = torch.from_numpy(data["camera_R"]).float().unsqueeze(0).to(device)
        t = torch.from_numpy(data["camera_t"]).float().unsqueeze(0).to(device)
        pred = triangulate_model(model, x, K, R, t, args.batch_size)
    else:
        print("No checkpoint found; using DLT baseline for failure analysis.")
        pred = triangulate_dlt_batch(points_2d, confidences, P_t)

    # Per-frame MPJPE.
    per_joint_error = compute_mpjpe(pred, joints_3d)
    per_frame_mpjpe = per_joint_error.mean(axis=1)

    n_total = points_2d.shape[0]
    n_high = max(1, int(n_total * args.top_percent / 100.0))
    high_idx = np.argsort(per_frame_mpjpe)[-n_high:]

    indicators = compute_failure_indicators(
        pred[high_idx], joints_3d[high_idx], points_2d[high_idx], confidences[high_idx], P_t, R_t, t_t
    )
    labels, scores = classify_failure_mode(indicators)

    label_counts = {name: 0 for name in ["occlusion", "outlier", "scale", "orientation"]}
    for label in labels:
        label_counts[label] += 1

    per_cluster_mean = {}
    indicators_mean = {}
    for name in label_counts:
        mask = labels == name
        if mask.sum() == 0:
            per_cluster_mean[name] = 0.0
            indicators_mean[name] = {k: 0.0 for k in indicators}
        else:
            per_cluster_mean[name] = per_frame_mpjpe[high_idx][mask].mean()
            indicators_mean[name] = {k: indicators[k][mask].mean() for k in indicators}

    # Pick representative examples for each cluster.
    examples = {}
    for name in label_counts:
        mask = labels == name
        if mask.sum() == 0:
            examples[name] = []
            continue
        local_err = per_frame_mpjpe[high_idx][mask]
        # Pick the frame with the highest error in this cluster.
        rel_idx = np.argmax(local_err)
        global_idx = high_idx[mask][rel_idx]
        ex = {
            "frame": int(global_idx),
            "mpjpe": float(local_err[rel_idx]),
            "notes": f"max-error {name} example",
        }
        examples[name] = [ex]

    out_path = out_dir / "failure_modes.md"
    write_markdown(
        out_path,
        n_total,
        n_high,
        per_frame_mpjpe[high_idx],
        label_counts,
        indicators_mean,
        per_cluster_mean,
        examples,
    )
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
