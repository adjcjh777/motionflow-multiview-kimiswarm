"""Failure-case analysis for the cross-view temporal residual + principal-point model.

Loads a checkpoint for ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``,
evaluates on a canonical .npz, and writes:

* per-joint / per-frame / per-view error tables
* predicted principal-point correction statistics
* per-view fusion weight statistics
* heatmaps and line plots under ``outputs/failure_analysis_crossview_pp/``
* a Markdown report under ``docs/swarm_iter_next/``

Usage:
    conda run -n mf python experiments/analyze_failures_crossview_pp.py \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_full_ppw005_20ep.pth \
        --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)

# MPI-INF-3DHP 28-joint skeleton (WebBridge convention).
JOINT_NAMES = [
    "pelvis", "thorax", "neck", "head",
    "l_shoulder", "l_elbow", "l_wrist", "l_hand",
    "r_shoulder", "r_elbow", "r_wrist", "r_hand",
    "l_hip", "l_knee", "l_ankle", "l_foot",
    "r_hip", "r_knee", "r_ankle", "r_foot",
    "spine", "l_hand_tip", "l_thumb", "r_hand_tip", "r_thumb",
    "l_eye", "l_ear", "r_eye",
]
while len(JOINT_NAMES) < 28:
    JOINT_NAMES.append(f"joint_{len(JOINT_NAMES)}")


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pa_mpjpe(pred, gt):
    from motionflow_mv.eval.metrics import pa_mpjpe as _pa_mpjpe
    return _pa_mpjpe(pred, gt)


def sliding_window_inference(model, points_2d, confidences, K, R, t,
                             clip_len: int = 13, stride: int = 13,
                             batch_size: int = 32,
                             device: torch.device = None,
                             return_residual: bool = False) -> tuple:
    """Run PP cross-view model on sliding windows and average predictions."""
    T, V, J, _ = points_2d.shape
    windows = []
    for start in range(0, T - clip_len + 1, stride):
        end = start + clip_len
        x = torch.cat([points_2d[start:end], confidences[start:end].unsqueeze(-1)], dim=-1)
        windows.append((start, end, x))

    pred_sum = torch.zeros(T, J, 3, dtype=torch.float32)
    pred_count = torch.zeros(T, 1, 1, dtype=torch.float32)
    weight_sum = torch.zeros(T, V, J, dtype=torch.float32)
    pp_delta_sum = torch.zeros(T, V, 2, dtype=torch.float32)
    if return_residual:
        residual_sum = torch.zeros(T, J, 3, dtype=torch.float32)
        residual_count = torch.zeros(T, 1, 1, dtype=torch.float32)

    K = K.to(device)
    R = R.to(device)
    t = t.to(device)

    last_valid_end = 0
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        xb = torch.stack([b[2] for b in batch], dim=0).to(device)

        hook_handle = None
        residual_container = []
        if return_residual:
            def _residual_hook(module, input, output):
                residual_container.append(output)
            hook_handle = model.residual_mlp.register_forward_hook(_residual_hook)

        with torch.no_grad():
            pred_b, weights_b, pp_delta_b = model(xb, K=K, R=R, t=t)
        pred_b = pred_b.cpu()
        weights_b = weights_b.cpu()
        pp_delta_b = pp_delta_b.cpu()
        if return_residual:
            if hook_handle is not None:
                hook_handle.remove()
            if residual_container:
                delta_b = residual_container[0].view_as(pred_b).cpu()
            else:
                delta_b = torch.zeros_like(pred_b)
        for j_idx, (start, end, _) in enumerate(batch):
            pred_sum[start:end] += pred_b[j_idx]
            pred_count[start:end] += 1.0
            weight_sum[start:end] += weights_b[j_idx]
            pp_delta_sum[start:end] += pp_delta_b[j_idx]
            if return_residual:
                residual_sum[start:end] += delta_b[j_idx]
                residual_count[start:end] += 1.0
            last_valid_end = max(last_valid_end, end)

    pred = pred_sum / pred_count
    weights = weight_sum / pred_count
    pp_delta = pp_delta_sum / pred_count
    if last_valid_end < T:
        pred[last_valid_end:] = pred[last_valid_end - 1].unsqueeze(0)
        weights[last_valid_end:] = weights[last_valid_end - 1].unsqueeze(0)
        pp_delta[last_valid_end:] = pp_delta[last_valid_end - 1].unsqueeze(0)
        if return_residual:
            residual_sum[last_valid_end:] = residual_sum[last_valid_end - 1].unsqueeze(0)
            residual_count[last_valid_end:] = residual_count[last_valid_end - 1].unsqueeze(0)
    if return_residual:
        residual = residual_sum / residual_count
        return pred, weights, pp_delta, residual
    return pred, weights, pp_delta


def compute_per_view_reprojection(pred_3d, points_2d, confidences, K, R, t):
    """Per-view reprojection error in pixels (T, V)."""
    T, V, J, _ = points_2d.shape
    reproj = np.zeros((T, V), dtype=np.float64)
    for v in range(V):
        P = K[v] @ np.concatenate([R[v], t[v][:, None]], axis=-1)
        Xh = np.concatenate([pred_3d, np.ones((T, J, 1))], axis=-1)
        x = (P[None, :, :] @ Xh.transpose(0, 2, 1)).transpose(0, 2, 1)
        x = x[..., :2] / np.clip(x[..., 2:3], 1e-6, None)
        err = np.linalg.norm(x - points_2d[:, v], axis=-1)
        mask = confidences[:, v] > 0.0
        reproj[:, v] = (err * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
    return reproj


def analyze(dataset_path: str, checkpoint_path: str, clip_len: int, stride: int,
            batch_size: int, d: int, n_st_layers: int, residual_hidden: int,
            out_dir: str, report_dir: str, device: torch.device, seed: int = 42):
    set_seed(seed)

    data = np.load(dataset_path)
    points_2d = torch.from_numpy(data["points_2d"]).float()
    confidences = torch.from_numpy(data["confidences"]).float()
    joints_3d = data["joints_3d"].astype(np.float64)
    K = torch.from_numpy(data["camera_K"]).float()
    R = torch.from_numpy(data["camera_R"]).float()
    t = torch.from_numpy(data["camera_t"]).float()
    T, V, J, _ = points_2d.shape
    print(f"Dataset: {dataset_path}\n  frames={T}, views={V}, joints={J}")

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=J, d=d, n_views=V, n_st_layers=n_st_layers,
        residual_hidden=residual_hidden, return_pp_delta=True,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    print("Running sliding-window inference ...")
    pred, weights, pp_delta, residual = sliding_window_inference(
        model, points_2d, confidences, K, R, t,
        clip_len=clip_len, stride=stride, batch_size=batch_size, device=device,
        return_residual=True,
    )
    pred = pred.numpy()
    weights = weights.numpy()
    pp_delta = pp_delta.numpy()
    residual = residual.numpy()

    per_joint_err = np.linalg.norm(pred - joints_3d, axis=-1)
    mean_per_joint = per_joint_err.mean(axis=0) * 1000.0
    worst_joints = np.argsort(mean_per_joint)[::-1]

    mpjpe_val = per_joint_err.mean() * 1000.0
    if np.isfinite(pred).all() and np.isfinite(joints_3d).all():
        pampjpe_val = pa_mpjpe(pred, joints_3d) * 1000.0
    else:
        pampjpe_val = float("nan")
        print("Warning: non-finite predictions; PA-MPJPE set to NaN")
    print(f"MPJPE: {mpjpe_val:.2f} mm")
    print(f"PA-MPJPE: {pampjpe_val:.2f} mm")

    per_frame_err = per_joint_err.mean(axis=1) * 1000.0
    worst_frames = np.argsort(per_frame_err)[::-1]

    per_view_reproj = compute_per_view_reprojection(
        pred, points_2d.numpy(), confidences.numpy(), K.numpy(), R.numpy(), t.numpy()
    )
    mean_view_err = per_view_reproj.mean(axis=0)
    median_view_err = np.median(per_view_reproj, axis=0)
    worst_views = np.argsort(median_view_err)[::-1]

    # PP correction magnitude per view.
    pp_delta_norm = np.linalg.norm(pp_delta, axis=-1)  # (T, V)
    mean_pp_delta = pp_delta_norm.mean(axis=0)

    # Weight statistics.
    mean_weights = weights.mean(axis=(0, 2))  # (V,)

    # Residual correction statistics (m -> mm).
    residual_norm = np.linalg.norm(residual, axis=-1) * 1000.0  # (T, J)
    mean_residual_per_joint = residual_norm.mean(axis=0)
    mean_residual_per_frame = residual_norm.mean(axis=1)
    worst_residual_joints = np.argsort(mean_residual_per_joint)[::-1]
    overall_residual = residual_norm.mean()
    print(f"Mean residual correction: {overall_residual:.2f} mm")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path / "failure_arrays.npz",
        pred_3d=pred,
        gt_3d=joints_3d,
        per_joint_err_mm=per_joint_err * 1000.0,
        per_frame_err_mm=per_frame_err,
        per_view_reproj_px=per_view_reproj,
        mean_per_joint_mm=mean_per_joint,
        mean_per_view_px=mean_view_err,
        median_per_view_px=median_view_err,
        pp_delta_norm_px=pp_delta_norm,
        mean_pp_delta_px=mean_pp_delta,
        mean_weights=mean_weights,
        residual_mm=residual * 1000.0,
        residual_norm_mm=residual_norm,
        mean_residual_per_joint_mm=mean_residual_per_joint,
        mean_residual_per_frame_mm=mean_residual_per_frame,
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_results(mean_per_joint, worst_joints, per_frame_err, worst_frames,
                     median_view_err, worst_views, mean_pp_delta, mean_weights,
                     mean_residual_per_joint, worst_residual_joints,
                     mean_residual_per_frame, out_path)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    write_report(
        report_path / "failure_analysis_crossview_pp.md",
        dataset_path, checkpoint_path, clip_len, mpjpe_val, pampjpe_val,
        worst_joints, mean_per_joint, worst_frames, per_frame_err,
        worst_views, mean_view_err, median_view_err, mean_pp_delta,
        mean_weights, mean_residual_per_joint, worst_residual_joints,
        mean_residual_per_frame, overall_residual, out_path,
    )
    print(f"Report written to: {report_path / 'failure_analysis_crossview_pp.md'}")


def plot_results(mean_per_joint, worst_joints, per_frame_err, worst_frames,
                 median_view_err, worst_views, mean_pp_delta, mean_weights,
                 mean_residual_per_joint, worst_residual_joints,
                 mean_residual_per_frame, out_path):
    import matplotlib.pyplot as plt

    # Per-joint error.
    plt.figure(figsize=(12, 6))
    names = [JOINT_NAMES[i] for i in worst_joints]
    vals = mean_per_joint[worst_joints]
    plt.barh(names[::-1], vals[::-1])
    plt.xlabel("Mean MPJPE per joint (mm)")
    plt.title("Per-joint error (worst to best)")
    plt.tight_layout()
    plt.savefig(out_path / "per_joint_error.png", dpi=150)
    plt.close()

    # Per-frame error.
    plt.figure(figsize=(14, 5))
    plt.plot(per_frame_err, alpha=0.6)
    for rank, f in enumerate(worst_frames[:10], 1):
        plt.axvline(f, color="red", alpha=0.15)
        if rank <= 5:
            plt.text(f, per_frame_err[f] * 1.05, f"{f}", fontsize=6, rotation=90)
    plt.xlabel("Frame index")
    plt.ylabel("MPJPE (mm)")
    plt.title("Per-frame MPJPE")
    plt.tight_layout()
    plt.savefig(out_path / "per_frame_error.png", dpi=150)
    plt.close()

    # Per-view reprojection.
    plt.figure(figsize=(10, 5))
    views = worst_views
    plt.bar(range(len(views)), median_view_err[views])
    plt.xticks(range(len(views)), [f"{v}" for v in views])
    plt.xlabel("Camera view index (sorted by median error)")
    plt.ylabel("Median reprojection error (px)")
    plt.title("Per-view reprojection error")
    plt.tight_layout()
    plt.savefig(out_path / "per_view_error.png", dpi=150)
    plt.close()

    # PP correction magnitude.
    plt.figure(figsize=(10, 5))
    plt.bar(range(V := len(mean_pp_delta)), mean_pp_delta)
    plt.xlabel("Camera view index")
    plt.ylabel("Mean PP correction magnitude (px)")
    plt.title("Predicted principal-point correction magnitude per view")
    plt.tight_layout()
    plt.savefig(out_path / "pp_delta_magnitude.png", dpi=150)
    plt.close()

    # Mean weights.
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(mean_weights)), mean_weights)
    plt.xlabel("Camera view index")
    plt.ylabel("Mean fusion weight")
    plt.title("Mean per-view fusion weight")
    plt.tight_layout()
    plt.savefig(out_path / "mean_view_weights.png", dpi=150)
    plt.close()

    # Residual correction magnitude per joint.
    plt.figure(figsize=(12, 6))
    names = [JOINT_NAMES[i] for i in worst_residual_joints]
    vals = mean_residual_per_joint[worst_residual_joints]
    plt.barh(names[::-1], vals[::-1])
    plt.xlabel("Mean residual correction magnitude per joint (mm)")
    plt.title("Residual correction magnitude per joint (worst to best)")
    plt.tight_layout()
    plt.savefig(out_path / "residual_correction_per_joint.png", dpi=150)
    plt.close()

    # Residual correction magnitude per frame.
    plt.figure(figsize=(14, 5))
    plt.plot(mean_residual_per_frame, alpha=0.6)
    plt.xlabel("Frame index")
    plt.ylabel("Mean residual correction magnitude (mm)")
    plt.title("Per-frame residual correction magnitude")
    plt.tight_layout()
    plt.savefig(out_path / "residual_correction_per_frame.png", dpi=150)
    plt.close()


def write_report(path, dataset_path, checkpoint_path, clip_len, mpjpe, pampjpe,
                 worst_joints, mean_per_joint, worst_frames, per_frame_err,
                 worst_views, mean_view_err, median_view_err, mean_pp_delta,
                 mean_weights, mean_residual_per_joint, worst_residual_joints,
                 mean_residual_per_frame, overall_residual, out_path):
    with open(path, "w") as f:
        f.write("# Failure-Case Analysis: Cross-View Residual + PP Model\n\n")
        f.write("## Setup\n\n")
        f.write(f"* Dataset: `{dataset_path}`\n")
        f.write(f"* Checkpoint: `{checkpoint_path}`\n")
        f.write(f"* clip_len: {clip_len}\n\n")
        f.write("## Overall metrics\n\n")
        f.write(f"* MPJPE: **{mpjpe:.2f} mm**\n")
        f.write(f"* PA-MPJPE: **{pampjpe:.2f} mm**\n")
        f.write(f"* Mean residual correction: **{overall_residual:.2f} mm**\n\n")
        f.write("## Worst joints\n\n")
        f.write("| Rank | Joint | MPJPE (mm) |\n")
        f.write("|---|---:|---:|\n")
        for rank, j in enumerate(worst_joints[:10], 1):
            f.write(f"| {rank} | {JOINT_NAMES[j]} | {mean_per_joint[j]:.2f} |\n")
        f.write("\n## Worst frames\n\n")
        for rank, frame in enumerate(worst_frames[:10], 1):
            f.write(f"{rank}. Frame {frame}: {per_frame_err[frame]:.2f} mm\n")
        f.write("\n## Per-view reprojection error\n\n")
        f.write("| View | Mean (px) | Median (px) | PP delta (px) | Mean weight |\n")
        f.write("|---|---|---|---|---|\n")
        for v in worst_views:
            f.write(f"| {v} | {mean_view_err[v]:.2f} | {median_view_err[v]:.2f} | "
                    f"{mean_pp_delta[v]:.2f} | {mean_weights[v]:.3f} |\n")
        f.write("\n## Residual correction\n\n")
        f.write(f"* Overall mean residual correction magnitude: **{overall_residual:.2f} mm**\n")
        f.write("| Rank | Joint | Mean residual correction (mm) |\n")
        f.write("|---|---:|---:|\n")
        for rank, j in enumerate(worst_residual_joints[:10], 1):
            f.write(f"| {rank} | {JOINT_NAMES[j]} | {mean_residual_per_joint[j]:.2f} |\n")
        f.write(f"\nFigures saved to: `{out_path}`\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--stride", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="outputs/failure_analysis_crossview_pp")
    parser.add_argument("--report_dir", type=str, default="docs/swarm_iter_next")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    analyze(args.dataset, args.checkpoint, args.clip_len, args.stride,
            args.batch_size, args.d, args.n_st_layers, args.residual_hidden,
            args.out_dir, args.report_dir, device, args.seed)


if __name__ == "__main__":
    main()
