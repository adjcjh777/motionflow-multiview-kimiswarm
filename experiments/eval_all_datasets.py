"""Unified evaluation protocol across H36M, synthetic, Shelf and Campus.

Computes MPJPE, PA-MPJPE, PCK@50/100/150mm, PCK-AUC and per-joint/per-view
breakdowns for DLT baseline and any available learned checkpoint.

Usage:
    # H36M subset
    /d/anaconda3/envs/jz_py310/python.exe experiments/eval_all_datasets.py \
        --h36m data/h36m_hf/s_01_act_02_multiview.npz

    # Synthetic (auto-generated if --synthetic is omitted and no file exists)
    /d/anaconda3/envs/jz_py310/python.exe experiments/eval_all_datasets.py --synthetic

    # Shelf/Campus (reprojection only, no 3D GT)
    /d/anaconda3/envs/jz_py310/python.exe experiments/eval_all_datasets.py \
        --shelf tmp/voxelpose-pytorch/data/Shelf \
        --campus tmp/voxelpose-pytorch/data/CampusSeq1

Summary (2025-08-04 swarm task):
    Added a single entry-point for cross-dataset evaluation. 3D GT is required
    for the full metric suite (H36M, synthetic). Shelf/Campus report
    reprojection error because 3D GT is not available locally.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics, summarize_metrics
from motionflow_mv.fusion.triangulation import triangulate_dlt


def _dlt_baseline(points_2d: np.ndarray, confidences: np.ndarray,
                  K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Triangulate a batch of multi-view 2D keypoints with DLT."""
    # points_2d: (B, V, J, 2), confidences: (B, V, J)
    # K, R, t describe one rig (V, ...).
    B, V, J, _ = points_2d.shape
    P = K @ np.concatenate([R, t[..., None]], axis=-1)  # (V, 3, 4)
    X = np.zeros((B, J, 3), dtype=np.float64)
    for b in range(B):
        for j_idx in range(J):
            w = confidences[b, :, j_idx]
            if w.sum() == 0:
                w = np.ones_like(w)
            X[b, j_idx] = triangulate_dlt(points_2d[b, :, j_idx], P, weights=w)
    return X


def _load_h36m(path: str):
    """Load an H36M .npz and return a small evaluation sample."""
    data = np.load(path)
    points_2d = data["points_2d"]
    confidences = data["confidences"]
    joints_3d = data["joints_3d"]
    K = data["camera_K"]
    R = data["camera_R"]
    t = data["camera_t"]
    return points_2d, confidences, joints_3d, K, R, t


def _generate_synthetic(n_views: int = 5, j: int = 17, t: int = 200, seed: int = 2025):
    """Generate a small synthetic multi-view sequence with 3D GT."""
    from experiments.train_attention_fusion import make_cameras
    rng = np.random.default_rng(seed)
    base = rng.uniform(-1.0, 1.0, size=(j, 3)) * np.array([0.5, 0.8, 1.5])
    base[:, 2] += 3.0
    trajectory = np.cumsum(rng.normal(0, 0.05, size=(t, 3)), axis=0)
    joints_3d = base[None, :, :] + trajectory[:, None, :]  # (T, J, 3)

    cameras = make_cameras(n_views, rng)
    proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)

    points_2d = np.zeros((t, n_views, j, 2), dtype=np.float64)
    confidences = np.ones((t, n_views, j), dtype=np.float64) * 0.9
    for v in range(n_views):
        P = proj_matrices[v]
        X_h = np.concatenate([joints_3d, np.ones((t, j, 1))], axis=-1)
        x_h = (P @ X_h.reshape(-1, 4).T).T.reshape(t, j, 3)
        points_2d[:, v] = x_h[..., :2] / x_h[..., 2:]
        points_2d[:, v] += rng.normal(0, 0.5, size=(t, j, 2))

    K = np.stack([cam.K for cam in cameras], axis=0)
    R = np.stack([cam.R for cam in cameras], axis=0)
    t = np.stack([cam.t for cam in cameras], axis=0)
    return points_2d, confidences, joints_3d, K, R, t


def _evaluate_h36m(path: str, max_frames: int = None, device: torch.device = None,
                     model=None):
    points_2d, confidences, joints_3d, K, R, t = _load_h36m(path)
    if max_frames and points_2d.shape[0] > max_frames:
        points_2d = points_2d[:max_frames]
        confidences = confidences[:max_frames]
        joints_3d = joints_3d[:max_frames]
    pred_dlt = _dlt_baseline(points_2d, confidences, K, R, t)
    results = {"dlt": compute_all_metrics(pred_dlt, joints_3d)}
    if model is not None:
        pred_model = _run_model(model, points_2d, confidences, K, R, t, device)
        results["model"] = compute_all_metrics(pred_model, joints_3d)
    return results


def _run_model(model, points_2d, confidences, K, R, t, device,
               batch_size: int = 64):
    """Run a ray-aware attention model on numpy arrays."""
    model.eval()
    K_t = torch.from_numpy(K).float().unsqueeze(0).to(device)
    R_t = torch.from_numpy(R).float().unsqueeze(0).to(device)
    t_t = torch.from_numpy(t).float().unsqueeze(0).to(device)
    x = torch.from_numpy(np.concatenate([points_2d, confidences[..., None]],
                                         axis=-1)).float()
    dataset = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            Kb = K_t.expand(xb.size(0), -1, -1, -1)
            Rb = R_t.expand(xb.size(0), -1, -1, -1)
            tb = t_t.expand(xb.size(0), -1, -1)
            pred, _ = model(xb, K=Kb, R=Rb, t=tb)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def _evaluate_synthetic(max_frames: int = None, device: torch.device = None,
                        model=None):
    points_2d, confidences, joints_3d, K, R, t = _generate_synthetic()
    if max_frames and points_2d.shape[0] > max_frames:
        points_2d = points_2d[:max_frames]
        confidences = confidences[:max_frames]
        joints_3d = joints_3d[:max_frames]
    pred_dlt = _dlt_baseline(points_2d, confidences, K, R, t)
    results = {"dlt": compute_all_metrics(pred_dlt, joints_3d)}
    if model is not None:
        pred_model = _run_model(model, points_2d, confidences, K, R, t, device)
        results["model"] = compute_all_metrics(pred_model, joints_3d)
    return results


def _reprojection_error(pred_3d: np.ndarray, points_2d: np.ndarray,
                        P: np.ndarray) -> np.ndarray:
    """Return per-joint per-view reprojection error in pixels."""
    V, J, _ = points_2d.shape
    X_h = np.concatenate([pred_3d, np.ones((J, 1))], axis=-1)  # (J, 4)
    x_h = (P @ X_h.T).T  # (V, J, 3)
    x = x_h[..., :2] / x_h[..., 2:]
    return np.linalg.norm(x - points_2d, axis=-1)  # (V, J)


def _evaluate_reprojection_only(loader, frame_start: int, frame_end: int,
                                  device: torch.device = None, model=None):
    """Evaluate plugins when only 2D GT + calibration is available.

    Returns per-joint and per-view mean reprojection errors (px).
    """
    from motionflow_mv.pipeline_utils import select_best_person_group

    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]
    proj = np.stack([cam.projection_matrix for cam in cameras], axis=0)

    errors = []
    for frame_idx in range(frame_start, frame_end + 1):
        frame_predictions = {cid: loader.get_frame_predictions(cid, frame_idx)
                             for cid in camera_ids}
        if any(len(frame_predictions[cid]) == 0 for cid in camera_ids):
            continue
        try:
            _, points_2d, confidences = select_best_person_group(
                frame_predictions, loader.cameras, camera_ids)
        except ValueError:
            continue
        pred = _dlt_baseline(points_2d[None], confidences[None],
                             cameras[0].K, cameras[0].R, cameras[0].t)
        pred = pred[0]
        err = _reprojection_error(pred, points_2d, proj)
        errors.append(err)
    if not errors:
        return None
    errors = np.stack(errors, axis=0)  # (F, V, J)
    return {
        "mean_px": float(errors.mean()),
        "per_view_mean_px": errors.mean(axis=(0, 2)).tolist(),
        "per_joint_mean_px": errors.mean(axis=(0, 1)).tolist(),
    }


def _load_model(args, device: torch.device):
    """Load a learned model if a checkpoint or architecture is requested."""
    checkpoint = args.checkpoint
    if checkpoint and not Path(checkpoint).exists():
        print(f"Checkpoint {checkpoint} not found; model evaluation skipped.")
        return None
    if not checkpoint:
        return None

    if args.model == "ray_attention_v3":
        from motionflow_mv.fusion.ray_attention_v3_model import \
            RayAttentionFusionModelV3
        model = RayAttentionFusionModelV3(
            j=args.joints, d=args.d, n_views=args.n_views,
        ).to(device)
    elif args.model == "ray_attention_v2":
        from motionflow_mv.fusion.ray_attention_v2_model import \
            RayAttentionFusionModelV2
        model = RayAttentionFusionModelV2(
            j=args.joints, d=args.d, n_views=args.n_views,
        ).to(device)
    else:
        raise ValueError(f"Unknown model {args.model}")

    model.load_state_dict(torch.load(checkpoint, map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"Loaded model {args.model} from {checkpoint}")
    return model


def _print_report(name: str, results: dict, detailed: bool = False):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    for method, report in results.items():
        print(f"\n  Method: {method}")
        print("  " + summarize_metrics(report).replace("\n", "\n  "))
        if detailed:
            print("  Per-joint MPJPE:")
            for j_idx, err in enumerate(report["per_joint_mpjpe"]):
                print(f"    joint {j_idx:2d}: {err:.4f}")
            if "per_joint_pa_mpjpe" in report:
                print("  Per-joint PA-MPJPE:")
                for j_idx, err in enumerate(report["per_joint_pa_mpjpe"]):
                    print(f"    joint {j_idx:2d}: {err:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified 3D pose evaluation across datasets."
    )
    parser.add_argument("--h36m", type=str, default=None,
                        help="Path to an H36M multiview .npz file.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate and evaluate on synthetic data.")
    parser.add_argument("--shelf", type=str, default=None,
                        help="Path to Shelf data root (reprojection only).")
    parser.add_argument("--campus", type=str, default=None,
                        help="Path to Campus data root (reprojection only).")
    parser.add_argument("--frame_start", type=int, default=300)
    parser.add_argument("--frame_end", type=int, default=600)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit number of frames for quick tests.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained checkpoint.")
    parser.add_argument("--model", type=str, default="ray_attention_v3",
                        choices=["ray_attention_v3", "ray_attention_v2"])
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--joints", type=int, default=17)
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--detailed", action="store_true",
                        help="Print per-joint/per-view breakdown.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = _load_model(args, device) if args.checkpoint else None

    if args.h36m:
        path = Path(args.h36m)
        if path.exists():
            results = _evaluate_h36m(args.h36m, max_frames=args.max_frames,
                                     device=device, model=model)
            _print_report(path.stem, results, detailed=args.detailed)
        else:
            print(f"H36M file {args.h36m} not found; skipping.")

    if args.synthetic:
        results = _evaluate_synthetic(max_frames=args.max_frames,
                                      device=device, model=model)
        _print_report("synthetic", results, detailed=args.detailed)

    if args.shelf:
        from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
        loader = VoxelPoseShelfLoader(args.shelf)
        report = _evaluate_reprojection_only(
            loader, args.frame_start, args.frame_end, device=device, model=model)
        if report:
            print("\nShelf reprojection (DLT baseline, px):")
            print(f"  mean: {report['mean_px']:.4f}")
            if args.detailed:
                print("  per-view:", report["per_view_mean_px"])
                print("  per-joint:", report["per_joint_mean_px"])
        else:
            print("No valid Shelf frames found.")

    if args.campus:
        from motionflow_mv.data.voxelpose_loader import VoxelPoseCampusLoader
        loader = VoxelPoseCampusLoader(args.campus)
        report = _evaluate_reprojection_only(
            loader, 0, 1200, device=device, model=model)
        if report:
            print("\nCampus reprojection (DLT baseline, px):")
            print(f"  mean: {report['mean_px']:.4f}")
            if args.detailed:
                print("  per-view:", report["per_view_mean_px"])
                print("  per-joint:", report["per_joint_mean_px"])
        else:
            print("No valid Campus frames found.")


if __name__ == "__main__":
    main()
