"""Synthetic two-person multi-view association smoke test.

Generates short multi-view clips with two people using a lightweight random
skeleton generator (no SMPL dependency), shuffles the per-view skeleton
detections, and recovers the correct person-to-detection association across
cameras by minimizing the multi-view reprojection error.

CPU smoke (fast, no GPU):
    python experiments/associate_multi_person_synthetic.py --smoke

Larger test:
    python experiments/associate_multi_person_synthetic.py --n_sequences 20 --n_frames 30
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.synthetic_3d_dataset import make_cameras
from motionflow_mv.fusion.triangulation import triangulate_dlt


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic two-person association smoke test.")
    parser.add_argument("--n_sequences", type=int, default=10, help="Number of clips to generate.")
    parser.add_argument("--n_frames", type=int, default=10, help="Frames per clip.")
    parser.add_argument("--n_views", type=int, default=4, help="Number of calibrated views.")
    parser.add_argument("--n_joints", type=int, default=17, help="Number of joints per skeleton.")
    parser.add_argument("--noise_std", type=float, default=1.0, help="2D Gaussian noise (pixels).")
    parser.add_argument("--person_separation", type=float, default=1.5,
                        help="Horizontal separation between the two people (meters).")
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke test (2 clips, 5 frames).")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def generate_person_clip(
    n_frames: int,
    n_views: int,
    n_joints: int,
    cameras: list,
    rng: np.random.Generator,
    offset: np.ndarray,
    noise_std: float,
):
    """Generate a smooth random skeleton trajectory and project it into n_views.

    Returns:
        points_2d: (T, V, J, 2)
        joints_3d: (T, J, 3)
    """
    # Base skeleton in metres.
    base = rng.uniform(-1.0, 1.0, size=(n_joints, 3)) * np.array([0.5, 0.8, 1.5])
    base[:, 2] += 3.0
    base[:, 0] += offset[0]
    base[:, 1] += offset[1]
    base[:, 2] += offset[2]

    positions = [base.copy()]
    for _ in range(n_frames - 1):
        delta = rng.normal(0, 0.05, size=base.shape)
        positions.append(positions[-1] + delta)
    positions = np.stack(positions, axis=0)

    proj_matrices = [cam.projection_matrix for cam in cameras]

    points_2d_list = []
    for t in range(n_frames):
        X = positions[t]
        X_h = np.hstack([X, np.ones((n_joints, 1))])
        per_view = []
        for P in proj_matrices:
            x_h = (P @ X_h.T).T
            x = x_h[:, :2] / x_h[:, 2:3]
            x = x + rng.normal(0, noise_std, size=x.shape)
            per_view.append(x)
        points_2d_list.append(np.stack(per_view, axis=0))

    points_2d = np.stack(points_2d_list, axis=0)
    return points_2d, positions


def build_unordered_detections(points_a: np.ndarray, points_b: np.ndarray, rng: np.random.Generator):
    """Shuffle the two people per view to simulate an unordered detector output.

    Args:
        points_a: (T, V, J, 2)
        points_b: (T, V, J, 2)


    Returns:
        detections: (T, V, 2, J, 2) array where detections[t, v, 0] and detections[t, v, 1]
                    are the two detected skeletons.
        gt_labels: (T, V) array, 0/1 indicating which slot corresponds to person A.
    """
    T, V, J, _ = points_a.shape
    detections = np.stack([points_a, points_b], axis=2)  # (T, V, 2, J, 2)
    gt_labels = np.zeros((T, V), dtype=np.int64)
    for t in range(T):
        for v in range(V):
            if rng.random() < 0.5:
                detections[t, v] = detections[t, v, ::-1]
                gt_labels[t, v] = 1
    return detections, gt_labels


def reprojection_error_for_labeling(
    labeling: tuple,
    detections: np.ndarray,
    proj_matrices: np.ndarray,
) -> float:
    """Compute total reprojection error for a given view labeling.

    Args:
        labeling: (V,) tuple of 0/1 selecting which slot is treated as person A.
        detections: (T, V, 2, J, 2) unordered detections.
        proj_matrices: (V, 3, 4)

    Returns:
        Scalar total reprojection error (pixels^2) across T frames, two people, all joints/views.
    """
    T, V, _, J, _ = detections.shape
    total_err = 0.0
    labeling_arr = np.array(labeling)
    for t in range(T):
        for person in range(2):
            slot_idx = labeling_arr if person == 0 else 1 - labeling_arr
            for j in range(J):
                pts = np.stack([detections[t, v, slot_idx[v], j] for v in range(V)], axis=0)
                X3d = triangulate_dlt(pts, proj_matrices)
                X3d_h = np.append(X3d, 1.0)
                for v in range(V):
                    x_h = proj_matrices[v] @ X3d_h
                    x_proj = x_h[:2] / x_h[2]
                    total_err += float(np.sum((x_proj - pts[v]) ** 2))
    return total_err


def associate_detections(
    detections: np.ndarray,
    proj_matrices: np.ndarray,
) -> np.ndarray:
    """Return predicted (T, V) labeling with the minimum reprojection error.

    Args:
        detections: (T, V, 2, J, 2) unordered detections.
        proj_matrices: (V, 3, 4)

    Returns:
        labels: (T, V) predicted labeling for person A.
    """
    T, V, _, _, _ = detections.shape
    labelings = list(product([0, 1], repeat=V))
    pred_labels = np.zeros((T, V), dtype=np.int64)
    for t in range(T):
        best_err = float("inf")
        best_labeling = None
        for labeling in labelings:
            err = reprojection_error_for_labeling(
                labeling, detections[t:t + 1], proj_matrices
            )
            if err < best_err:
                best_err = err
                best_labeling = labeling
        pred_labels[t] = best_labeling
    return pred_labels


def compute_metrics(pred_labels: np.ndarray, gt_labels: np.ndarray) -> dict:
    """Compute association accuracy allowing a global person swap."""
    correct = (pred_labels == gt_labels).all(axis=1)
    correct_swap = (pred_labels == (1 - gt_labels)).all(axis=1)
    frame_acc = np.maximum(correct, correct_swap).mean()
    return {
        "frame_accuracy": float(frame_acc),
        "raw_accuracy": float(correct.mean()),
        "swap_accuracy": float(correct_swap.mean()),
        "n_frames": int(gt_labels.shape[0]),
        "n_views": int(gt_labels.shape[1]),
    }


def main():
    args = parse_args()
    if args.smoke:
        args.n_sequences = 2
        args.n_frames = 5
        args.n_views = 4
        print("Running CPU smoke test...")

    rng = np.random.default_rng(args.seed)
    print(f"Generating {args.n_sequences} clips x {args.n_frames} frames x {args.n_views} views")

    all_metrics = []
    total_time = 0.0

    for seq_idx in range(args.n_sequences):
        cameras = make_cameras(args.n_views, rng)
        proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)

        points_2d_a, joints_3d_a = generate_person_clip(
            args.n_frames, args.n_views, args.n_joints, cameras, rng,
            offset=np.array([0.0, 0.0, 0.0]), noise_std=args.noise_std,
        )
        points_2d_b, joints_3d_b = generate_person_clip(
            args.n_frames, args.n_views, args.n_joints, cameras, rng,
            offset=np.array([args.person_separation, 0.0, 0.0]), noise_std=args.noise_std,
        )

        detections, gt_labels = build_unordered_detections(points_2d_a, points_2d_b, rng)

        t0 = time.perf_counter()
        pred_labels = associate_detections(detections, proj_matrices)
        total_time += time.perf_counter() - t0

        metrics = compute_metrics(pred_labels, gt_labels)
        metrics["seq_idx"] = seq_idx
        all_metrics.append(metrics)

    overall_acc = np.mean([m["frame_accuracy"] for m in all_metrics])

    print("\n=== Synthetic Two-Person Association Results ===")
    print(f"Sequences: {args.n_sequences}, Frames/seq: {args.n_frames}, Views: {args.n_views}")
    print(f"Person separation: {args.person_separation} m, Noise std: {args.noise_std} px")
    print(f"Overall frame-wise association accuracy: {overall_acc * 100:.2f}%")
    print(f"Mean per-frame association time: {total_time / (args.n_sequences * args.n_frames) * 1000:.2f} ms")
    print("\nPer-sequence metrics:")
    for m in all_metrics:
        print(f"  seq {m['seq_idx']:02d}: frame_acc={m['frame_accuracy'] * 100:.2f}% "
              f"(raw={m['raw_accuracy'] * 100:.2f}%, swap={m['swap_accuracy'] * 100:.2f}%)")


if __name__ == "__main__":
    main()
