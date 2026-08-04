"""Evaluate the DLT output on Shelf/VoxelPose 2D predictions.

Metrics computed without 3D GT:
- reprojection error (pixels)
- per-frame 3D skeleton centroid / scale
"""

import argparse
from pathlib import Path
import pickle

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.pipeline_utils import select_best_person_group


def reprojection_error(pred_3d: np.ndarray, points_2d: np.ndarray, camera) -> np.ndarray:
    """Return per-joint reprojection error in pixels."""
    P = camera.projection_matrix
    X_h = np.hstack([pred_3d, np.ones((pred_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return np.linalg.norm(x - points_2d, axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DLT output on Shelf 2D predictions.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--result_pkl", type=str, required=True)
    parser.add_argument("--frame_start", type=int, default=300)
    parser.add_argument("--frame_end", type=int, default=600)
    args = parser.parse_args()

    loader = VoxelPoseShelfLoader(args.data_root)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))

    with open(args.result_pkl, "rb") as f:
        results = pickle.load(f, encoding="latin1")

    all_errors = []
    for frame_idx in range(args.frame_start, args.frame_end + 1):
        if frame_idx not in results:
            continue
        pred_3d = results[frame_idx]

        frame_predictions = {cid: loader.get_frame_predictions(cid, frame_idx) for cid in camera_ids}
        if any(len(p) == 0 for p in frame_predictions.values()):
            continue
        try:
            _, points_2d, _ = select_best_person_group(
                frame_predictions, loader.cameras, camera_ids
            )
        except ValueError:
            continue

        for i, cid in enumerate(camera_ids):
            err = reprojection_error(pred_3d, points_2d[i], loader.get_camera(cid))
            all_errors.append(err)

    if all_errors:
        all_errors = np.concatenate(all_errors)
        print(f"Frames evaluated: {len(results)}")
        print(f"Reprojection error (px) - mean: {all_errors.mean():.2f}, median: {np.median(all_errors):.2f}, max: {all_errors.max():.2f}")
    else:
        print("No frames to evaluate.")


if __name__ == "__main__":
    main()
