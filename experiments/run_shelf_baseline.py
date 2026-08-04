"""Placeholder baseline experiment on Shelf / Campus.

This script documents the intended workflow. It requires:
    - a dataset root with `calibration/` and `frames/` directories
    - per-view 2D keypoints (or 2D GT) for each frame

Once data is prepared, the script loads cameras, runs confidence-weighted DLT,
and computes MPJPE/PA-MPJPE against 3D GT.
"""

import argparse
import json
import numpy as np

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.pipeline import MultiViewPipeline
from motionflow_mv.eval.metrics import mpjpe, pa_mpjpe


def load_cameras(calibration_dir: str, n_views: int):
    """Load K, R, t from JSON files calibration/cam_00.json ... cam_NN.json."""
    cameras = []
    for i in range(n_views):
        with open(f"{calibration_dir}/cam_{i:02d}.json") as f:
            cam_dict = json.load(f)
        cameras.append(Camera(K=cam_dict["K"], R=cam_dict["R"], t=cam_dict["t"]))
    return cameras


def main():
    parser = argparse.ArgumentParser(description="Run confidence-weighted DLT on Shelf/Campus.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--n_views", type=int, default=5)
    args = parser.parse_args()

    cameras = load_cameras(f"{args.data_root}/calibration", args.n_views)

    # Placeholder: load per-view 2D detections and 3D GT
    # points_2d shape: (T, V, J, 2); confidences: (T, V, J); gt_3d: (T, J, 3)
    points_2d = np.load(f"{args.data_root}/points_2d.npy")
    confidences = np.load(f"{args.data_root}/confidences.npy")
    gt_3d = np.load(f"{args.data_root}/gt_3d.npy")

    pipeline = MultiViewPipeline(estimator=None)  # 2D already provided in this baseline
    t, v, j, _ = points_2d.shape
    pred_3d = np.zeros((t, j, 3))
    for frame_idx in range(t):
        pred_3d[frame_idx] = pipeline.fuse_frame(
            points_2d[frame_idx], confidences[frame_idx], cameras
        )

    print(f"MPJPE: {mpjpe(pred_3d, gt_3d):.2f} mm")
    print(f"PA-MPJPE: {pa_mpjpe(pred_3d, gt_3d):.2f} mm")


if __name__ == "__main__":
    main()
