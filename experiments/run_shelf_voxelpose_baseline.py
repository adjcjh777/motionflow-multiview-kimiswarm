"""Run confidence-weighted DLT on VoxelPose Shelf 2D predictions.

Expected data tree (download from VoxelPose repo or the Google Drive mirror):
    data/Shelf/
        calibration_shelf.json
        pred_shelf_maskrcnn_hrnet_coco.pkl

Usage:
    .venv/bin/python experiments/run_shelf_voxelpose_baseline.py --data_root data/Shelf --frame_start 300 --frame_end 600 --output outputs/shelf_dlt_result.pkl
"""

import argparse
from pathlib import Path
import pickle

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.pipeline import MultiViewPipeline
from motionflow_mv.pipeline_utils import select_best_person_group


def main():
    parser = argparse.ArgumentParser(description="Run DLT on VoxelPose Shelf predictions.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to Shelf data root containing calibration and pred .pkl")
    parser.add_argument("--frame_start", type=int, default=300)
    parser.add_argument("--frame_end", type=int, default=600)
    parser.add_argument("--output", type=str, default="outputs/shelf_dlt_result.pkl")
    args = parser.parse_args()

    loader = VoxelPoseShelfLoader(args.data_root)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]

    print(f"Loaded {len(cameras)} cameras: {camera_ids}")

    pipeline = MultiViewPipeline(estimator=None)
    results = {}

    for frame_idx in range(args.frame_start, args.frame_end + 1):
        # collect per-view predictions and match the same person across views
        frame_predictions = {cid: loader.get_frame_predictions(cid, frame_idx) for cid in camera_ids}
        # skip if any view has no detections
        if any(len(p) == 0 for p in frame_predictions.values()):
            continue

        try:
            _, points_2d, confidences = select_best_person_group(
                frame_predictions, loader.cameras, camera_ids
            )
        except ValueError:
            continue

        pred_3d = pipeline.fuse_frame(
            points_2d,
            confidences,
            cameras,
        )
        results[frame_idx] = pred_3d

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}, 3D centroid: {pred_3d.mean(axis=0)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {len(results)} frames to {output_path}")


if __name__ == "__main__":
    main()
