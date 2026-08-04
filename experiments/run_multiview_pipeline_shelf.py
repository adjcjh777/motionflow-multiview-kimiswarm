"""End-to-end multi-view pipeline demo on Shelf.

Loads the matched Shelf 2D predictions, triangulates each frame with the
MultiViewPipeline, and saves the resulting 3D skeletons.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/run_multiview_pipeline_shelf.py
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.pipeline import MultiViewPipeline


class Shelf2DEstimator:
    """Adapter that returns stored 2D predictions for a frame."""

    def __init__(self, dataset: dict):
        self.dataset = dataset

    def extract_for_frame(self, frame_idx: int):
        item = self.dataset[frame_idx]
        return {
            "keypoints_2d": item["points_2d"],
            "confidence": item["input"][..., 2],
        }


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end multi-view pipeline on Shelf.")
    parser.add_argument("--output", type=str, default="outputs/shelf_pipeline_3d.pkl")
    args = parser.parse_args()

    data_root = "tmp/voxelpose-pytorch/data/Shelf"
    loader = VoxelPoseShelfLoader(data_root)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]

    with open("outputs/shelf_matched_dataset.pkl", "rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    estimator = Shelf2DEstimator(dataset)
    pipeline = MultiViewPipeline(estimator=estimator)

    results = {}
    for frame_idx in sorted(dataset.keys()):
        det = estimator.extract_for_frame(frame_idx)
        pred_3d = pipeline.fuse_frame(det["keypoints_2d"], det["confidence"], cameras)
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
