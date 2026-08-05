"""Precompute matched Shelf/VoxelPose frames for faster training/eval.

Saves a dict: {frame_idx: {'input': (V, J, 3), 'target_3d': (J, 3)}}.
"""

import argparse
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.pipeline import MultiViewPipeline
from motionflow_mv.pipeline_utils import select_best_person_group


def main():
    parser = argparse.ArgumentParser(description="Precompute matched Shelf dataset.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--frame_start", type=int, default=300)
    parser.add_argument("--frame_end", type=int, default=600)
    parser.add_argument("--output", type=str, default="outputs/shelf_matched_dataset.pkl")
    args = parser.parse_args()

    loader = VoxelPoseShelfLoader(args.data_root)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]
    pipeline = MultiViewPipeline(estimator=None)

    dataset = {}
    for frame_idx in range(args.frame_start, args.frame_end + 1):
        frame_predictions = {cid: loader.get_frame_predictions(cid, frame_idx) for cid in camera_ids}
        if any(len(p) == 0 for p in frame_predictions.values()):
            continue
        try:
            _, points_2d, confidences = select_best_person_group(
                frame_predictions, loader.cameras, camera_ids
            )
        except ValueError:
            continue

        pred_3d = pipeline.fuse_frame(points_2d, confidences, cameras)
        points_2d_norm = points_2d / 1000.0
        inp = np.concatenate([points_2d_norm, confidences[..., None]], axis=-1)
        dataset[frame_idx] = {"input": inp, "target_3d": pred_3d, "points_2d": points_2d}

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}, total: {len(dataset)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset)} matched frames to {output_path}")


if __name__ == "__main__":
    main()
