"""DLT ensemble baseline: triangulate from random view subsets and average.

Usage:
    .venv/bin/python experiments/dlt_ensemble_shelf.py
"""

from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.pipeline import MultiViewPipeline


def triangulate_subset(points_2d, confidences, cameras, indices):
    pipeline = MultiViewPipeline(estimator=None)
    return pipeline.fuse_frame(points_2d[indices], confidences[indices], [cameras[i] for i in indices])


def main():
    loader = VoxelPoseShelfLoader("tmp/voxelpose-pytorch/data/Shelf")
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]

    with open("outputs/shelf_matched_dataset.pkl", "rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    # Try a few random 3-view subsets
    np.random.seed(42)
    n_views = len(cameras)
    view_indices = np.arange(n_views)

    all_errors = []
    for item in dataset.values():
        points_2d = item["points_2d"]
        confidences = item["input"][:, :, 2]

        preds = []
        for _ in range(10):
            subset = np.random.choice(view_indices, size=3, replace=False)
            pred = triangulate_subset(points_2d, confidences, cameras, subset)
            preds.append(pred)
        pred_3d = np.mean(preds, axis=0)

        for i, cid in enumerate(camera_ids):
            P = loader.get_camera(cid).projection_matrix
            X_h = np.hstack([pred_3d, np.ones((pred_3d.shape[0], 1))])
            x_h = (P @ X_h.T).T
            x = x_h[:, :2] / x_h[:, 2:3]
            err = np.linalg.norm(x - points_2d[i], axis=-1)
            all_errors.append(err)

    all_errors = np.concatenate(all_errors)
    print(f"DLT Ensemble (10 random 3-view subsets) - mean: {all_errors.mean():.2f}, "
          f"median: {np.median(all_errors):.2f}, max: {all_errors.max():.2f}")


if __name__ == "__main__":
    main()
