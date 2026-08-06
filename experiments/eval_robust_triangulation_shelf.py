"""Evaluate the trained RobustTriangulationModel on Shelf and compare to DLT.

Usage:
    /d/anaconda3/envs/mf/python.exe experiments/eval_robust_triangulation_shelf.py
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.fusion.robust_triangulation import RobustTriangulationModel


DATA_ROOT = "tmp/voxelpose-pytorch/data/Shelf"
PICKLE_PATH = "outputs/shelf_matched_dataset.pkl"


def reprojection_error_3d(pred_3d, points_2d, camera):
    """Compute per-joint reprojection error for a single camera.

    Args:
        pred_3d: (J, 3)
        points_2d: (J, 2) pixel coordinates
        camera: Camera object

    Returns:
        (J,) error in pixels
    """
    P = camera.projection_matrix
    X_h = np.hstack([pred_3d, np.ones((pred_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return np.linalg.norm(x - points_2d, axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RobustTriangulationModel on Shelf.")
    parser.add_argument("--checkpoint", type=str, default="outputs/robust_triangulation_shelf.pth")
    parser.add_argument("--d", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = VoxelPoseShelfLoader(DATA_ROOT)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]
    proj_matrices = torch.tensor(
        np.stack([cam.projection_matrix for cam in cameras], axis=0),
        dtype=torch.float32,
        device=device,
    )

    with open(PICKLE_PATH, "rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    # Load model
    model = RobustTriangulationModel(j=17, d=args.d, n_views=len(cameras)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    dlt_errors = []
    model_errors = []
    with torch.no_grad():
        for fid, item in dataset.items():
            points_2d = item["points_2d"]
            confidence = item["input"][..., 2]
            inp = np.concatenate([points_2d, confidence[..., None]], axis=-1)
            inp_t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(inp_t, proj_matrices).squeeze(0).cpu().numpy()

            for i, cid in enumerate(camera_ids):
                cam = loader.get_camera(cid)
                dlt_errors.append(reprojection_error_3d(item["target_3d"], points_2d[i], cam))
                model_errors.append(reprojection_error_3d(pred, points_2d[i], cam))

    dlt_errors = np.concatenate(dlt_errors)
    model_errors = np.concatenate(model_errors)

    print("Method | mean (px) | median (px) | max (px)")
    print("-" * 60)
    print(f"DLT    | {dlt_errors.mean():.2f} | {np.median(dlt_errors):.2f} | {dlt_errors.max():.2f}")
    print(f"Robust | {model_errors.mean():.2f} | {np.median(model_errors):.2f} | {model_errors.max():.2f}")


if __name__ == "__main__":
    main()
