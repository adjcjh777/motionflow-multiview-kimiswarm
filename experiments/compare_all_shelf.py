"""Compare all trained AttentionFusion checkpoints on Shelf/VoxelPose data.

Usage:
    .venv/bin/python experiments/compare_all_shelf.py
"""

from pathlib import Path
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.fusion.attention_model import AttentionFusionModel
from motionflow_mv.fusion.robust_triangulation import RobustTriangulationModel


def reprojection_error(pred_3d: np.ndarray, points_2d: np.ndarray, camera) -> np.ndarray:
    P = camera.projection_matrix
    X_h = np.hstack([pred_3d, np.ones((pred_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return np.linalg.norm(x - points_2d, axis=-1)


def main():
    data_root = "tmp/voxelpose-pytorch/data/Shelf"
    loader = VoxelPoseShelfLoader(data_root)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))

    with open("outputs/shelf_matched_dataset.pkl", "rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    print("Method | mean (px) | median (px) | max (px)")
    print("-" * 60)

    # DLT baseline from dataset target_3d
    all_errors = []
    for item in dataset.values():
        pred_3d = item["target_3d"]
        points_2d = item["points_2d"]
        for i, cid in enumerate(camera_ids):
            err = reprojection_error(pred_3d, points_2d[i], loader.get_camera(cid))
            all_errors.append(err)
    all_errors_arr = np.concatenate(all_errors)
    print(f"DLT | {all_errors_arr.mean():.2f} | {np.median(all_errors_arr):.2f} | {all_errors_arr.max():.2f}")

    # Evaluate trained models
    checkpoints = {
        "attention_fusion_shelf.pth": 64,
        "attention_fusion_shelf_ft.pth": 32,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ckpt_name, d in checkpoints.items():
        ckpt_path = Path("outputs") / ckpt_name
        if not ckpt_path.exists():
            continue
        model = AttentionFusionModel(j=17, d=d, n_views=5).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        all_errors = []
        with torch.no_grad():
            for item in dataset.values():
                inp = torch.tensor(item["input"], dtype=torch.float32).unsqueeze(0).to(device)
                pred_3d = model(inp).squeeze(0).cpu().numpy() * 1000.0
                points_2d = item["points_2d"]
                for i, cid in enumerate(camera_ids):
                    err = reprojection_error(pred_3d, points_2d[i], loader.get_camera(cid))
                    all_errors.append(err)
        all_errors_arr = np.concatenate(all_errors)
        print(f"{ckpt_name} | {all_errors_arr.mean():.2f} | {np.median(all_errors_arr):.2f} | {all_errors_arr.max():.2f}")

    # RobustTriangulationModel
    proj_matrices = torch.tensor(
        np.stack([loader.get_camera(cid).projection_matrix for cid in camera_ids], axis=0),
        dtype=torch.float32,
        device=device,
    )
    robust_path = Path("outputs") / "robust_triangulation_shelf.pth"
    if robust_path.exists():
        model = RobustTriangulationModel(j=17, d=32, n_views=5).to(device)
        model.load_state_dict(torch.load(robust_path, map_location=device, weights_only=True))
        model.eval()
        all_errors = []
        with torch.no_grad():
            for item in dataset.values():
                points_2d = item["points_2d"]
                confidence = item["input"][..., 2]
                inp = np.concatenate([points_2d, confidence[..., None]], axis=-1)
                inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)
                pred_3d = model(inp, proj_matrices).squeeze(0).cpu().numpy()
                for i, cid in enumerate(camera_ids):
                    err = reprojection_error(pred_3d, points_2d[i], loader.get_camera(cid))
                    all_errors.append(err)
        all_errors_arr = np.concatenate(all_errors)
        print(f"robust_triangulation_shelf.pth | {all_errors_arr.mean():.2f} | {np.median(all_errors_arr):.2f} | {all_errors_arr.max():.2f}")


if __name__ == "__main__":
    main()
