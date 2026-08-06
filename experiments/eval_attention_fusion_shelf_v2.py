"""Evaluate trained AttentionFusionV2 on Shelf/VoxelPose data.

Usage:
    .venv/bin/python experiments/eval_attention_fusion_shelf_v2.py \
        --data_root data/Shelf \
        --checkpoint outputs/attention_fusion_shelf_v2.pth \
        --frame_start 300 --frame_end 600
"""

import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.fusion.attention_model_v2 import AttentionFusionModelV2


def reprojection_error(pred_3d: np.ndarray, points_2d: np.ndarray, camera) -> np.ndarray:
    P = camera.projection_matrix
    X_h = np.hstack([pred_3d, np.ones((pred_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return np.linalg.norm(x - points_2d, axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AttentionFusionV2 on Shelf data.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--frame_start", type=int, default=300)
    parser.add_argument("--frame_end", type=int, default=600)
    parser.add_argument("--d", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = VoxelPoseShelfLoader(args.data_root)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]
    proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)
    proj_flat = proj_matrices.reshape(len(cameras), 12) / 1000.0

    pkl_path = Path("outputs/shelf_matched_dataset.pkl")
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    n_views = len(cameras)
    j = 17
    model = AttentionFusionModelV2(j=j, d=args.d, n_views=n_views).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    all_errors = []
    for frame_idx in range(args.frame_start, args.frame_end + 1):
        if frame_idx not in dataset:
            continue
        item = dataset[frame_idx]
        inp = torch.tensor(item["input"], dtype=torch.float32).unsqueeze(0).to(device)
        cam = torch.tensor(proj_flat, dtype=torch.float32).unsqueeze(0).to(device)
        points_2d = item["points_2d"]

        with torch.no_grad():
            pred_3d = model(inp, cam).squeeze(0).cpu().numpy() * 1000.0

        for i, cid in enumerate(camera_ids):
            err = reprojection_error(pred_3d, points_2d[i], loader.get_camera(cid))
            all_errors.append(err)

    if all_errors:
        all_errors = np.concatenate(all_errors)
        print(f"Frames evaluated: {len(dataset)}")
        print(f"AttentionFusionV2 reprojection error (px) - mean: {all_errors.mean():.2f}, "
              f"median: {np.median(all_errors):.2f}, max: {all_errors.max():.2f}")
    else:
        print("No frames to evaluate.")


if __name__ == "__main__":
    main()
