"""Evaluate the trained TemporalRefinerModel on Shelf and compare to DLT.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/eval_temporal_refiner_shelf.py
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.fusion.temporal_refiner import TemporalRefinerModel


DATA_ROOT = "tmp/voxelpose-pytorch/data/Shelf"
PICKLE_PATH = "outputs/shelf_matched_dataset.pkl"


def reprojection_error_3d(pred_3d, points_2d, camera):
    P = camera.projection_matrix
    X_h = np.hstack([pred_3d, np.ones((pred_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return np.linalg.norm(x - points_2d, axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TemporalRefinerModel on Shelf.")
    parser.add_argument("--checkpoint", type=str, default="outputs/temporal_refiner_shelf.pth")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = VoxelPoseShelfLoader(DATA_ROOT)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))

    with open(PICKLE_PATH, "rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    model = TemporalRefinerModel(j=17, d=args.d, n_views=5, hidden=args.hidden).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    half = args.window // 2
    frames = sorted(dataset.keys())

    dlt_errors = []
    model_errors = []
    with torch.no_grad():
        for i in range(half, len(frames) - half):
            w = frames[i - half:i + half + 1]
            inputs = []
            baselines = []
            for f in w:
                item = dataset[f]
                points_2d = item["points_2d"]
                conf = item["input"][..., 2]
                inputs.append(np.concatenate([points_2d, conf[..., None]], axis=-1))
                baselines.append(item["target_3d"])
            inp = torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            baseline = torch.tensor(np.stack(baselines, axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(inp, baseline).squeeze(0).cpu().numpy()
            center_item = dataset[frames[i]]
            for j, cid in enumerate(camera_ids):
                cam = loader.get_camera(cid)
                dlt_errors.append(reprojection_error_3d(center_item["target_3d"], center_item["points_2d"][j], cam))
                model_errors.append(reprojection_error_3d(pred, center_item["points_2d"][j], cam))

    dlt_errors = np.concatenate(dlt_errors)
    model_errors = np.concatenate(model_errors)

    print("Method | mean (px) | median (px) | max (px)")
    print("-" * 60)
    print(f"DLT     | {dlt_errors.mean():.2f} | {np.median(dlt_errors):.2f} | {dlt_errors.max():.2f}")
    print(f"Temporal| {model_errors.mean():.2f} | {np.median(model_errors):.2f} | {model_errors.max():.2f}")


if __name__ == "__main__":
    main()
