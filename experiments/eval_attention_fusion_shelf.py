"""Evaluate a trained AttentionFusion model on Shelf/VoxelPose data.

Compare its reprojection error against the matched 2D observations.
By default loads precomputed matched frames from `outputs/shelf_matched_dataset.pkl`;
falls back to on-the-fly matching if not available.

Usage:
    .venv/bin/python experiments/eval_attention_fusion_shelf.py \
        --data_root data/Shelf \
        --checkpoint outputs/attention_fusion_shelf.pth \
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
from motionflow_mv.fusion.attention_model import AttentionFusionModel
from motionflow_mv.pipeline_utils import select_best_person_group


def reprojection_error(pred_3d: np.ndarray, points_2d: np.ndarray, camera) -> np.ndarray:
    """Return per-joint reprojection error in pixels."""
    P = camera.projection_matrix
    X_h = np.hstack([pred_3d, np.ones((pred_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return np.linalg.norm(x - points_2d, axis=-1)


def load_precomputed_dataset(pkl_path: Path, frame_start: int, frame_end: int):
    """Return dict {frame_idx: {'input': (V,J,3), 'target_3d': (J,3), 'points_2d': (V,J,2)}}."""
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return {k: v for k, v in data.items() if frame_start <= k <= frame_end}


def main():
    parser = argparse.ArgumentParser(description="Evaluate AttentionFusion on Shelf data.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--frame_start", type=int, default=300)
    parser.add_argument("--frame_end", type=int, default=600)
    parser.add_argument("--d", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = VoxelPoseShelfLoader(args.data_root)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))

    precomputed = load_precomputed_dataset(
        Path("outputs/shelf_matched_dataset.pkl"), args.frame_start, args.frame_end
    )

    # Infer shape from first valid frame
    sample_input = None
    if precomputed:
        first = next(iter(precomputed.values()))
        sample_input = first["input"]
    else:
        for frame_idx in range(args.frame_start, args.frame_end + 1):
            frame_predictions = {cid: loader.get_frame_predictions(cid, frame_idx) for cid in camera_ids}
            if any(len(p) == 0 for p in frame_predictions.values()):
                continue
            try:
                _, points_2d, confidences = select_best_person_group(
                    frame_predictions, loader.cameras, camera_ids
                )
                sample_input = np.concatenate([points_2d / 1000.0, confidences[..., None]], axis=-1)
                break
            except ValueError:
                continue

    if sample_input is None:
        print("No valid frames found in the specified range.")
        return

    n_views, j, _ = sample_input.shape
    model = AttentionFusionModel(j=j, d=args.d, n_views=n_views).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    all_errors = []
    for frame_idx in range(args.frame_start, args.frame_end + 1):
        if precomputed and frame_idx in precomputed:
            item = precomputed[frame_idx]
            inp = torch.tensor(item["input"], dtype=torch.float32).unsqueeze(0).to(device)
            points_2d = item["points_2d"] if "points_2d" in item else None
        else:
            frame_predictions = {cid: loader.get_frame_predictions(cid, frame_idx) for cid in camera_ids}
            if any(len(p) == 0 for p in frame_predictions.values()):
                continue
            try:
                _, points_2d, confidences = select_best_person_group(
                    frame_predictions, loader.cameras, camera_ids
                )
            except ValueError:
                continue
            points_2d_norm = points_2d / 1000.0
            inp = torch.tensor(
                np.concatenate([points_2d_norm, confidences[..., None]], axis=-1),
                dtype=torch.float32,
            ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_3d = model(inp).squeeze(0).cpu().numpy() * 1000.0

        if points_2d is None:
            continue
        for i, cid in enumerate(camera_ids):
            err = reprojection_error(pred_3d, points_2d[i], loader.get_camera(cid))
            all_errors.append(err)

    if all_errors:
        all_errors = np.concatenate(all_errors)
        print(f"Frames evaluated: {len(precomputed) if precomputed else 'unknown'}")
        print(f"AttentionFusion reprojection error (px) - mean: {all_errors.mean():.2f}, "
              f"median: {np.median(all_errors):.2f}, max: {all_errors.max():.2f}")
    else:
        print("No frames to evaluate.")


if __name__ == "__main__":
    main()
