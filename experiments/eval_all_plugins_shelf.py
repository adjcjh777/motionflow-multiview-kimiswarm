"""Evaluate all FusionModule plugins on Shelf multi-view 2D keypoints.

Reports reprojection error (px) for each plugin because Shelf 3D GT is not
available locally.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/eval_all_plugins_shelf.py \
        --data_root tmp/voxelpose-pytorch/data/Shelf --frame_start 300 --frame_end 600
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.fusion import FUSION_REGISTRY
from motionflow_mv.pipeline_utils import select_best_person_group


def reprojection_error(pred_3d: np.ndarray, points_2d: np.ndarray, camera) -> np.ndarray:
    """Return per-joint reprojection error in pixels."""
    P = camera.projection_matrix
    X_h = np.hstack([pred_3d, np.ones((pred_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return np.linalg.norm(x - points_2d, axis=-1)


def load_plugin_checkpoints(device):
    """Optionally load locally trained synthetic checkpoints into plugins."""
    checkpoints = {
        "attention": "outputs/attention_fusion_synthetic_shared.pth",
        "robust_triangulation": "outputs/robust_triangulation_synthetic.pth",
        "residual_refiner": "outputs/residual_refiner_synthetic.pth",
        "temporal_refiner": "outputs/temporal_refiner_synthetic.pth",
    }
    for name, path in checkpoints.items():
        checkpoint_path = Path(path)
        if checkpoint_path.exists():
            module = FUSION_REGISTRY.get(name)
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            module.model.load_state_dict(state)
            module.model.to(device)
            module.model.eval()


def main():
    parser = argparse.ArgumentParser(description="Evaluate all fusion plugins on Shelf data.")
    parser.add_argument("--data_root", type=str, default="tmp/voxelpose-pytorch/data/Shelf")
    parser.add_argument("--frame_start", type=int, default=300)
    parser.add_argument("--frame_end", type=int, default=600)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    load_plugin_checkpoints(device)

    loader = VoxelPoseShelfLoader(args.data_root)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]

    per_plugin_errors = {name: [] for name in FUSION_REGISTRY.names()}

    for frame_idx in range(args.frame_start, args.frame_end + 1):
        frame_predictions = {cid: loader.get_frame_predictions(cid, frame_idx) for cid in camera_ids}
        if any(len(frame_predictions[cid]) == 0 for cid in camera_ids):
            continue
        try:
            _, points_2d, confidences = select_best_person_group(
                frame_predictions, loader.cameras, camera_ids
            )
        except ValueError:
            continue

        # Plugin-specific input/output scaling:
        #   - attention/robust_triangulation were trained on 2D/1000 and output
        #     positions in meters, so their output is converted back to mm.
        #   - dlt/residual_refiner/temporal_refiner operate directly in the
        #     camera-world units used by Shelf (mm) and should not be scaled.
        points_2d_px = points_2d[None]  # (1, V, J, 2)
        points_2d_norm = (points_2d / 1000.0)[None]  # (1, V, J, 2)
        confidences_batch = confidences[None]  # (1, V, J)

        for name in sorted(FUSION_REGISTRY.names()):
            module = FUSION_REGISTRY.get(name)
            try:
                if name in ("attention", "robust_triangulation"):
                    input_2d = points_2d_norm
                    output_scale = 1000.0
                else:
                    input_2d = points_2d_px
                    output_scale = 1.0
                pred_3d = module.fuse(input_2d, confidences_batch, cameras)  # (1, J, 3) or (J, 3)
                if pred_3d.ndim == 3:
                    pred_3d = pred_3d[0]
                pred_3d_mm = pred_3d * output_scale
                for i, cid in enumerate(camera_ids):
                    err = reprojection_error(pred_3d_mm, points_2d[i], loader.get_camera(cid))
                    per_plugin_errors[name].append(err)
            except Exception as e:
                print(f"Plugin {name} failed on frame {frame_idx}: {e}")

    print("\nShelf reprojection error (px)")
    print(f"{'Plugin':<20} {'Mean':>10} {'Median':>10} {'Max':>10}")
    print("-" * 52)
    for name in sorted(FUSION_REGISTRY.names()):
        if per_plugin_errors[name]:
            errors = np.concatenate(per_plugin_errors[name])
            print(f"{name:<20} {errors.mean():>10.2f} {np.median(errors):>10.2f} {errors.max():>10.2f}")
        else:
            print(f"{name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")


if __name__ == "__main__":
    main()
