"""End-to-end demo of the temporal residual fusion plugin.

Loads a canonical multi-view .npz, fuses the 2D keypoints with the
``ray_attention_temporal_residual`` plugin, and reports MPJPE against the
3D ground truth.

Usage:
    D:/anaconda3/envs/mf/python.exe experiments/demo_ray_attention_temporal_residual.py \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
        --max_frames 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion import FUSION_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to canonical .npz")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.dataset)
    points_2d = data["points_2d"]
    confidences = data["confidences"]
    joints_gt = data["joints_3d"]
    cameras = [Camera(K=k, R=r, t=t) for k, r, t in zip(data["camera_K"], data["camera_R"], data["camera_t"])]

    if args.max_frames is not None:
        points_2d = points_2d[: args.max_frames]
        confidences = confidences[: args.max_frames]
        joints_gt = joints_gt[: args.max_frames]

    j = points_2d.shape[2]
    n_views = len(cameras)

    from motionflow_mv.fusion.ray_attention_temporal_residual_module import RayAttentionTemporalResidualFusionModule
    module = RayAttentionTemporalResidualFusionModule(j=j, d=64, n_views=n_views, checkpoint_path=args.checkpoint)
    module.model = module.model.to(device)

    # Process in non-overlapping clips to avoid long-sequence transformer issues.
    clip_len = 13
    preds = []
    for start in range(0, len(points_2d) - clip_len + 1, clip_len):
        end = start + clip_len
        pred_clip = module.fuse(points_2d[start:end], confidences[start:end], cameras)
        preds.append(pred_clip)
    pred = np.concatenate(preds, axis=0)
    # Trim to match GT length for the processed frames.
    n_processed = len(pred)
    joints_gt = joints_gt[:n_processed]
    err = np.linalg.norm(pred - joints_gt, axis=-1).mean()
    print(f"Fused {n_processed} frames, {j} joints, {n_views} views")
    print(f"MPJPE: {err * 1000:.2f} mm")


if __name__ == "__main__":
    main()
