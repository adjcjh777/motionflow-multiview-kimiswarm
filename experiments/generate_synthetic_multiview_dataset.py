"""Generate synthetic multi-view 3D pose training data from SMPL with AMASS augmentation.

This script is a thin command-line wrapper around the reusable
``motionflow_mv.data.synthetic_3d_dataset`` module.  It supports four camera
rig modes (``h36m``, ``mpiinf3dhp``, ``legacy``, ``random``), optional AMASS
motion clips, and a configurable augmentation pipeline.

Usage (H36M-matched cameras, mm units):
    python experiments/generate_synthetic_multiview_dataset.py \
        --n_sequences 500 --frames_per_seq 30 --output outputs/synthetic_multiview_dataset.npz

With AMASS motions:
    python experiments/generate_synthetic_multiview_dataset.py \
        --amass_root data/amass \
        --n_sequences 1000 --frames_per_seq 60 \
        --output outputs/synthetic_amass_dataset.npz
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.synthetic_3d_dataset import (
    AugmentConfig,
    generate_synthetic_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic multi-view dataset from SMPL/AMASS."
    )
    parser.add_argument("--n_sequences", type=int, default=500,
                        help="Number of sequences to generate.")
    parser.add_argument("--frames_per_seq", type=int, default=30,
                        help="Frames per sequence.")
    parser.add_argument("--n_views", type=int, default=4,
                        help="Number of calibrated views.")
    parser.add_argument("--noise_std", type=float, default=1.0,
                        help="Standard deviation of 2D Gaussian noise (pixels).")
    parser.add_argument("--occlusion_rate", type=float, default=0.1,
                        help="Fraction of joints to occlude per view.")
    parser.add_argument("--outlier_rate", type=float, default=0.02,
                        help="Fraction of joints to corrupt as outliers.")
    parser.add_argument("--outlier_scale", type=float, default=100.0,
                        help="Magnitude of outlier jumps (pixels).")
    parser.add_argument("--mirror_prob", type=float, default=0.0,
                        help="Probability of horizontally mirroring a sequence.")
    parser.add_argument("--scale_jitter", type=float, default=0.0,
                        help="Log-normal scale jitter std.")
    parser.add_argument("--camera_mode", type=str, default="h36m",
                        choices=["h36m", "mpiinf3dhp", "legacy", "random"],
                        help="Camera rig sampling mode.")
    parser.add_argument("--world_scale", type=float, default=1000.0,
                        help="Scale SMPL output by this factor (1000 for H36M mm).")
    parser.add_argument("--smpl_model_path", type=str, default="data/smpl/SMPL_NEUTRAL.pkl",
                        help="Path to the SMPL neutral model.")
    parser.add_argument("--amass_root", type=str, default=None,
                        help="Optional path to AMASS *_poses.npz files.")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str,
                        default="outputs/synthetic_multiview_dataset.npz",
                        help="Output .npz path.")
    parser.add_argument("--device", type=str, default=None,
                        help="PyTorch device (default: cuda if available).")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    augment_config = AugmentConfig(
        noise_std=args.noise_std,
        occlusion_rate=args.occlusion_rate,
        outlier_rate=args.outlier_rate,
        outlier_scale=args.outlier_scale,
        mirror_prob=args.mirror_prob,
        scale_jitter=args.scale_jitter,
    )

    output_path = generate_synthetic_dataset(
        output=args.output,
        smpl_model_path=args.smpl_model_path,
        n_sequences=args.n_sequences,
        n_frames=args.frames_per_seq,
        n_views=args.n_views,
        camera_mode=args.camera_mode,
        amass_root=args.amass_root,
        augment_config=augment_config,
        world_scale=args.world_scale,
        seed=args.seed,
        device=device,
    )

    print(f"Saved synthetic dataset to {output_path}")


if __name__ == "__main__":
    main()
