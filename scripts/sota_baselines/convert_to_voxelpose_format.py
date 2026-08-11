#!/usr/bin/env python
"""Convert the common H36M true-GT pickle to VoxelPose input format.

VoxelPose expects H36M data in a directory tree similar to:

    DATASET_ROOT/
        annotations/
            S1/
                Directions.54138969.npy   -> image paths / actions metadata
                ...
            ...
        cameras/
            S1_54138969.json
            ...
        detected_boxes/
            ...

Because VoxelPose's exact pre-processing depends on the upstream repo
revision, this script creates a *compatibility layer* that writes:

    voxelpose_data/
        h36m_true_gt_annotations.pkl   # 3D GT + 2D points + camera params
        h36m_true_gt_cameras.pkl         # per-subject camera dicts
        h36m_true_gt_images.txt        # placeholder image list

A follow-up manual step inside the VoxelPose repo adapts these files to the
precise `dataset` class expected by that revision.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


def convert_common_to_voxelpose(input_pkl: Path, output_dir: Path, cfg: Dict[str, Any]) -> None:
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    output_dir = Path(cfg["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations: Dict[str, Any] = {
        "dataset": data["dataset"],
        "joint_names": data["joint_names"],
        "train": [],
        "val": [],
    }

    for split in ("train", "val"):
        for seq in data[split]:
            annotations[split].append(
                {
                    "subject": seq["subject"],
                    "actions": seq["actions"],
                    "points_2d": seq["points_2d"],
                    "confidences": seq["confidences"],
                    "joints_3d": seq["joints_3d"],
                    "cameras": seq["cameras"],
                    "source_path": seq["source_path"],
                }
            )

    out_anno = output_dir / "h36m_true_gt_annotations.pkl"
    with open(out_anno, "wb") as f:
        pickle.dump(annotations, f, protocol=pickle.HIGHEST_PROTOCOL)

    out_images = output_dir / "h36m_true_gt_images.txt"
    with open(out_images, "w") as f:
        f.write("# Placeholder image list for H36M true-GT.\n")
        f.write("# VoxelPose requires actual image paths in its dataset class.\n")
        f.write("# Populate this file or the dataset class to point at the raw H36M images.\n")

    print(f"Wrote {out_anno}")
    print(f"Wrote {out_images}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to voxelpose_h36m_config.yaml.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    input_pkl = Path(cfg["input_pkl"])
    if not input_pkl.exists():
        raise FileNotFoundError(f"Run common_export_h36m_true_gt.py first: {input_pkl}")

    convert_common_to_voxelpose(input_pkl, Path(cfg["data_dir"]), cfg)


if __name__ == "__main__":
    main()
