#!/usr/bin/env python
"""Export the corrected H36M true-GT npz files to a common baseline format.

The exported pickle contains:

    {
        "train": [
            {
                "subject": int,
                "actions": List[int],
                "points_2d":  (F, V, J, 2) np.ndarray,
                "confidences": (F, V, J)    np.ndarray,
                "joints_3d":   (F, J, 3)    np.ndarray,
                "cameras": [
                    {"K": (3,3), "R": (3,3), "t": (3,)}, ...
                ],
            },
            ...
        ],
        "val": [...],
        "joint_names": List[str],
        "dataset": "h36m_true_gt",
    }

This format is consumed by the VoxelPose and MVPose prep scripts in
scripts/sota_baselines/.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


# H36M true-GT skeleton uses 17 joints (matching our canonical npz).
H36M_JOINT_NAMES = [
    "Hip",
    "RHip",
    "RKnee",
    "RFoot",
    "LHip",
    "LKnee",
    "LFoot",
    "Spine",
    "Thorax",
    "Neck",
    "Head",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RShoulder",
    "RElbow",
    "RWrist",
]


def load_split_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_subject_from_stem(stem: str) -> int:
    """Infer subject number from filename like s_09_acts_..._multiview_m."""
    parts = stem.split("_")
    return int(parts[1])


def extract_actions_from_stem(stem: str) -> List[int]:
    """Infer action numbers from filename like s_09_acts_02_03_..._multiview_m."""
    parts = stem.split("_")
    # Tokens after "acts" are action numbers; the trailing "multiview_m" is not.
    try:
        idx = parts.index("acts")
    except ValueError:
        return []
    actions: List[int] = []
    for p in parts[idx + 1 :]:
        if p == "multiview" or p == "multiview_m" or p == "true":
            break
        try:
            actions.append(int(p))
        except ValueError:
            break
    return actions


def export_sequence(npz_path: Path) -> Dict[str, Any]:
    data = np.load(npz_path)

    points_2d = data["points_2d"]  # (F, V, J, 2)
    confidences = data["confidences"]  # (F, V, J)
    joints_3d = data["joints_3d"]  # (F, J, 3)
    camera_K = data["camera_K"]  # (V, 3, 3)
    camera_R = data["camera_R"]  # (V, 3, 3)
    camera_t = data["camera_t"]  # (V, 3)

    cameras = [
        {"K": camera_K[v].astype(np.float64), "R": camera_R[v].astype(np.float64), "t": camera_t[v].astype(np.float64)}
        for v in range(camera_K.shape[0])
    ]

    stem = npz_path.stem
    return {
        "subject": extract_subject_from_stem(stem),
        "actions": extract_actions_from_stem(stem),
        "points_2d": points_2d.astype(np.float32),
        "confidences": confidences.astype(np.float32),
        "joints_3d": joints_3d.astype(np.float64),
        "cameras": cameras,
        "source_path": str(npz_path),
    }


def export_all(
    split_yaml: Path,
    output_pkl: Path,
    joint_names: List[str],
) -> None:
    split = load_split_yaml(split_yaml)

    result: Dict[str, Any] = {
        "dataset": "h36m_true_gt",
        "joint_names": joint_names,
        "train": [],
        "val": [],
    }

    for key in ("train_paths", "val_paths"):
        out_key = "train" if key == "train_paths" else "val"
        for path_str in split.get(key, []) or []:
            npz_path = Path(path_str)
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing {npz_path}; run make_h36m_true_gt_metres.py first.")
            print(f"Exporting {out_key}: {npz_path}")
            result[out_key].append(export_sequence(npz_path))

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {output_pkl}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split_yaml",
        type=Path,
        default=Path("configs/splits/h36m_true_gt_standard.yaml"),
        help="Path to the H36M true-GT split YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/sota_baselines/h36m_true_gt_baseline_format.pkl"),
        help="Output pickle path.",
    )
    args = parser.parse_args()
    export_all(args.split_yaml, args.output, H36M_JOINT_NAMES)


if __name__ == "__main__":
    main()
