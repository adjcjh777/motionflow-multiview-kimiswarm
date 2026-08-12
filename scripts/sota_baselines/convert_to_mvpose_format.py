#!/usr/bin/env python
"""Convert the common H36M true-GT pickle to MVPose input format.

zju3dv/mvpose operates on COCO17-ordered 2D/3D joints internally. This script
therefore reorders the common baseline pickle into COCO17 order and writes a
single pickle per split plus a JSON manifest.

The upstream repo also expects per-frame dictionaries with heatmaps and cropped
images; that adapter is left for a follow-up step. The pickles produced here
contain:

    {
        "dataset": str,
        "joint_names": List[str],
        "sequences": [
            {
                "subject": int,
                "actions": List[int],
                "cameras": [...],       # list of {K, R, t}
                "points_2d": (F, V, J, 2),
                "confidences": (F, V, J),
                "joints_3d": (F, J, 3),
            },
            ...
        ],
    }
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


def _get_detector(cfg: Dict[str, Any]) -> str:
    """Return detector setting, supporting both legacy and A800 config keys."""
    if "training" in cfg:
        return cfg["training"].get("detector", "gt")
    if "inference" in cfg:
        return cfg["inference"].get("detector", "gt")
    return "gt"


def _apply_joint_mapping(
    sequences: List[Dict[str, Any]],
    mapping_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Reorder H36M 17 joints to the target joint order (e.g. COCO17).

    Missing target joints (e.g. facial keypoints) are filled from the mapped
    source joint but marked with zero confidence so the upstream pictorial
    model can ignore them.
    """
    map_by_index = mapping_cfg["map_by_index"]
    face_indices = set(mapping_cfg.get("face_indices", []))
    target_names = mapping_cfg["names"]
    source_names = None

    reordered = []
    for seq in sequences:
        pts2d = seq["points_2d"]  # (F, V, J, 2)
        confs = seq["confidences"]  # (F, V, J)
        j3d = seq["joints_3d"]  # (F, J, 3)
        F, V = pts2d.shape[:2]
        J_out = len(target_names)

        new_pts2d = np.zeros((F, V, J_out, 2), dtype=pts2d.dtype)
        new_confs = np.zeros((F, V, J_out), dtype=confs.dtype)
        new_j3d = np.zeros((F, J_out, 3), dtype=j3d.dtype)

        for out_idx, src_idx in enumerate(map_by_index):
            if src_idx < 0:
                # Missing joint (should not happen with current map).
                continue
            new_pts2d[:, :, out_idx] = pts2d[:, :, src_idx]
            new_j3d[:, out_idx] = j3d[:, src_idx]
            if out_idx in face_indices:
                # Face joints are not present in H36M; mark as absent.
                new_confs[:, :, out_idx] = 0.0
            else:
                new_confs[:, :, out_idx] = confs[:, :, src_idx]

        # Fill any remaining missing entries with the head position and zero conf.
        for out_idx in face_indices:
            if out_idx < len(map_by_index) and map_by_index[out_idx] >= 0:
                continue
            head_idx = 10  # Head in H36M order
            new_pts2d[:, :, out_idx] = pts2d[:, :, head_idx]
            new_j3d[:, out_idx] = j3d[:, head_idx]
            new_confs[:, :, out_idx] = 0.0

        if source_names is None:
            source_names = seq.get("joint_names")

        reordered.append(
            {
                "subject": seq["subject"],
                "actions": seq["actions"],
                "cameras": seq["cameras"],
                "points_2d": new_pts2d,
                "confidences": new_confs,
                "joints_3d": new_j3d,
                "source_path": seq.get("source_path"),
                "source_joint_names": source_names,
            }
        )
    return reordered


def convert_common_to_mvpose(input_pkl: Path, output_dir: Path, cfg: Dict[str, Any]) -> None:
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    output_dir = Path(cfg["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    joint_names = data["joint_names"]
    sequences = data["train"] + data["val"]

    if "joint_mapping" in cfg:
        sequences = _apply_joint_mapping(sequences, cfg["joint_mapping"])
        joint_names = cfg["joint_mapping"]["names"]

    # Split back into train/val by matching the original counts.
    n_train = len(data["train"])
    split_dict = {
        "dataset": data["dataset"],
        "joint_names": joint_names,
        "train": sequences[:n_train],
        "val": sequences[n_train:],
    }

    for split in ("train", "val"):
        out_path = output_dir / f"h36m_true_gt_{split}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "dataset": split_dict["dataset"],
                    "joint_names": split_dict["joint_names"],
                    "sequences": split_dict[split],
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Wrote {out_path}")

    manifest = {
        "dataset": data["dataset"],
        "train": str(output_dir / "h36m_true_gt_train.pkl"),
        "val": str(output_dir / "h36m_true_gt_val.pkl"),
        "num_views": cfg.get("inference", cfg.get("training", {})).get("num_views", 4),
        "num_joints": len(joint_names),
        "detector": _get_detector(cfg),
        "joint_format": "coco17" if "joint_mapping" in cfg else "h36m",
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to mvpose_h36m config YAML.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    input_pkl = Path(cfg["input_pkl"])
    if not input_pkl.exists():
        raise FileNotFoundError(f"Run common_export_h36m_true_gt.py first: {input_pkl}")

    convert_common_to_mvpose(input_pkl, Path(cfg["data_dir"]), cfg)


if __name__ == "__main__":
    main()
