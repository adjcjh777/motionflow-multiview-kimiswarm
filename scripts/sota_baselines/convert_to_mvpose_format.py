#!/usr/bin/env python
"""Convert the common H36M true-GT pickle to MVPose input format.

MVPose operates on per-frame dictionaries of the form:

    {
        "subject": int,
        "actions": List[int],
        "cameras": [...],   # list of {K, R, t}
        "points_2d": (F, V, J, 2),
        "confidences": (F, V, J),
        "joints_3d": (F, J, 3),
    }

This script writes a single pickle per split and a JSON manifest that the
MVPose launcher can consume.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import yaml


def convert_common_to_mvpose(input_pkl: Path, output_dir: Path, cfg: Dict[str, Any]) -> None:
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    output_dir = Path(cfg["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        split_data = {
            "dataset": data["dataset"],
            "joint_names": data["joint_names"],
            "sequences": data[split],
        }
        out_path = output_dir / f"h36m_true_gt_{split}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote {out_path}")

    manifest = {
        "dataset": data["dataset"],
        "train": str(output_dir / "h36m_true_gt_train.pkl"),
        "val": str(output_dir / "h36m_true_gt_val.pkl"),
        "num_views": 4,
        "num_joints": 17,
        "detector": cfg["training"]["detector"],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to mvpose_h36m_config.yaml.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    input_pkl = Path(cfg["input_pkl"])
    if not input_pkl.exists():
        raise FileNotFoundError(f"Run common_export_h36m_true_gt.py first: {input_pkl}")

    convert_common_to_mvpose(input_pkl, Path(cfg["data_dir"]), cfg)


if __name__ == "__main__":
    main()
