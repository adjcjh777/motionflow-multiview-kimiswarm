#!/usr/bin/env python3
"""Convert configs/splits/webbridge_all_train.yaml to the mixed-loader format.

The mixed loader expects `train_paths`, `train_names`, `val_paths`, `val_names`.
Names are inferred from the file paths (h36m, mpi, 3dpw, aist).
"""
from __future__ import annotations

import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "configs" / "splits" / "webbridge_all_train.yaml"
DST = ROOT / "configs" / "splits" / "webbridge_all_train_mixed.yaml"


def _infer_name(path: str) -> str:
    lower = path.lower()
    if "h36m" in lower:
        return "h36m"
    if "mpiinf3dhp" in lower or "mpi" in lower:
        return "mpi"
    if "3dpw" in lower:
        return "3dpw"
    if "aistpp" in lower or "aist" in lower:
        return "aist"
    return "unknown"


def main() -> None:
    with open(SRC, "r") as f:
        cfg = yaml.safe_load(f)

    train_paths = cfg.get("train", [])
    val_paths = cfg.get("val", [])

    out = {
        "name": "WebBridge All Datasets Mixed (17-joint padded)",
        "note": (
            "Combined H36M, MPI-INF-3DHP, 3DPW, and AIST++ in the mixed-loader format. "
            "Data is padded to 14 views and mapped to the 17-joint skeleton by the mixed loader."
        ),
        "train_paths": train_paths,
        "train_names": [_infer_name(p) for p in train_paths],
        "val_paths": val_paths,
        "val_names": [_infer_name(p) for p in val_paths],
    }

    with open(DST, "w") as f:
        yaml.dump(out, f, sort_keys=False)

    print(f"Wrote {DST} with {len(train_paths)} train / {len(val_paths)} val files.")


if __name__ == "__main__":
    main()
