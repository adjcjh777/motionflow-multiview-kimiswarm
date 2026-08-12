#!/usr/bin/env python3
"""Generate configs/splits/h36m_aistpp_mixed.yaml.

Combines:
- H36M true-GT standard protocol train split (S1, S5, S6, S7, S8)
- AIST++ train split

Eval split:
- H36M true-GT S9/S11
- AIST++ val split

Both datasets share the same 17-joint canonical skeleton, so the mixed loader
re-uses the identity skeleton map for both ``h36m`` and ``aist`` domain IDs.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs/splits/h36m_aistpp_mixed.yaml"

H36M_TRAIN_FILES = [
    "data/h36m_true_gt/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
    "data/h36m_true_gt/s_05_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
    "data/h36m_true_gt/s_06_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
    "data/h36m_true_gt/s_07_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
    "data/h36m_true_gt/s_08_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
]

H36M_VAL_FILES = [
    "data/h36m_true_gt/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
    "data/h36m_true_gt/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
]


def load_aistpp_split() -> tuple[list[str], list[str], list[str], list[str]]:
    """Return AIST++ train/val paths and names from its mixed manifest."""
    aistpp_yaml = ROOT / "configs/splits/aistpp_train_val_mixed.yaml"
    with open(aistpp_yaml) as f:
        data = yaml.safe_load(f)
    return data["train_paths"], data["train_names"], data["val_paths"], data["val_names"]


def main() -> None:
    aist_train_paths, aist_train_names, aist_val_paths, aist_val_names = load_aistpp_split()

    train_paths = H36M_TRAIN_FILES + aist_train_paths
    train_names = ["h36m"] * len(H36M_TRAIN_FILES) + aist_train_names
    val_paths = H36M_VAL_FILES + aist_val_paths
    val_names = ["h36m"] * len(H36M_VAL_FILES) + aist_val_names

    manifest = {
        "name": "H36M True GT + AIST++ Mixed (standard protocol)",
        "description": (
            "Mixed-dataset manifest combining H36M true-GT standard protocol "
            "(S1,S5-S8 train; S9/S11 eval) with AIST++ train/val. "
            "Both domains use the same 17-joint canonical skeleton."
        ),
        "train_paths": train_paths,
        "train_names": train_names,
        "val_paths": val_paths,
        "val_names": val_names,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        yaml.dump(manifest, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {OUTPUT}")
    print(f"  train: {len(train_paths)} files ({len(H36M_TRAIN_FILES)} h36m + {len(aist_train_paths)} aist)")
    print(f"  val:   {len(val_paths)} files ({len(H36M_VAL_FILES)} h36m + {len(aist_val_paths)} aist)")


if __name__ == "__main__":
    main()
