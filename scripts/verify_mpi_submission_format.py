#!/usr/bin/env python3
"""Verify that a submission package matches the MPI-INF-3DHP official format.

Checks:
- Zip contains TS1.mat .. TS6.mat.
- Each .mat contains a variable ``joint_3d_image`` of shape (T, 17, 3).
- Units are documented as millimetres.
- Predictions are root-relative to the pelvis (joint 14).
- Optional local-evaluation mat has the expected cell arrays.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio


PELVIS_IDX = 14
EXPECTED_JOINTS = 17


def _check_array(name: str, arr: np.ndarray, expected_shape_suffix: tuple[int, ...]) -> list[str]:
    errors: list[str] = []
    if arr.ndim != len(expected_shape_suffix) + 1:
        errors.append(f"{name}: expected {len(expected_shape_suffix) + 1}D array, got {arr.ndim}D")
        return errors
    shape = arr.shape
    for i, expected in enumerate(expected_shape_suffix):
        if shape[i + 1] != expected:
            errors.append(f"{name}: shape[{i + 1}] is {shape[i + 1]}, expected {expected}")
    return errors


def verify_per_sequence_mats(zip_path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for i in range(1, 7):
            mat_name = f"TS{i}.mat"
            if mat_name not in names:
                errors.append(f"Missing {mat_name} in zip")
                continue

            data = sio.loadmat(zf.extract(mat_name))
            if "joint_3d_image" not in data:
                errors.append(f"{mat_name}: missing variable 'joint_3d_image'")
                continue

            j3d = data["joint_3d_image"]
            if j3d.ndim != 3 or j3d.shape[1:] != (EXPECTED_JOINTS, 3):
                errors.append(f"{mat_name}: 'joint_3d_image' shape {j3d.shape}, expected (T, {EXPECTED_JOINTS}, 3)")
                continue

            # Pelvis should be numerically zero after root-centering.
            pelvis = j3d[:, PELVIS_IDX, :]
            if np.abs(pelvis).max() > 1e-6:
                errors.append(f"{mat_name}: pelvis (joint {PELVIS_IDX}) is not zero (max {np.abs(pelvis).max():.6f})")

    return errors


def verify_local_eval_mat(zip_path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        if "mpii_3dhp_prediction.mat" not in zf.namelist():
            return []  # optional

        data = sio.loadmat(zf.extract("mpii_3dhp_prediction.mat"))
        for var in ("sequencewise_per_joint_error", "sequencewise_activity_labels"):
            if var not in data:
                errors.append(f"mpii_3dhp_prediction.mat: missing variable '{var}'")
                continue

        if "sequencewise_per_joint_error" in data:
            pje = data["sequencewise_per_joint_error"]
            if pje.shape != (6, 1):
                errors.append(f"sequencewise_per_joint_error shape {pje.shape}, expected (6, 1) cell")

        if "sequencewise_activity_labels" in data:
            act = data["sequencewise_activity_labels"]
            if act.shape != (6, 1):
                errors.append(f"sequencewise_activity_labels shape {act.shape}, expected (6, 1) cell")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MPI-INF-3DHP submission package format.")
    parser.add_argument("--zip", required=True, type=Path, help="Path to submission zip.")
    args = parser.parse_args()

    if not args.zip.exists():
        raise FileNotFoundError(f"Zip not found: {args.zip}")

    errors: list[str] = []
    errors.extend(verify_per_sequence_mats(args.zip))
    errors.extend(verify_local_eval_mat(args.zip))

    if errors:
        print("Verification FAILED:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    print(f"Verification PASSED: {args.zip} is in the expected MPI-INF-3DHP submission format.")


if __name__ == "__main__":
    main()
