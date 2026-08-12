#!/usr/bin/env python3
"""Package per-sequence 3D predictions into the official MPI-INF-3DHP .mat format.

Input
-----
npz_path : str
    Path to the project's test-set predictions .npz, with keys
    "TS1_v14_multiview" .. "TS6_v14_multiview", each shape (T, 17, 3) in metres.

test_root : str
    Path to the downloaded MPI-INF-3DHP test set root (contains TS1..TS6).

output_mat : str
    Destination .mat file (v7.2 / older, not v7.3).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


PELVIS_RELEVANT_IDX = 14  # 0-based pelvis in the 17-joint "relevant" skeleton


def load_test_annotations(mat_path: Path) -> tuple:
    """Return (univ_annot3, valid_frame, activity_annotation)."""
    try:
        import h5py

        with h5py.File(mat_path, "r") as f:
            # (3, 17, 1, T) in mm
            univ_annot3 = f["univ_annot3"][:]
            valid_frame = f["valid_frame"][:, 0].astype(bool)
            activity_annotation = f["activity_annotation"][:, 0].astype(int)
    except Exception:
        from scipy.io import loadmat

        data = loadmat(mat_path)
        univ_annot3 = data["univ_annot3"]
        valid_frame = data["valid_frame"].astype(bool).ravel()
        activity_annotation = data["activity_annotation"].ravel().astype(int)
    return univ_annot3, valid_frame, activity_annotation


def package_submission(npz_path: str, test_root: str, output_mat: str) -> None:
    preds = np.load(npz_path)
    test_root = Path(test_root)

    per_joint_error_cells = []
    activity_label_cells = []

    for i in range(1, 7):
        key = f"TS{i}_v14_multiview"
        pred_m = preds[key]  # (T, 17, 3) in metres
        pred_mm = pred_m * 1000.0  # convert to mm

        mat_path = test_root / f"TS{i}" / "annot_data.mat"
        univ_annot3, valid_frame, activity_annotation = load_test_annotations(mat_path)

        # Ensure correct shape (T, 17, 3) in mm
        if univ_annot3.shape[0] == 3:
            # shape is (3, 17, 1, T)
            gt = univ_annot3[:, :, 0, :].transpose(2, 1, 0)
        else:
            gt = univ_annot3

        # Root-relative: subtract pelvis joint
        root = gt[:, PELVIS_RELEVANT_IDX, :][:, None, :]  # (T, 1, 3)
        gt_rel = gt - root
        pred_rel = pred_mm - root

        # Per-joint Euclidean error (mm), shape (T, 17)
        per_joint_err = np.linalg.norm(pred_rel - gt_rel, axis=-1)

        # Keep only valid frames
        per_joint_err = per_joint_err[valid_frame]  # (N_i, 17)
        activity_labels = activity_annotation[valid_frame]  # (N_i,)

        # MATLAB cell entry must be (17, 1, N_i) double
        per_joint_error_cells.append(per_joint_err.T[:, None, :])
        activity_label_cells.append(activity_labels[:, None])

    sio.savemat(
        output_mat,
        {
            "sequencewise_per_joint_error": np.array(per_joint_error_cells, dtype=object),
            "sequencewise_activity_labels": np.array(activity_label_cells, dtype=object),
        },
        do_compression=True,
    )
    print(f"Saved submission file: {output_mat}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package MPI-INF-3DHP server submission.")
    parser.add_argument("--npz", required=True, help="Path to test-set predictions .npz.")
    parser.add_argument("--test_root", required=True, help="Path to MPI test set root.")
    parser.add_argument("--output", default="outputs/mpii_3dhp_prediction.mat")
    args = parser.parse_args()

    package_submission(args.npz, args.test_root, args.output)
