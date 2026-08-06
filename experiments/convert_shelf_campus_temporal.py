"""Convert Shelf/Campus pseudo-GT sequences to temporal canonical WebBridge .npz clips.

The source files

    data/shelf_campus/Shelf_Seq1/pseudogt_m.npz   (3200 frames, 5 views)
    data/shelf_campus/Campus_Seq1/pseudogt_m.npz  (1423 frames, 3 views)

are already in canonical WebBridge layout (points_2d, confidences, joints_3d,
camera_K, camera_R, camera_t).  This script simply splits each sequence into
train / validation temporal canonical .npz files under
``data/webbridge/shelf_campus/`` so that the temporal training scripts can
consume them with the same loader used for MPI-INF-3DHP.

Usage
-----
    conda run -n mf python experiments/convert_shelf_campus_temporal.py
"""

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def split_sequence(npz_path: Path, train_ratio: float = 0.8):
    """Split a canonical .npz sequence into train/val along the temporal axis.

    Camera intrinsics/extrinsics are not temporal, so they are copied unchanged
    into both train and val dictionaries.
    """
    data = np.load(npz_path)
    total = data["points_2d"].shape[0]
    n_train = int(total * train_ratio)

    temporal_keys = {"points_2d", "confidences", "joints_3d"}
    camera_keys = {"camera_K", "camera_R", "camera_t"}

    train = {k: data[k][:n_train] for k in temporal_keys}
    val = {k: data[k][n_train:] for k in temporal_keys}
    for k in camera_keys:
        train[k] = data[k]
        val[k] = data[k]
    return train, val


def save_canonical(out_path: Path, data: dict):
    """Persist a canonical dict as .npz with the standard keys."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Shelf/Campus to temporal canonical WebBridge .npz splits"
    )
    parser.add_argument("--shelf_src", type=str,
                        default=str(PROJECT_ROOT / "data/shelf_campus/Shelf_Seq1/pseudogt_m.npz"))
    parser.add_argument("--campus_src", type=str,
                        default=str(PROJECT_ROOT / "data/shelf_campus/Campus_Seq1/pseudogt_m.npz"))
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--out_dir", type=str,
                        default=str(PROJECT_ROOT / "data/webbridge/shelf_campus"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shelf (5 views)
    shelf_train, shelf_val = split_sequence(Path(args.shelf_src), args.train_ratio)
    save_canonical(out_dir / "shelf_seq1_train_v5_multiview_m.npz", shelf_train)
    save_canonical(out_dir / "shelf_seq1_val_v5_multiview_m.npz", shelf_val)
    print(
        f"Shelf: train={shelf_train['points_2d'].shape[0]} frames, "
        f"val={shelf_val['points_2d'].shape[0]} frames -> {out_dir}"
    )

    # Campus (3 views)
    campus_train, campus_val = split_sequence(Path(args.campus_src), args.train_ratio)
    save_canonical(out_dir / "campus_seq1_train_v3_multiview_m.npz", campus_train)
    save_canonical(out_dir / "campus_seq1_val_v3_multiview_m.npz", campus_val)
    print(
        f"Campus: train={campus_train['points_2d'].shape[0]} frames, "
        f"val={campus_val['points_2d'].shape[0]} frames -> {out_dir}"
    )


if __name__ == "__main__":
    main()
