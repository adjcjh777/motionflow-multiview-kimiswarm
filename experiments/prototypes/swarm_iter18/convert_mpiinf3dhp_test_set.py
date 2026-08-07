"""Convert the MPI-INF-3DHP test set (TS1-TS6) to canonical multi-view .npz.

Each test sequence folder contains a single-view image sequence and an
``annot_data.mat`` file.  The .mat file stores 2D/3D annotations for that
single view.  Because the test set 3D ground truth is not meant to be
public, this converter stores only the 2D keypoints and the 14-camera
calibration (read from a reference ``camera.calibration`` file).  The
``joints_3d`` array is kept as an empty placeholder.

Output layout
-------------
For every ``TS{i}`` folder, one canonical ``.npz`` is written with keys
``points_2d``, ``confidences``, ``joints_3d``, ``camera_K``, ``camera_R``,
``camera_t``.  ``points_2d`` and ``confidences`` have shape
``(T, 14, J, 2)`` and ``(T, 14, J)`` respectively; only the annotated view
contains non-zero data, the other 13 views are zero-filled and marked as
not confident.

Usage
-----
    python experiments/prototypes/swarm_iter18/convert_mpiinf3dhp_test_set.py \
        --test_root data/webbridge/mpi_inf_3dhp/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set \
        --calib data/webbridge/mpi_inf_3dhp/raw/S1/Seq1/camera.calibration \
        --out_dir data/webbridge/mpi_inf_3dhp/test_set

Author: research swarm (iter 18)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# Allow importing from the project root regardless of cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # experiments/prototypes/swarm_iter18 -> project root
sys.path.insert(0, str(PROJECT_ROOT))

from motionflow_mv.calibration.camera import Camera


def _parse_camera_calibration(calib_path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Parse intrinsics and extrinsics from an MPI-INF-3DHP camera.calibration file.

    Returns:
        Tuple of (K_list, R_list, t_list) each with one entry per camera.
        ``K`` is (3, 3), ``R`` is (3, 3), ``t`` is (3,).
    """
    text = calib_path.read_text()
    blocks = re.split(r"\nname\s+\d+", text)
    K_list: list[np.ndarray] = []
    R_list: list[np.ndarray] = []
    t_list: list[np.ndarray] = []
    for block in blocks[1:]:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        intrinsic, extrinsic = None, None
        for line in lines:
            if line.startswith("intrinsic"):
                intrinsic = np.fromstring(line.split("intrinsic", 1)[1], sep=" ")
            elif line.startswith("extrinsic"):
                extrinsic = np.fromstring(line.split("extrinsic", 1)[1], sep=" ")
        if intrinsic is None or extrinsic is None:
            continue
        K4 = intrinsic.reshape(4, 4)
        K_list.append(K4[:3, :3].astype(np.float64))
        E4 = extrinsic.reshape(4, 4)
        R_list.append(E4[:3, :3].astype(np.float64))
        t_list.append(E4[:3, 3].astype(np.float64))
    return K_list, R_list, t_list


def _load_test_annotations(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load 2D keypoints and valid-frame mask from an ``annot_data.mat`` file.

    The test set .mat is an HDF5 (v7.3) file with shape ``(T, 1, J, 2)``
    for the single annotated view.

    Returns:
        points_2d: (T, J, 2) array of image-space keypoints.
        valid: (T,) boolean array indicating valid frames.
    """
    import h5py

    with h5py.File(mat_path, "r") as f:
        annot2 = f["annot2"][:]  # (T, 1, J, 2)
        valid = f["valid_frame"][:, 0].astype(bool)  # (T,)
    points_2d = annot2[:, 0, :, :].astype(np.float64)
    return points_2d, valid


def convert_test_sequence(
    ts_dir: Path,
    calib_path: Path,
    out_path: Path,
    camera_index: int = 0,
) -> Path:
    """Convert a single ``TS*`` folder to a canonical .npz file.

    Args:
        ts_dir: Path to the ``TS{i}`` folder.
        calib_path: Path to a reference ``camera.calibration`` file holding
            the standard 14-camera rig.
        out_path: Destination .npz path.
        camera_index: Which view slot (0-13) the single annotated view is
            placed into. Defaults to 0 because the test set release does not
            label which physical camera each sequence corresponds to.

    Returns:
        Path to the written .npz file.
    """
    mat_path = ts_dir / "annot_data.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Test annotation file not found: {mat_path}")

    K_list, R_list, t_list = _parse_camera_calibration(calib_path)
    if len(K_list) != 14:
        raise ValueError(
            f"Expected 14 cameras in calibration, found {len(K_list)}: {calib_path}"
        )

    points_2d_single, valid = _load_test_annotations(mat_path)
    T, J, _ = points_2d_single.shape

    points_2d = np.zeros((T, 14, J, 2), dtype=np.float64)
    points_2d[:, camera_index, :, :] = points_2d_single

    confidences = np.zeros((T, 14, J), dtype=np.float32)
    confidences[:, camera_index, :] = valid[:, None]

    # Public 3D GT is not included in the canonical test-set .npz.
    joints_3d = np.zeros((T, J, 3), dtype=np.float64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=np.stack(K_list, axis=0),
        camera_R=np.stack(R_list, axis=0),
        camera_t=np.stack(t_list, axis=0),
    )
    return out_path


def convert_mpiinf3dhp_test_set(
    test_root: Path,
    calib_path: Path,
    out_dir: Path,
    camera_index: int = 0,
) -> list[Path]:
    """Convert all locally available TS1-TS6 folders to canonical .npz files.

    Args:
        test_root: Path to the folder containing ``TS1``...``TS6``.
        calib_path: Reference ``camera.calibration`` file.
        out_dir: Destination directory for the .npz files.
        camera_index: View slot (0-13) for the single annotated view.

    Returns:
        List of paths to the written .npz files.
    """
    out_paths: list[Path] = []
    for i in range(1, 7):
        ts_dir = test_root / f"TS{i}"
        if not ts_dir.exists():
            print(f"Skipping TS{i}: folder not found at {ts_dir}")
            continue
        out_path = out_dir / f"TS{i}_v14_multiview.npz"
        print(f"Converting TS{i} -> {out_path} ...")
        convert_test_sequence(ts_dir, calib_path, out_path, camera_index=camera_index)
        out_paths.append(out_path)
    return out_paths


def main():
    parser = argparse.ArgumentParser(
        description="Convert MPI-INF-3DHP test set (TS1-TS6) to canonical .npz."
    )
    parser.add_argument(
        "--test_root",
        type=Path,
        default=Path(
            "data/webbridge/mpi_inf_3dhp/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set"
        ),
        help="Path to the folder containing TS1..TS6 (default: standard project location).",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=Path("data/webbridge/mpi_inf_3dhp/raw/S1/Seq1/camera.calibration"),
        help="Reference camera.calibration file for the 14-camera rig.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/webbridge/mpi_inf_3dhp/test_set"),
        help="Where to write the per-TS .npz files.",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="View slot (0-13) to place the single annotated view into.",
    )
    args = parser.parse_args()

    if not args.test_root.exists():
        raise FileNotFoundError(f"Test root not found: {args.test_root}")
    if not args.calib.exists():
        raise FileNotFoundError(
            f"Reference camera calibration not found: {args.calib}\n"
            "Provide a camera.calibration from any MPI-INF-3DHP training sequence."
        )

    out_paths = convert_mpiinf3dhp_test_set(
        args.test_root, args.calib, args.out_dir, camera_index=args.camera_index
    )
    print("Done. Converted files:")
    for p in out_paths:
        print(" ", p)


if __name__ == "__main__":
    main()
