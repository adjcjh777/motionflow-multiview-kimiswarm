#!/usr/bin/env python3
"""Fetch / convert true Human3.6M 3D ground truth into the canonical WebBridge npz format.

This script does **not** download anything without explicit user confirmation.  It
only converts already-acquired H36M mocap data into the project's canonical
``.npz`` layout so that the true ``joints_3d`` can be plugged into
``motionflow_mv.data.webbridge_loader.convert_human36m`` via the
``--true-gt-path`` argument.

Expected source layout (one of the following)::

    data/h36m_true_gt/
      PosesD3_Positions_S1/
        MyPoseFeatures/D3_Positions/Walking.cdf
        MyPoseFeatures/D3_Positions/Discussion.cdf
        ...

    data/h36m_true_gt/
      PosesD3_Positions/
        S1/MyPoseFeatures/D3_Positions/Walking.cdf
        ...

Output::

    data/h36m_true_gt/
      s_01_act_14_true_gt.npz   # joints_3d (T, 17, 3) in mm, world coordinates
      ...

The 17 joints selected from the official 32-joint H36M skeleton are the active
subset used by the rest of this repo (indices 0-16 of the raw ``Pose`` variable).
If your CDF has a different joint order, pass ``--mapping`` with a list of 17
indices.

Examples
--------
    # Just check for data and print instructions if missing
    python scripts/fetch_h36m_true_gt.py --root data/h36m_true_gt --dry-run

    # Convert every .cdf found under the official tree
    python scripts/fetch_h36m_true_gt.py --root data/h36m_true_gt --out-dir data/h36m_true_gt
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Human3.6M action name -> id mapping used by the WebBridge preprocessing.
# ---------------------------------------------------------------------------
n_ACTION_TO_ID: Dict[str, int] = {
    "Directions": 1,
    "Discussion": 2,
    "Eating": 3,
    "Greeting": 4,
    "Phoning": 5,
    "Posing": 6,
    "Purchases": 7,
    "Sitting": 8,
    "SittingDown": 9,
    "Smoking": 10,
    "Photo": 11,
    "Waiting": 12,
    "Walking": 13,
    "WalkDog": 14,
    "WalkTogether": 15,
    "Discussion2": 16,  # legacy / alternate
}

# Default mapping: select the first 17 joints from the raw H36M 32-joint skeleton.
# This matches the active subset used by the existing WebBridge H36M .npz files.
DEFAULT_H36M_32_TO_17 = list(range(17))


# ---------------------------------------------------------------------------
# CDF loading helpers (best-effort)
# ---------------------------------------------------------------------------

SUPPORTED_BACKENDS = []

try:
    import cdflib  # type: ignore
    SUPPORTED_BACKENDS.append("cdflib")
except Exception:
    cdflib = None  # type: ignore

try:
    import netCDF4  # type: ignore
    SUPPORTED_BACKENDS.append("netCDF4")
except Exception:
    netCDF4 = None  # type: ignore

# h5py is not a CDF reader, but keep an import guard in case a future pipeline
# stores the raw mocap in HDF5 format.
try:
    import h5py  # type: ignore # noqa: F401
except Exception:
    h5py = None  # type: ignore


def _load_cdf_backend(cdf_path: Path) -> Tuple[str, np.ndarray]:
    """Read the ``Pose`` variable from a Human3.6M CDF file.

    Returns
    -------
    backend_name, pose_array where pose_array has shape (T, 96) or (T, 32, 3).
    """
    # Prefer cdflib because it is the reference backend used by VideoPose3D.
    if cdflib is not None:
        try:
            cdf = cdflib.CDF(str(cdf_path))
            # Try common variable names.
            for var_name in ("Pose", "pose", "Joint", "joint"):
                try:
                    pose = cdf.varget(var_name)
                    if pose is not None:
                        return "cdflib", np.asarray(pose)
                except Exception:
                    pass
        except Exception:
            pass

    if netCDF4 is not None:
        try:
            ds = netCDF4.Dataset(str(cdf_path), "r")
            for var_name in ("Pose", "pose", "Joint", "joint"):
                if var_name in ds.variables:
                    return "netCDF4", np.asarray(ds.variables[var_name][:])
        except Exception:
            pass

    if h5py is not None:
        try:
            with h5py.File(cdf_path, "r") as f:
                for var_name in ("Pose", "pose", "Joint", "joint"):
                    if var_name in f:
                        return "h5py", np.asarray(f[var_name][:])
        except Exception:
            pass

    raise RuntimeError(
        f"Could not read {cdf_path}. Install a CDF reader such as cdflib or netCDF4. "
        "(e.g.  pip install cdflib  )"
    )


def _reshape_pose(pose: np.ndarray) -> np.ndarray:
    """Reshape the raw CDF Pose array to (T, 32, 3)."""
    if pose.ndim == 2 and pose.shape[-1] == 96:
        return pose.reshape(-1, 32, 3)
    if pose.ndim == 2 and pose.shape[-1] == 32:
        # Some preprocessed releases already store (T, 32, 3) incorrectly squeezed?
        # Try to recover if the leading dimension is a multiple of 3.
        if pose.shape[0] % 3 == 0:
            return pose.reshape(pose.shape[0] // 3, 32, 3)
        raise ValueError(
            f"Unexpected CDF Pose shape {pose.shape}; expected (T, 96) or (T, 32, 3)"
        )
    if pose.ndim == 3 and pose.shape[-2:] == (32, 3):
        return pose
    raise ValueError(
        f"Unexpected CDF Pose shape {pose.shape}; expected (T, 96) or (T, 32, 3)"
    )


def _select_joints(pose_32: np.ndarray, mapping: Sequence[int]) -> np.ndarray:
    """Select the requested 17 joints from the 32-joint H36M pose."""
    if len(mapping) != 17:
        raise ValueError(f"Skeleton mapping must have length 17, got {len(mapping)}")
    if max(mapping) >= 32 or min(mapping) < 0:
        raise ValueError(f"Mapping indices must be in [0, 31], got {mapping}")
    return pose_32[:, list(mapping), :]


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_official_cdfs(root: Path) -> List[Path]:
    """Find all H36M D3_Positions .cdf files under *root*."""
    # Official layout after extracting the per-subject archives.
    paths = sorted(root.rglob("D3_Positions/*.cdf"))
    return paths


def find_preprocessed_npz(root: Path) -> Optional[Path]:
    """Return path to a preprocessed release npz if it exists."""
    candidates = [
        root / "data_3d_h36m.npz",
        root / "h36m.npz",
        root / "h36m_17.npz",
        root / "h36m_true_gt.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def action_name_to_id(name: str) -> Optional[int]:
    """Map a CDF action name (e.g. 'Walking') to the WebBridge action id."""
    # Strip any camera or trial suffix.
    clean = re.sub(r"(_[0-9]+)?(\.cdf)?$", "", name)
    if clean in n_ACTION_TO_ID:
        return n_ACTION_TO_ID[clean]
    # Try case-insensitive.
    lower = clean.lower()
    for k, v in n_ACTION_TO_ID.items():
        if k.lower() == lower:
            return v
    return None


def parse_subject_from_path(cdf_path: Path) -> Optional[int]:
    """Extract subject number from a path like .../S1/MyPoseFeatures/...."""
    for part in cdf_path.parts:
        m = re.match(r"^[Ss](\d+)$", part)
        if m:
            return int(m.group(1))
        # Also support PosesD3_Positions_S1 style folder names.
        m = re.match(r".*[Ss](\d+)$", part)
        if m:
            return int(m.group(1))
    return None


def convert_cdf_to_npz(
    cdf_path: Path,
    out_dir: Path,
    mapping: Sequence[int] = None,
    unit: str = "mm",
    dry_run: bool = False,
) -> Optional[Path]:
    """Convert a single H36M CDF to the canonical true-GT .npz format.

    Parameters
    ----------
    cdf_path:
        Path to the CDF file.
    out_dir:
        Directory where the .npz will be written.
    mapping:
        17 indices selecting the desired joints from the 32-joint raw skeleton.
    unit:
        ``"mm"`` or ``"m"``. The raw H36M CDF values are in millimeters.
    dry_run:
        If True, only print what would be done and return None.

    Returns
    -------
    Path to the output .npz, or None in dry-run mode.
    """
    if mapping is None:
        mapping = DEFAULT_H36M_32_TO_17

    subject = parse_subject_from_path(cdf_path)
    if subject is None:
        subject = 0

    action_name = cdf_path.stem
    action_id = action_name_to_id(action_name)
    if action_id is None:
        # Fallback to a sanitized filename.
        action_id = 0
        safe_name = re.sub(r"[^A-Za-z0-9]", "_", action_name)
        out_name = f"s_{subject:02d}_act_{action_id:02d}_{safe_name}_true_gt.npz"
    else:
        out_name = f"s_{subject:02d}_act_{action_id:02d}_true_gt.npz"

    out_path = out_dir / out_name

    if dry_run:
        print(f"  [dry-run] Would convert {cdf_path} -> {out_path}")
        return None

    backend, pose = _load_cdf_backend(cdf_path)
    pose_32 = _reshape_pose(pose)
    joints_3d = _select_joints(pose_32, mapping)

    if unit == "m":
        joints_3d = joints_3d / 1000.0
    elif unit != "mm":
        raise ValueError(f"unit must be 'mm' or 'm', got {unit!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        joints_3d=joints_3d.astype(np.float64),
        subject=subject,
        action_id=action_id,
        action_name=action_name,
        source_cdf=str(cdf_path),
        unit=unit,
    )
    print(f"  [{backend}] Converted {cdf_path} -> {out_path}  shape={joints_3d.shape}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_acquisition_instructions() -> None:
    instructions = textwrap.dedent(
        """
        Human3.6M true 3D ground-truth is not present locally.

        To obtain it:

        1. Request academic access / register at the official site:
           http://vision.imar.ro/human36m/

        2. Sign the license agreement and log in.

        3. Download the pose files (you do not need the videos for 3D GT):
             Poses -> D3 Positions -> "D3 Positions" for the desired subjects.
           The downloadables are typically named:
             Poses_D3_Positions_S1.tgz, Poses_D3_Positions_S5.tgz, ...

        4. Extract the archives under this project, e.g.:
             mkdir -p data/h36m_true_gt
             tar -xzf Poses_D3_Positions_S1.tgz -C data/h36m_true_gt
           You should end up with a layout like:
             data/h36m_true_gt/PosesD3_Positions_S1/MyPoseFeatures/D3_Positions/Walking.cdf

        5. Re-run this script:
             python scripts/fetch_h36m_true_gt.py --root data/h36m_true_gt

        References:
          - Paper: Ionescu et al., "Human3.6M: Large Scale Datasets and
            Predictive Methods for 3D Human Sensing in Natural Environments,"
            TPAMI 2014.
          - VideoPose3D preprocessing:
            https://github.com/facebookresearch/VideoPose3D/blob/main/data/prepare_data_h36m.py
        """
    )
    print(instructions)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert official Human3.6M mocap CDFs to canonical true-GT npz files."
    )
    parser.add_argument("--root", type=Path, default=Path("data/h36m_true_gt"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--unit",
        type=str,
        default="mm",
        choices=["mm", "m"],
        help="Output length unit. The raw CDF values are in millimeters.",
    )
    parser.add_argument(
        "--mapping",
        type=int,
        nargs=17,
        default=None,
        metavar="IDX",
        help="17 indices selecting joints from the H36M 32-joint raw skeleton.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be converted without writing files.",
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="No-op safety flag. This script never downloads data automatically.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root

    if args.confirm_download:
        print(
            "--confirm-download is ignored; this script only converts local data. "
            "Run with --dry-run to see instructions."
        )

    cdf_paths = find_official_cdfs(root)
    preprocessed_npz = find_preprocessed_npz(root)

    if preprocessed_npz is not None:
        print(f"Found preprocessed release: {preprocessed_npz}")

    if not cdf_paths and preprocessed_npz is None:
        print(f"No official H36M CDF files found under {root.resolve()}")
        print_acquisition_instructions()
        return 1

    print(f"Found {len(cdf_paths)} CDF file(s).")
    if not SUPPORTED_BACKENDS:
        print(
            "ERROR: No CDF reader is installed. Install cdflib or netCDF4.",
            file=sys.stderr,
        )
        print("  pip install cdflib", file=sys.stderr)
        return 1

    print(f"CDF backends available: {', '.join(SUPPORTED_BACKENDS)}")

    mapping = args.mapping if args.mapping else DEFAULT_H36M_32_TO_17
    produced: List[Path] = []
    for cdf_path in cdf_paths:
        out = convert_cdf_to_npz(
            cdf_path,
            out_dir,
            mapping=mapping,
            unit=args.unit,
            dry_run=args.dry_run,
        )
        if out is not None:
            produced.append(out)

    if args.dry_run:
        print(f"\nDry run complete. Would produce {len(cdf_paths)} .npz file(s).")
    else:
        print(f"\nProduced {len(produced)} true-GT .npz file(s) in {out_dir.resolve()}.")
        if produced:
            print("Next step: pass the desired npz to the WebBridge converter:")
            print(
                "  python -m motionflow_mv.data.webbridge_loader human36m "
                "--data_root data/h36m_hf --subject 1 --actions 2 "
                f"--true-gt-path {produced[0]} --out data/h36m_hf/s_01_act_02_multiview.npz"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
