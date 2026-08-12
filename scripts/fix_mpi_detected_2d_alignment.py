#!/usr/bin/env python3
"""Apply a coordinate transformation to align MPI-INF-3DHP detected 2D with mocap labels.

Background
----------
The MPI-INF-3DHP real-detected-2D .npz files produced by
``scripts/generate_mpi_detected_2d_from_avi.py`` use MediaPipe Pose on the raw
AVI frames.  On some sequences the detected keypoints land in a different
2-D coordinate frame than the canonical ``annot2`` ground-truth keypoints, so
DLT triangulation on the detected points yields very large MPJPE
(~400–2800 mm).  This script estimates and applies the affine 2-D transform
(scaling + optional flip/offset) that maps the detected points back into the
same frame as the mocap-projected ground truth.

Modes
-----
* ``diagnose``  – load a pair of canonical-GT and detected-2D .npz files and
  report the best-fit affine transform (scale/flip/offset) and the resulting
  reprojection / 2-D alignment error.  No files are modified.
* ``apply``     – read detected-2D .npz files, apply a user-specified affine
  transform, and write corrected .npz files to ``--output_dir``.

The transform is parameterised as

    x' =  scale_x * (x - cx) + flip_x * (cx - x) + offset_x
    y' =  scale_y * (y - cy) + flip_y * (cy - y) + offset_y

which covers the common failure modes found so far:

* resolution scaling between the raw AVI frame and the canonical ``annot2``
  coordinate system,
* a vertical or horizontal image flip,
* an origin / principal-point offset.

The script can also take an explicit 2x2 matrix + translation vector for
arbitrary affine transforms.

Examples
--------
    # Diagnose one file pair
    python scripts/fix_mpi_detected_2d_alignment.py diagnose \
        --canonical data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
        --detected  data/webbridge/mpi_inf_3dhp_detected_2d/s_01_seq_02_v14_multiview_m.npz

    # Apply an explicit scaling (e.g. 1920 -> 2048 in both axes)
    python scripts/fix_mpi_detected_2d_alignment.py apply \
        --input_dir  data/webbridge/mpi_inf_3dhp_detected_2d \
        --output_dir data/webbridge/mpi_inf_3dhp_detected_2d_aligned \
        --scale 1.0667 1.0667

    # Apply a vertical flip about the image centre at 2048x2048
    python scripts/fix_mpi_detected_2d_alignment.py apply \
        --input_dir  data/webbridge/mpi_inf_3dhp_detected_2d \
        --output_dir data/webbridge/mpi_inf_3dhp_detected_2d_aligned \
        --flip y --image_size 2048 2048

    # Apply an arbitrary 2x2 matrix + translation
    python scripts/fix_mpi_detected_2d_alignment.py apply \
        --input_dir  data/webbridge/mpi_inf_3dhp_detected_2d \
        --output_dir data/webbridge/mpi_inf_3dhp_detected_2d_aligned \
        --matrix 1.0,0.0,0.0,1.0 --offset 0.0,0.0

Author: research swarm (data foundation repair, 2026-08)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load a canonical .npz and return a mutable dict of arrays."""
    data = dict(np.load(path))
    return data


def _save_npz(path: Path, data: Dict[str, np.ndarray]) -> None:
    """Save a dict of arrays as .npz, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def _parse_float_pair(s: str) -> Tuple[float, float]:
    """Parse a string like '1.0,2.0' into (x, y) floats."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected two comma-separated floats, got {s!r}")
    return float(parts[0]), float(parts[1])


def _parse_matrix(s: str) -> np.ndarray:
    """Parse a 4-element comma-separated matrix into a 2x2 numpy array."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"expected 4 comma-separated floats for matrix, got {s!r}")
    return np.array([[float(parts[0]), float(parts[1])],
                     [float(parts[2]), float(parts[3])]], dtype=np.float64)


def _build_affine_transform(
    scale: Optional[Tuple[float, float]] = None,
    flip: Optional[str] = None,
    image_size: Optional[Tuple[float, float]] = None,
    offset: Optional[Tuple[float, float]] = None,
    matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a 2x2 matrix + translation that maps (x, y) -> M @ (x, y) + t.

    Parameters
    ----------
    scale:
        Optional ``(sx, sy)`` scale factors applied after removing the image
        centre (if ``image_size`` is given).
    flip:
        Optional ``"x"``, ``"y"`` or ``"both"``.  Flips are performed about the
        image centre.
    image_size:
        Optional ``(W, H)`` used as the centre for scale and flip operations.
    offset:
        Optional ``(tx, ty)`` translation added after all other transforms.
    matrix:
        Optional explicit 2x2 matrix.  If given, it overrides ``scale``.

    Returns
    -------
    M: (2, 2) transformation matrix.
    t: (2,) translation vector.
    """
    if matrix is not None:
        M = matrix.astype(np.float64)
    else:
        M = np.eye(2, dtype=np.float64)

    t = np.zeros(2, dtype=np.float64)

    if image_size is not None:
        W, H = image_size
        cx, cy = W / 2.0, H / 2.0
    else:
        cx, cy = 0.0, 0.0

    # Scale about image centre
    if scale is not None:
        sx, sy = scale
        # Equivalent to: translate(-c), scale, translate(c)
        M = np.array([[sx, 0.0], [0.0, sy]]) @ M
        t = np.array([cx - sx * cx, cy - sy * cy]) + np.array([[sx, 0.0], [0.0, sy]]) @ t

    # Flip about image centre
    if flip is not None:
        flip = flip.lower()
        fx, fy = -1.0 if flip in ("x", "both") else 1.0, -1.0 if flip in ("y", "both") else 1.0
        M = np.array([[fx, 0.0], [0.0, fy]]) @ M
        t = np.array([(1.0 - fx) * cx, (1.0 - fy) * cy]) + np.array([[fx, 0.0], [0.0, fy]]) @ t

    # Final translation
    if offset is not None:
        t = t + np.array(offset, dtype=np.float64)

    return M, t


def apply_transform(points_2d: np.ndarray, M: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply affine transform to a (..., 2) array of 2-D points.

    Parameters
    ----------
    points_2d: array of shape (..., 2).
    M: (2, 2) matrix.
    t: (2,) translation.

    Returns
    -------
    Transformed points of same shape.
    """
    orig_shape = points_2d.shape
    pts = points_2d.reshape(-1, 2)
    transformed = (M @ pts.T).T + t
    return transformed.reshape(orig_shape)


def _project_3d_to_2d(
    joints_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Project 3-D joints to 2-D using perspective projection.

    Parameters
    ----------
    joints_3d: (T, J, 3) world coordinates.
    K: (V, 3, 3) intrinsic matrices.
    R: (V, 3, 3) world-to-camera rotations.
    t: (V, 3) world-to-camera translations.

    Returns
    -------
    points_2d: (T, V, J, 2) projected 2-D coordinates.
    """
    T, J = joints_3d.shape[:2]
    V = K.shape[0]
    points = np.zeros((T, V, J, 2), dtype=np.float64)
    for v in range(V):
        X_cam = (R[v] @ joints_3d.reshape(-1, 3).T).T + t[v]
        Z = X_cam[:, 2:3]
        xy = X_cam[:, :2] / (Z + 1e-8)
        proj = (K[v][:2, :2] @ xy.T).T + K[v][:2, 2]
        points[:, v, :, :] = proj.reshape(T, J, 2)
    return points


def estimate_alignment_transform(
    detected: np.ndarray,
    reference: np.ndarray,
    confidences: np.ndarray,
    min_conf: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate the least-squares affine transform mapping detected -> reference.

    Parameters
    ----------
    detected: (T, V, J, 2) detected 2-D keypoints.
    reference: (T, V, J, 2) reference 2-D keypoints (e.g. mocap projection).
    confidences: (T, V, J) detection confidences.
    min_conf: only use keypoints with ``confidence >= min_conf``.

    Returns
    -------
    M: (2, 2) matrix.
    t: (2,) translation.
    rmse: RMSE in pixels after applying the transform.
    """
    mask = confidences >= min_conf
    src = detected[mask]
    dst = reference[mask]
    if len(src) < 3:
        raise ValueError(f"Too few confident keypoints to estimate transform: {len(src)}")

    # Solve [src, 1] * A = dst  where A is (2x3)
    X = np.hstack([src, np.ones((len(src), 1))])
    A, *_ = np.linalg.lstsq(X, dst, rcond=None)
    M = A[:2, :2].T
    t = A[2, :]

    pred = (M @ src.T).T + t
    rmse = float(np.sqrt(np.mean((pred - dst) ** 2)))
    return M, t, rmse


def diagnose_alignment(
    canonical_path: Path,
    detected_path: Path,
    min_conf: float = 0.1,
) -> Dict[str, object]:
    """Compare detected 2-D to the mocap-projected reference and report transforms.

    Returns a dict with diagnostic statistics.
    """
    can = _load_npz(canonical_path)
    det = _load_npz(detected_path)

    required = {"points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t"}
    for data, label in ((can, "canonical"), (det, "detected")):
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"{label} .npz missing keys: {missing}")

    can_points = can["points_2d"]
    det_points = det["points_2d"]
    det_conf = det["confidences"]

    # Project true 3D using stored cameras as the reference 2-D frame.
    ref_points = _project_3d_to_2d(can["joints_3d"], can["camera_K"], can["camera_R"], can["camera_t"])

    # Raw detected vs projected reference (only confident detections)
    mask = det_conf >= min_conf
    raw_diff = np.abs(det_points - ref_points)
    raw_rmse = float(np.sqrt(np.mean(raw_diff[mask] ** 2)))

    # Estimate affine transform
    M, t, fit_rmse = estimate_alignment_transform(det_points, ref_points, det_conf, min_conf=min_conf)

    # Decompose into scale/flip/offset hints
    scale_x = float(np.linalg.norm(M[:, 0]))
    scale_y = float(np.linalg.norm(M[:, 1]))
    det_M = float(np.linalg.det(M))

    return {
        "canonical_path": canonical_path,
        "detected_path": detected_path,
        "n_frames": can_points.shape[0],
        "n_views": can_points.shape[1],
        "n_joints": can_points.shape[2],
        "detected_fraction": float(mask.mean()),
        "raw_rmse_px": raw_rmse,
        "fit_rmse_px": fit_rmse,
        "affine_matrix": M,
        "affine_translation": t,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "determinant": det_M,
        "suggest_flip_x": det_M < 0,
    }


def cmd_diagnose(args: argparse.Namespace) -> None:
    """Run the diagnose subcommand."""
    result = diagnose_alignment(args.canonical, args.detected, min_conf=args.min_conf)

    print("=" * 70)
    print("MPI-INF-3DHP detected-2D alignment diagnosis")
    print("=" * 70)
    print(f"  canonical : {result['canonical_path']}")
    print(f"  detected  : {result['detected_path']}")
    print(f"  shape     : T={result['n_frames']} V={result['n_views']} J={result['n_joints']}")
    print(f"  detected fraction : {result['detected_fraction']:.3%}")
    print(f"  raw RMSE (detected vs mocap projection) : {result['raw_rmse_px']:.2f} px")
    print(f"  fit RMSE after affine transform         : {result['fit_rmse_px']:.2f} px")
    print()
    print("Estimated affine transform  detected -> mocap frame:")
    print(f"  M = [[{result['affine_matrix'][0, 0]:.6f}, {result['affine_matrix'][0, 1]:.6f}],")
    print(f"       [{result['affine_matrix'][1, 0]:.6f}, {result['affine_matrix'][1, 1]:.6f}]]")
    print(f"  t = [{result['affine_translation'][0]:.6f}, {result['affine_translation'][1]:.6f}]")
    print()
    print("Hints:")
    print(f"  implied scale : x={result['scale_x']:.4f}, y={result['scale_y']:.4f}")
    print(f"  determinant   : {result['determinant']:.4f} (negative -> contains a flip)")
    if result["suggest_flip_x"]:
        print("  -> transformation appears to contain a flip (mirror)")
    print()
    print("Use the --matrix and --offset values above in `apply` mode once the")
    print("root cause is confirmed; or pass --scale/--flip/--offset equivalents.")


def cmd_apply(args: argparse.Namespace) -> None:
    """Run the apply subcommand."""
    M, t = _build_affine_transform(
        scale=args.scale,
        flip=args.flip,
        image_size=args.image_size,
        offset=args.offset,
        matrix=args.matrix,
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if input_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "input_dir and output_dir must be different to prevent overwriting source data."
        )

    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    print(f"{'DRY RUN - ' if args.dry_run else ''}Applying affine transform to {len(files)} .npz files:")
    print(f"  M = [[{M[0, 0]:.6f}, {M[0, 1]:.6f}], [{M[1, 0]:.6f}, {M[1, 1]:.6f}]]")
    print(f"  t = [{t[0]:.6f}, {t[1]:.6f}]")
    print(f"  input_dir  : {input_dir}")
    print(f"  output_dir : {output_dir}")

    if args.dry_run:
        # Validate first file only and report what would be written.
        path = files[0]
        data = _load_npz(path)
        if "points_2d" not in data:
            print(f"  skip {path.name} (no points_2d)")
        else:
            transformed = apply_transform(data["points_2d"], M, t)
            print(f"  would write {output_dir / path.name}")
            print(f"  input shape {data['points_2d'].shape}, output shape {transformed.shape}")
        print("Dry run complete; no files were written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        data = _load_npz(path)
        if "points_2d" not in data:
            print(f"  skip {path.name} (no points_2d)")
            continue

        data["points_2d"] = apply_transform(data["points_2d"], M, t)
        out_path = output_dir / path.name
        _save_npz(out_path, data)
        print(f"  wrote {out_path}")

    print("Done.  Re-run the DLT baseline on the corrected directory to verify alignment.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align MPI-INF-3DHP detected 2D keypoints with the mocap label frame."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # diagnose
    diag = subparsers.add_parser("diagnose", help="Estimate the affine transform for one file pair.")
    diag.add_argument("--canonical", type=Path, required=True,
                      help="Path to the canonical GT .npz file.")
    diag.add_argument("--detected", type=Path, required=True,
                      help="Path to the detected-2D .npz file.")
    diag.add_argument("--min-conf", type=float, default=0.1,
                      help="Confidence threshold for keypoints used in fitting (default: 0.1).")
    diag.set_defaults(func=cmd_diagnose)

    # apply
    apply_parser = subparsers.add_parser("apply", help="Apply a coordinate transform to detected-2D .npz files.")
    apply_parser.add_argument("--input_dir", type=Path, required=True,
                              help="Directory containing detected-2D .npz files.")
    apply_parser.add_argument("--output_dir", type=Path, required=True,
                              help="Where to write corrected .npz files.")
    apply_parser.add_argument("--scale", type=_parse_float_pair, default=None,
                              help="Scale factors as 'sx,sy'.")
    apply_parser.add_argument("--flip", type=str, choices=["x", "y", "both"], default=None,
                              help="Flip about the image centre.")
    apply_parser.add_argument("--image_size", type=_parse_float_pair, default=None,
                              help="Image size as 'W,H' used as the centre for scale/flip.")
    apply_parser.add_argument("--offset", type=_parse_float_pair, default=None,
                              help="Translation as 'tx,ty'.")
    apply_parser.add_argument("--matrix", type=_parse_matrix, default=None,
                              help="Explicit 2x2 matrix as 'a,b,c,d' (row-major).")
    apply_parser.add_argument("--dry_run", action="store_true",
                              help="Validate the first input file without writing any output.")
    apply_parser.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
