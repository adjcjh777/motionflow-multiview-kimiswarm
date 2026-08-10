#!/usr/bin/env python3
"""Regenerate MPI-INF-3DHP canonical .npz files with detected 2D keypoints.

The canonical MPI-INF-3DHP .npz files currently store the ground-truth ``annot2``
2D keypoints (projected from ``univ_annot3``).  For the standard *detected-2D*
protocol, these GT 2D points must be replaced by actual 2D detections from an
off-the-shelf pose detector (HRNet / OpenPose / RTMPose) run on each raw camera
frame.

This script is a runnable skeleton.  It walks the existing canonical .npz files,
optionally loads the raw MPI images, runs a detector stub (replace with real
HRNet/OpenPose code), and writes new .npz files with the same structure but with
``points_2d`` and ``confidences`` coming from the detector.

Usage
-----
    # Generate detected-2D .npz files using the fallback (GT + noise) for now.
    python scripts/generate_mpi_detected_2d.py \
        --input_dir data/webbridge/mpi_inf_3dhp \
        --output_dir data/webbridge/mpi_inf_3dhp_detected_2d \
        --fallback_noise 2.0

    # Once detector integration is ready:
    python scripts/generate_mpi_detected_2d.py \
        --input_dir data/webbridge/mpi_inf_3dhp \
        --output_dir data/webbridge/mpi_inf_3dhp_detected_2d \
        --detector hrnet \
        --image_dir data/webbridge/mpi_inf_3dhp/raw

TODO
----
* Integrate a real 2D detector (HRNet, OpenPose, or RTMPose).
* Map detector joints to the MPI-INF-3DHP 28-joint skeleton.
* Decide whether to keep/merge confidence scores from the detector.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Detector interface
# ---------------------------------------------------------------------------
class DetectorStub:
    """Stub 2D detector.

    Replace this with real HRNet / OpenPose / RTMPose integration.  The expected
    interface is ``__call__(image_paths) -> (keypoints, confidences)`` where
    ``keypoints`` has shape ``(V, J, 2)`` and ``confidences`` has shape ``(V, J)``.
    """

    def __init__(self, name: str = "stub", device: str = "cpu") -> None:
        self.name = name
        self.device = device

    def __call__(self, image_paths: list, points_2d_gt: np.ndarray) -> tuple:
        """Return placeholder detections.

        Parameters
        ----------
        image_paths:
            List of image file paths for each view.  May be empty if frames are
            not available locally.
        points_2d_gt:
            Ground-truth 2D points for this frame, shape ``(V, J, 2)``.

        Returns
        -------
        points_2d:
            Detected 2D keypoints, shape ``(V, J, 2)``.
        confidences:
            Per-joint detection confidences, shape ``(V, J)``.
        """
        # TODO: replace with actual detector inference.
        # For the skeleton, fall back to GT + small noise so the script is
        # runnable and produces valid .npz files.
        return points_2d_gt, np.ones(points_2d_gt.shape[:2], dtype=np.float32)


def _build_detector(name: str, device: str = "cpu"):
    """Build a 2D detector by name."""
    name = name.lower()
    if name == "hrnet":
        # TODO: import and wrap mmpose / HRNet here.
        raise NotImplementedError(
            "HRNet detector not yet integrated. "
            "Use --detector stub or --fallback_noise instead."
        )
    if name == "openpose":
        # TODO: import and wrap OpenPose here.
        raise NotImplementedError(
            "OpenPose detector not yet integrated. "
            "Use --detector stub or --fallback_noise instead."
        )
    if name in {"stub", "none", ""}:
        return DetectorStub(name="stub", device=device)
    raise ValueError(f"Unknown detector: {name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_OMIT_RE = re.compile(r"_smoke|_v4_|test_set")


def _find_source_npz(input_dir: Path) -> list:
    """Return list of canonical .npz files to re-generate."""
    files = sorted(input_dir.glob("*.npz"))
    # Keep only full 14-view files, skip smoke / 4-view / test_set variants to
    # avoid duplicates.  Test set is handled separately.
    return [p for p in files if not _OMIT_RE.search(p.name)]


def _add_fallback_noise(points_2d: np.ndarray, noise_std: float) -> np.ndarray:
    """Add pixel noise to GT 2D points to mimic a real detector (skeleton only)."""
    if noise_std <= 0:
        return points_2d
    noise = np.random.normal(scale=noise_std, size=points_2d.shape).astype(np.float64)
    return points_2d + noise


def _generate_detected_sequence(
    input_npz: Path,
    output_npz: Path,
    detector,
    fallback_noise: float,
    image_dir: Path | None,
) -> None:
    """Create a detected-2D copy of one canonical .npz file."""
    data = dict(np.load(input_npz))
    required = {"points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"{input_npz} missing keys: {missing}")

    points_2d = data["points_2d"].astype(np.float64)
    confidences = data["confidences"].astype(np.float32)
    joints_3d = data["joints_3d"]
    T, V, J, _ = points_2d.shape

    detected_2d = np.zeros_like(points_2d)
    detected_conf = np.zeros_like(confidences)

    for t in range(T):
        # TODO: build per-view image paths from image_dir when available.
        image_paths = []
        d2, dc = detector(image_paths, points_2d[t])
        detected_2d[t] = d2
        detected_conf[t] = dc

    # Fallback / synthetic noise layer.  Once the real detector is wired in,
    # remove this branch.
    if fallback_noise > 0:
        detected_2d = _add_fallback_noise(detected_2d, fallback_noise)
        detected_conf = np.clip(detected_conf * 0.9, 0.0, 1.0).astype(np.float32)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        points_2d=detected_2d,
        confidences=detected_conf,
        joints_3d=joints_3d,
        camera_K=data["camera_K"],
        camera_R=data["camera_R"],
        camera_t=data["camera_t"],
    )

    print(f"  wrote {output_npz}  shape={detected_2d.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate MPI-INF-3DHP .npz with detected 2D keypoints."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/webbridge/mpi_inf_3dhp"),
        help="Directory containing existing canonical .npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/webbridge/mpi_inf_3dhp_detected_2d"),
        help="Where to write detected-2D .npz files.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="stub",
        help="Detector name: stub, hrnet, openpose.  Only stub is implemented.",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="Root of raw MPI images (optional; not used by stub detector).",
    )
    parser.add_argument(
        "--fallback_noise",
        type=float,
        default=2.0,
        help="Pixel noise std added to GT 2D when using stub detector.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the real detector (cuda/cpu).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = _find_source_npz(input_dir)
    if not source_files:
        raise FileNotFoundError(f"No canonical .npz files found in {input_dir}")

    detector = _build_detector(args.detector, device=args.device)

    print(f"Generating detected-2D MPI-INF-3DHP .npz files:")
    print(f"  input_dir  : {input_dir.resolve()}")
    print(f"  output_dir : {output_dir.resolve()}")
    print(f"  detector   : {args.detector}")
    print(f"  files      : {len(source_files)}")

    for input_npz in source_files:
        output_npz = output_dir / input_npz.name
        _generate_detected_sequence(
            input_npz,
            output_npz,
            detector,
            fallback_noise=args.fallback_noise,
            image_dir=args.image_dir,
        )

    print("Done. Update configs/splits/mpiinf3dhp_detected_2d.yaml to point here.")


if __name__ == "__main__":
    main()
