"""Download and convert a CMU Panoptic sequence to the canonical WebBridge .npz.

This script is a self-contained pipeline for a *single* Panoptic sequence:

  1. Downloads ``calibration_<seq>.json``.
  2. Downloads ``hdPose3d_stage1_coco19.tar`` (3D body keypoints, COCO19 skeleton).
  3. Extracts the per-frame JSON pose files.
  4. Projects the 3D keypoints into each selected HD camera to obtain 2D points.
  5. Writes a canonical ``.npz`` with the arrays expected by the ray-attention
     training pipeline.

Only the calibration and the 3D pose tarball are downloaded; the actual HD
videos are skipped because they are large and not required to produce a
canonical multi-view pose ``.npz``.

Example
-------
    python experiments/convert_panoptic_v1.py \
        --sequence 171204_pose1_sample \
        --out_dir data/webbridge/panoptic \
        --n_views 4

Dependencies
------------
    numpy (already used by the project)
"""

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera


BASE_URL = "http://domedb.perception.cs.cmu.edu/webdata/dataset"


COCO19_JOINT_NAMES = [
    "Neck",
    "Nose",
    "BodyCenter",
    "lShoulder",
    "lElbow",
    "lWrist",
    "lHip",
    "lKnee",
    "lAnkle",
    "rShoulder",
    "rElbow",
    "rWrist",
    "rHip",
    "rKnee",
    "rAnkle",
    "lEye",
    "lEar",
    "rEye",
    "rEar",
]


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* if it does not already exist."""
    if dest.exists():
        print(f"  exists {dest}")
        return
    print(f"  downloading {url}\n       -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  saved {dest} ({dest.stat().st_size} bytes)")


def _load_cameras(calibration_path: Path, n_views: int | None = None) -> List[Camera]:
    """Load the first *n_views* HD cameras from a Panoptic calibration file."""
    with open(calibration_path) as f:
        calib = json.load(f)

    hd_cams = [cam for cam in calib["cameras"] if cam.get("type") == "hd"]
    if not hd_cams:
        raise ValueError(f"No HD cameras found in {calibration_path}")

    selected = hd_cams if n_views is None else hd_cams[:n_views]
    cameras = []
    for cam in selected:
        K = np.array(cam["K"], dtype=np.float64)
        R = np.array(cam["R"], dtype=np.float64)
        # Panoptic calibration stores translations in centimeters. The canonical
        # format uses meters to match the rest of the project.
        t = np.array(cam["t"], dtype=np.float64).reshape(3) / 100.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def _project_points(joints3d_m: np.ndarray, cameras: List[Camera]) -> np.ndarray:
    """Project (J, 3) world points into each camera; returns (V, J, 2)."""
    V = len(cameras)
    J = joints3d_m.shape[0]
    points_2d = np.zeros((V, J, 2), dtype=np.float64)
    for i, cam in enumerate(cameras):
        P = cam.projection_matrix
        X_h = np.hstack([joints3d_m, np.ones((J, 1))])
        x = (P @ X_h.T).T
        x = x[:, :2] / x[:, 2:3]
        points_2d[i] = x
    return points_2d


def _parse_pose_frame(frame_path: Path, person_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return joints3d (J, 3) and confidence (J,) for one body in a Panoptic frame.

    The JSON stores ``joints19`` as a flat array of ``[x, y, z, conf]`` for each
    of the 19 COCO keypoints.  Units are converted from centimeters to meters.
    """
    with open(frame_path) as f:
        data = json.load(f)

    bodies = data.get("bodies", [])
    if not bodies:
        raise ValueError(f"No bodies in {frame_path}")

    body = bodies[person_id]
    joints19 = np.array(body["joints19"], dtype=np.float64).reshape(-1, 4)
    joints3d = joints19[:, :3] / 100.0  # Panoptic stores 3D keypoints in cm.
    conf = joints19[:, 3]
    return joints3d, conf


def _collect_frames(
    pose_dir: Path,
    cameras: List[Camera],
    person_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect all frames from *pose_dir* and return canonical arrays."""
    frames = sorted(pose_dir.glob("body3DScene_*.json"))
    if not frames:
        raise FileNotFoundError(f"No body3DScene JSON files found in {pose_dir}")

    points_2d_list = []
    conf_list = []
    joints_3d_list = []

    for frame_path in frames:
        joints3d, conf = _parse_pose_frame(frame_path, person_id=person_id)
        points_2d = _project_points(joints3d, cameras)

        points_2d_list.append(points_2d)
        conf_list.append(conf)
        joints_3d_list.append(joints3d)

    points_2d = np.stack(points_2d_list, axis=0)  # (T, V, J, 2)
    confidences = np.tile(
        np.stack(conf_list, axis=0)[:, None, :],
        (1, len(cameras), 1),
    )  # (T, V, J)
    joints_3d = np.stack(joints_3d_list, axis=0)  # (T, J, 3)

    return points_2d, confidences, joints_3d


def _save_canonical_npz(
    out_path: Path,
    points_2d: np.ndarray,
    confidences: np.ndarray,
    joints_3d: np.ndarray,
    cameras: List[Camera],
) -> None:
    """Save arrays in the canonical multi-view ``.npz`` format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=np.stack([cam.K for cam in cameras], axis=0),
        camera_R=np.stack([cam.R for cam in cameras], axis=0),
        camera_t=np.stack([cam.t for cam in cameras], axis=0),
    )


def download_panoptic_sequence(
    sequence: str,
    out_dir: Path,
) -> Path:
    """Download calibration and 3D pose for a Panoptic sequence.

    Returns the path to the per-sequence working directory.
    """
    seq_dir = Path(out_dir) / "panoptic" / sequence
    seq_dir.mkdir(parents=True, exist_ok=True)

    calib_url = f"{BASE_URL}/{sequence}/calibration_{sequence}.json"
    calib_path = seq_dir / f"calibration_{sequence}.json"

    pose_url = f"{BASE_URL}/{sequence}/hdPose3d_stage1_coco19.tar"
    pose_tar_path = seq_dir / "hdPose3d_stage1_coco19.tar"

    print(f"[CMU Panoptic] sequence: {sequence}")
    _download(calib_url, calib_path)
    _download(pose_url, pose_tar_path)

    pose_dir = seq_dir / "hdPose3d_stage1_coco19"
    if not any(pose_dir.glob("body3DScene_*.json")):
        print(f"  extracting {pose_tar_path.name}")
        with tarfile.open(pose_tar_path, "r") as tar:
            tar.extractall(path=seq_dir)
    else:
        print(f"  already extracted: {pose_dir}")

    return seq_dir


def convert_panoptic_sequence(
    seq_dir: Path,
    n_views: int | None = None,
    person_id: int = 0,
) -> Path:
    """Convert an already-downloaded Panoptic sequence to canonical .npz."""
    sequence = seq_dir.name
    calibration_path = seq_dir / f"calibration_{sequence}.json"
    pose_dir = seq_dir / "hdPose3d_stage1_coco19"
    out_path = seq_dir / f"{sequence}_canonical.npz"

    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration not found: {calibration_path}")
    if not pose_dir.exists():
        raise FileNotFoundError(f"Pose directory not found: {pose_dir}")

    cameras = _load_cameras(calibration_path, n_views=n_views)
    print(f"  selected {len(cameras)} HD cameras")

    points_2d, confidences, joints_3d = _collect_frames(
        pose_dir, cameras, person_id=person_id
    )
    _save_canonical_npz(out_path, points_2d, confidences, joints_3d, cameras)

    print(f"  saved {out_path}")
    print(f"    points_2d:   {points_2d.shape}")
    print(f"    confidences: {confidences.shape}")
    print(f"    joints_3d:   {joints_3d.shape}")
    print(f"    camera_K:    {cameras[0].K.shape} x {len(cameras)}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert a CMU Panoptic sequence to canonical .npz"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="171204_pose1_sample",
        help="Panoptic sequence name (default: 171204_pose1_sample).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/webbridge"),
        help="Root output directory (default: data/webbridge).",
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=None,
        help="Number of HD cameras to keep (default: all HD cameras).",
    )
    parser.add_argument(
        "--person_id",
        type=int,
        default=0,
        help="Which person to extract if multiple are present (default: 0).",
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only download assets, do not convert.",
    )
    args = parser.parse_args()

    seq_dir = download_panoptic_sequence(args.sequence, args.out_dir)
    if not args.download_only:
        convert_panoptic_sequence(seq_dir, n_views=args.n_views, person_id=args.person_id)


if __name__ == "__main__":
    main()
