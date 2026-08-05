"""Loader for Shelf/Campus multi-view pose datasets.

Reads calibration.json and annotation_3d.json, constructs Camera objects, and
projects 3D ground-truth joints into each view to obtain per-view 2D
keypoints.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..calibration.camera import Camera


def load_cameras(calibration_path: Path) -> Tuple[List[Camera], List[str]]:
    """Load calibrated cameras from calibration.json.

    The calibration is assumed to contain per-camera entries with a normalized
    intrinsics matrix K and a world-to-camera transform Tw.
    """
    with open(calibration_path) as f:
        calib = json.load(f)

    camera_names = sorted(calib["cameras"].keys())
    cameras = []
    for name in camera_names:
        cam_dict = calib["cameras"][name]
        K = np.array(cam_dict["K"], dtype=np.float64)  # (3, 3), normalized
        Tw = np.array(cam_dict["Tw"], dtype=np.float64)  # (4, 4)
        R = Tw[:3, :3]
        t = Tw[:3, 3]  # t = -R @ c, consistent with our Camera convention
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras, camera_names


def _project_points(points_3d: np.ndarray, camera: Camera) -> np.ndarray:
    """Project (J, 3) points to 2D using the camera projection matrix."""
    P = camera.projection_matrix
    X_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return x


def load_3d_annotations(annotation_path: Path, person_id: int = 0):
    """Load 3D annotations for a single person.

    Returns:
        timestamps: (T,) array of frame timestamps
        joints_3d: (T, J, 3) array of 3D joint positions
        visibilities: (T, J) array of visibility scores
    """
    with open(annotation_path) as f:
        frames = json.load(f)

    timestamps = []
    joints_3d_list = []
    visibility_list = []

    for frame in frames:
        timestamps.append(frame["timestamp"])
        poses = frame["poses"]
        if not poses:
            joints_3d_list.append(None)
            visibility_list.append(None)
            continue

        # Find the requested person_id; fall back to first available.
        pose = None
        for p in poses:
            if p["id"] == person_id:
                pose = p
                break
        if pose is None and poses:
            pose = poses[0]

        if pose is None:
            joints_3d_list.append(None)
            visibility_list.append(None)
            continue

        joints = np.array(pose["points_3d"], dtype=np.float64)
        scores = np.array(pose["scores"], dtype=np.float64)
        joints_3d_list.append(joints)
        visibility_list.append(scores)

    return timestamps, joints_3d_list, visibility_list


def build_shelf_dataset(
    data_root: Path,
    person_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Camera]]:
    """Build (points_2d, confidences, joints_3d, cameras) from Shelf/Campus GT.

    Args:
        data_root: path to Shelf_Seq1 or Campus_Seq1 folder.
        person_id: which person to extract.

    Returns:
        points_2d: (T, V, J, 2)
        confidences: (T, V, J)
        joints_3d: (T, J, 3)
        cameras: list of V Camera objects
    """
    calibration_path = data_root / "calibration.json"
    annotation_path = data_root / "annotation_3d.json"

    cameras, _ = load_cameras(calibration_path)
    timestamps, joints_3d_list, visibility_list = load_3d_annotations(annotation_path, person_id)

    all_points_2d = []
    all_confidences = []
    all_joints_3d = []

    for joints_3d, scores in zip(joints_3d_list, visibility_list):
        if joints_3d is None:
            continue
        points_2d = np.stack([_project_points(joints_3d, cam) for cam in cameras], axis=0)
        # Use visibility scores as confidences; if all zero, set to ones.
        confidences = scores[None, :].repeat(len(cameras), axis=0)
        if confidences.sum() == 0:
            confidences = np.ones_like(confidences)
        all_points_2d.append(points_2d)
        all_confidences.append(confidences)
        all_joints_3d.append(joints_3d)

    return (
        np.array(all_points_2d),
        np.array(all_confidences),
        np.array(all_joints_3d),
        cameras,
    )
