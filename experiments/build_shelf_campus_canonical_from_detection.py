"""Build non-circular canonical .npz files for Shelf/Campus from detections + true 3D.

The existing ``pseudogt_m.npz`` files under ``data/shelf_campus/*/`` mix GT-projection
and true-3D labels inconsistently.  This script rebuilds the WebBridge canonical
format from the raw JSON files so that:

* ``points_2d`` comes from COCO-17 detections selected by projecting true 3D.
* ``joints_3d`` is the true 3D annotation from ``annotation_3d.json``.
* All 3D coordinates and camera translations are in meters.

The resulting files are written to ``data/webbridge/shelf_campus_detected/``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.triangulation import triangulate_dlt


# fmt: off
# Shelf/Campus 14-joint annotation order (from docs/swarm_iter2/shelf_campus_source.md):
#   0: r-ankle, 1: r-knee, 2: r-hip, 3: l-hip, 4: l-knee, 5: l-ankle,
#   6: r-wrist, 7: r-elbow, 8: r-shoulder, 9: l-shoulder, 10: l-elbow,
#   11: l-wrist, 12: bottom-head, 13: top-head
#
# COCO 17-joint detection order assumed in detection.json:
#   0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
#   5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
#   9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
#   13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
COCO17_TO_SHELF14 = np.array([
    16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9, 0, 0,
], dtype=np.int64)
# Inverse: for each COCO-17 joint, which Shelf/Campus 14 joint to use.
# Head-related COCO joints reuse the bottom-head (12) annotation.
SHELF14_TO_COCO17 = np.array([
    12, 12, 12, 12, 12,   # nose, left/right eye, left/right ear
    9, 8, 10, 7, 11, 6,   # shoulders, elbows, wrist
    3, 2, 4, 1, 5, 0,     # hips, knees, ankles
], dtype=np.int64)
# fmt: on


def _load_detection_lookup(detection_path: Path) -> Dict[float, Dict[str, dict]]:
    """Return ``timestamp -> camera_name -> frame`` lookup."""
    with open(detection_path) as f:
        det = json.load(f)

    lookup: Dict[float, Dict[str, dict]] = {}
    for key, frame in det["frames"].items():
        cam_name = key.split("/")[0]
        timestamp = float(frame["timestamp"])
        lookup.setdefault(timestamp, {})[cam_name] = frame
    return lookup


def _build_cameras(calibration_path: Path, image_wh: List[int]) -> Tuple[List[Camera], List[str]]:
    """Load cameras from calibration.json and scale intrinsics to pixel units.

    ``calibration.json`` stores the world-to-camera transform ``Tw`` and a *normalized*
    intrinsic matrix ``K``.  We convert to pixel units while preserving the principal
    point for projection/detection matching.  The returned cameras use the standard
    ``Camera`` convention: ``P = K [R | t]`` where ``R,t`` are world-to-camera.
    """
    with open(calibration_path) as f:
        calib = json.load(f)

    W, H = image_wh
    cameras = []
    camera_names = sorted(calib["cameras"].keys())
    for name in camera_names:
        cam = calib["cameras"][name]
        K = np.array(cam["K"], dtype=np.float64)
        Tw = np.array(cam["Tw"], dtype=np.float64)

        K = K.copy()
        K[0, 0] *= W
        K[0, 2] *= W
        K[1, 1] *= H
        K[1, 2] *= H

        # Use the same convention as ``eval_shelf_campus_standard.py``: the stored
        # Tw is treated as world-to-camera, so the correct projection matrix is
        # P = K [R_wc^T | -R_wc^T t_wc] (camera centre in world coordinates).
        R_wc = Tw[:3, :3]
        t_wc = Tw[:3, 3]
        R = R_wc.T
        t = -R_wc.T @ t_wc
        cameras.append(Camera(K=K, R=R, t=t))

    return cameras, camera_names


def _project_points(points_3d: np.ndarray, camera: Camera) -> np.ndarray:
    """Project (J, 3) points to 2D using the camera projection matrix."""
    P = camera.projection_matrix
    X_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return x


def _select_detection(
    det_frame: dict,
    gt_2d_14: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Select the detection closest to the projected true 3D.

    Args:
        det_frame: dict with ``poses`` list of detection candidates.
        gt_2d_14: (14, 2) projected true 3D in Shelf/Campus skeleton order.

    Returns:
        ``(points_2d, scores)`` of shape (17, 2) and (17,) for the best candidate,
        or ``None`` if no candidate exists.
    """
    poses = det_frame.get("poses", [])
    if not poses:
        return None

    best_error = float("inf")
    best: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for pose in poses:
        p2d = np.array(pose["points_2d"], dtype=np.float64)
        if p2d.shape[0] != 17:
            continue
        scores = np.array(pose["scores"], dtype=np.float64)
        p2d_14 = p2d[COCO17_TO_SHELF14]
        error = np.linalg.norm(p2d_14 - gt_2d_14, axis=-1).mean()
        if error < best_error:
            best_error = error
            best = (p2d, scores)
    return best


def _skeleton14_to_17(points_14: np.ndarray) -> np.ndarray:
    """Map a (..., 14, 3) or (..., 14) array to 17 COCO joints."""
    return points_14[..., SHELF14_TO_COCO17, :]


def _scale_K(K: np.ndarray, image_wh: List[int]) -> np.ndarray:
    """Scale the normalized intrinsic matrix to pixel units, preserving cx/cy."""
    W, H = image_wh
    Ks = K.copy()
    Ks[:, 0, 0] *= W
    Ks[:, 0, 2] *= W
    Ks[:, 1, 1] *= H
    Ks[:, 1, 2] *= H
    return Ks


def _convert_sequence(
    data_root: Path,
    output_root: Path,
    person_id: int = 0,
    train_ratio: float = 0.8,
) -> Optional[Tuple[Path, Path]]:
    """Convert one Shelf/Campus sequence to train/val canonical .npz files.

    Args:
        data_root: path to ``Shelf_Seq1`` or ``Campus_Seq1`` folder.
        output_root: directory where ``<seq>_{train,val}_detected_m.npz`` are written.
        person_id: which person to extract from ``annotation_3d.json``.
        train_ratio: fraction of frames assigned to the train split.

    Returns:
        Tuple of (train_path, val_path) or ``None`` if no frames were extracted.
    """
    data_root = Path(data_root)
    detection_path = data_root / "detection.json"
    annotation_path = data_root / "annotation_3d.json"
    calibration_path = data_root / "calibration.json"

    if not detection_path.exists():
        raise FileNotFoundError(f"detection.json missing: {detection_path}")
    if not annotation_path.exists():
        print(f"  {data_root.name}: annotation_3d.json missing, skipping")
        return None
    if not calibration_path.exists():
        raise FileNotFoundError(f"calibration.json missing: {calibration_path}")

    with open(detection_path) as f:
        det_meta = json.load(f)
    image_wh = det_meta["image_wh"]

    cameras_full, camera_names = _build_cameras(calibration_path, image_wh)
    det_lookup = _load_detection_lookup(detection_path)

    with open(annotation_path) as f:
        annotations = json.load(f)

    points_2d_list: List[np.ndarray] = []
    confidences_list: List[np.ndarray] = []
    joints_3d_list: List[np.ndarray] = []

    skipped_no_annotation = 0
    skipped_no_detection = 0

    for frame in annotations:
        ts = frame["timestamp"]
        det_at_ts = det_lookup.get(ts)
        if det_at_ts is None:
            skipped_no_detection += 1
            continue
        if not all(cam in det_at_ts for cam in camera_names):
            skipped_no_detection += 1
            continue

        # Find requested person in true 3D annotations.
        gt_pose = None
        for pose in frame["poses"]:
            if pose["id"] == person_id:
                gt_pose = pose
                break
        if gt_pose is None:
            skipped_no_annotation += 1
            continue

        gt_3d_14 = np.array(gt_pose["points_3d"], dtype=np.float64)
        if gt_3d_14.shape[0] != 14:
            skipped_no_annotation += 1
            continue

        per_cam_p2d: List[np.ndarray] = []
        per_cam_conf: List[np.ndarray] = []
        all_views_present = True
        for cam_name, camera in zip(camera_names, cameras_full):
            gt_2d_proj = _project_points(gt_3d_14, camera)
            selected = _select_detection(det_at_ts[cam_name], gt_2d_proj)
            if selected is None:
                all_views_present = False
                break
            per_cam_p2d.append(selected[0])
            per_cam_conf.append(selected[1])

        if not all_views_present:
            skipped_no_detection += 1
            continue

        points_2d_list.append(np.stack(per_cam_p2d, axis=0))
        confidences_list.append(np.stack(per_cam_conf, axis=0))
        # Map the 14-joint true 3D to the 17-joint COCO layout used by the detections.
        joints_3d_list.append(_skeleton14_to_17(gt_3d_14) / 100.0)

    if not points_2d_list:
        print(f"  {data_root.name}: no frames extracted for person_id={person_id}")
        return None

    points_2d = np.stack(points_2d_list, axis=0)
    confidences = np.stack(confidences_list, axis=0)
    joints_3d = np.stack(joints_3d_list, axis=0)

    # Temporal split: first train_ratio frames -> train, remainder -> val.
    n_frames = points_2d.shape[0]
    n_train = int(n_frames * train_ratio)

    # Camera arrays stored with zero principal point, in meters, using the same
    # convention as the selection cameras.
    with open(calibration_path) as f:
        calib = json.load(f)
    K_raw = np.stack([
        np.array(calib["cameras"][name]["K"], dtype=np.float64)
        for name in camera_names
    ], axis=0)
    K_out = _scale_K(K_raw, image_wh)
    R_out = np.stack([
        np.array(calib["cameras"][name]["Tw"], dtype=np.float64)[:3, :3].T
        for name in camera_names
    ], axis=0)
    t_out = np.stack([
        -np.array(calib["cameras"][name]["Tw"], dtype=np.float64)[:3, :3].T
        @ np.array(calib["cameras"][name]["Tw"], dtype=np.float64)[:3, 3]
        for name in camera_names
    ], axis=0) / 100.0

    train_dict = {
        "points_2d": points_2d[:n_train],
        "confidences": confidences[:n_train],
        "joints_3d": joints_3d[:n_train],
        "camera_K": K_out,
        "camera_R": R_out,
        "camera_t": t_out,
    }
    val_dict = {
        "points_2d": points_2d[n_train:],
        "confidences": confidences[n_train:],
        "joints_3d": joints_3d[n_train:],
        "camera_K": K_out,
        "camera_R": R_out,
        "camera_t": t_out,
    }

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seq_name = data_root.name.lower().replace("_seq1", "")
    train_path = output_root / f"{seq_name}_seq1_train_detected_m.npz"
    val_path = output_root / f"{seq_name}_seq1_val_detected_m.npz"

    np.savez_compressed(train_path, **train_dict)
    np.savez_compressed(val_path, **val_dict)

    print(f"  {data_root.name}: {n_frames} frames -> {train_path}, {val_path}")
    print(f"    skipped: {skipped_no_annotation} no annotation, {skipped_no_detection} missing detections")

    return train_path, val_path


def _diagnose_val(val_path: Path) -> None:
    """Run a DLT re-triangulation diagnosis on the validation split."""
    data = np.load(val_path, allow_pickle=True)
    p2d = data["points_2d"]
    j3d = data["joints_3d"]
    K = data["camera_K"]
    R = data["camera_R"]
    t = data["camera_t"]

    P = np.zeros((K.shape[0], 3, 4), dtype=np.float64)
    for v in range(K.shape[0]):
        Rt = np.concatenate([R[v], t[v][:, None]], axis=1)
        P[v] = K[v] @ Rt

    n_frames, n_views, n_joints, _ = p2d.shape
    re_tri = np.zeros_like(j3d)
    for f in range(n_frames):
        for j in range(n_joints):
            re_tri[f, j] = triangulate_dlt(p2d[f, :, j], P)

    direct_mje = np.linalg.norm(re_tri - j3d, axis=-1).mean() * 1000.0
    root_mje = np.linalg.norm(
        (re_tri - re_tri.mean(axis=-2, keepdims=True)) - (j3d - j3d.mean(axis=-2, keepdims=True)),
        axis=-1,
    ).mean() * 1000.0
    per_joint = np.linalg.norm(re_tri - j3d, axis=-1).mean(axis=0) * 1000.0

    print(f"  DLT re-triangulation diagnostics for {val_path}:")
    print(f"    frames={n_frames}, views={n_views}, joints={n_joints}")
    print(f"    direct MJE (no root align): {direct_mje:.4f} mm")
    print(f"    root-aligned MPJPE:       {root_mje:.4f} mm")
    print(f"    max per-joint error:      {per_joint.max():.4f} mm")
    print(f"    median per-joint error:   {np.median(per_joint):.4f} mm")
    print(f"    mean per-joint error:     {per_joint.mean():.4f} mm")


def main():
    parser = argparse.ArgumentParser(
        description="Build non-circular canonical .npz for Shelf/Campus from detections + true 3D."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/shelf_campus/Shelf_Seq1",
        help="Path to a Shelf/Campus sequence folder (e.g. data/shelf_campus/Shelf_Seq1).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/webbridge/shelf_campus_detected",
        help="Directory to write the output .npz files.",
    )
    parser.add_argument(
        "--person_id",
        type=int,
        default=0,
        help="Person ID in annotation_3d.json to extract.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of frames to use for the train split.",
    )
    parser.add_argument(
        "--campus_sibling",
        action="store_true",
        default=True,
        help="Also process the sibling Campus_Seq1 / Shelf_Seq1 if it exists.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    roots = [data_root]
    # If the requested sequence is one of the Shelf/Campus pair, also process the other.
    if args.campus_sibling and data_root.parent.name == "shelf_campus":
        if "Shelf" in data_root.name:
            sibling = data_root.parent / "Campus_Seq1"
        elif "Campus" in data_root.name:
            sibling = data_root.parent / "Shelf_Seq1"
        else:
            sibling = None
        if sibling is not None and sibling.exists() and sibling not in roots:
            roots.append(sibling)

    all_paths = []
    for root in roots:
        paths = _convert_sequence(root, output_root, args.person_id, args.train_ratio)
        if paths is not None:
            all_paths.append(paths)
            train_path, val_path = paths
            _diagnose_val(val_path)

    if not all_paths:
        print("No sequences were converted.")
        return

    print("\nSaved files:")
    for train_path, val_path in all_paths:
        print(f"  {train_path}")
        print(f"  {val_path}")


if __name__ == "__main__":
    main()
