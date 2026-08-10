"""Standard 3D evaluation for Shelf/Campus using true 3D GT.

Loads ``detection.json`` as the 2D input, ``annotation_3d.json`` as the true 3D
ground truth, triangulates the per-view 2D detections, and reports MPJPE and
PA-MPJPE against the true 3D.

This script is intentionally a single-person proof-of-concept: for each frame
and camera we select the detected person that best matches the projected true
3D of the requested ``person_id``.  This avoids building a full multi-person
association layer while still giving a clean 3D-error signal.

Usage:
    /d/anaconda3/python experiments/eval_shelf_campus_standard.py \
        --data_root data/shelf_campus/Shelf_Seq1 --person_id 0

The 17-joint COCO-style detections are mapped to the 14-joint Shelf/Campus
annotation skeleton defined in ``docs/swarm_iter2/shelf_campus_source.md``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.data.shelf_loader import load_cameras
from motionflow_mv.eval.metrics import mpjpe, pa_mpjpe
from motionflow_mv.fusion.triangulation import triangulate_dlt_torch

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
    16,  # 0  r-ankle  <- right_ankle
    14,  # 1  r-knee   <- right_knee
    12,  # 2  r-hip    <- right_hip
    11,  # 3  l-hip    <- left_hip
    13,  # 4  l-knee   <- left_knee
    15,  # 5  l-ankle  <- left_ankle
    10,  # 6  r-wrist  <- right_wrist
    8,   # 7  r-elbow  <- right_elbow
    6,   # 8  r-shoulder<- right_shoulder
    5,   # 9  l-shoulder<- left_shoulder
    7,   # 10 l-elbow  <- left_elbow
    9,   # 11 l-wrist  <- left_wrist
    0,   # 12 bottom-head <- nose
    0,   # 13 top-head   <- nose (no direct COCO keypoint, reuse nose)
], dtype=np.int64)
# fmt: on


def _project_points(points_3d: np.ndarray, camera: Camera) -> np.ndarray:
    """Project (J, 3) points to 2D using the camera projection matrix."""
    P = camera.projection_matrix
    X_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return x


def load_detection_lookup(detection_path: Path) -> Dict[float, Dict[str, dict]]:
    """Return ``timestamp -> camera_name -> frame`` lookup."""
    with open(detection_path) as f:
        det = json.load(f)

    lookup: Dict[float, Dict[str, dict]] = {}
    for key, frame in det["frames"].items():
        cam_name = key.split("/")[0]
        timestamp = float(frame["timestamp"])
        lookup.setdefault(timestamp, {})[cam_name] = frame
    return lookup


def build_camera_objects(calibration_path: Path, image_wh: List[int]) -> Tuple[List[Camera], List[str]]:
    """Load cameras and convert normalized intrinsics to pixel units.

    ``calibration.json`` stores the world-to-camera transform ``Tw`` as well as
    a normalized intrinsic matrix ``K``.  The correct projection matrix is
    ``P = K_pixel * inv(Tw)``, which is equivalent to using the camera-to-world
    rotation/translation in the pinhole model.  This conversion mirrors the
    reference snippet in ``docs/swarm_iter2/shelf_campus_source.md``.
    """
    cameras, camera_names = load_cameras(calibration_path)
    W, H = image_wh
    for cam in cameras:
        # Convert normalized intrinsics to pixel units (do not zero out the
        # principal point).  fx,cx are scaled by W; fy,cy are scaled by H.
        K = cam.K.copy()
        K[0, 0] *= W
        K[0, 2] *= W
        K[1, 1] *= H
        K[1, 2] *= H
        cam.K = K

        # ``load_cameras`` interprets Tw as world-to-camera.  The dataset's
        # projection convention is P = K * inv(Tw).
        R_wc = cam.R.copy()
        t_wc = cam.t.copy()
        cam.R = R_wc.T
        cam.t = -R_wc.T @ t_wc

    return cameras, camera_names


def select_detection_for_person(
    camera: Camera,
    det_frame: dict,
    gt_2d: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Select the detection closest to the projected true 3D.

    Args:
        camera: camera with pixel-unit intrinsics.
        det_frame: dict with ``poses`` (list of detection candidates).
        gt_2d: (14, 2) projected true 3D for this camera/person.

    Returns:
        ``(points_2d, scores)`` of shape (17, 2) and (17,) for the best candidate,
        or ``None`` if no candidate exists.
    """
    poses = det_frame.get("poses", [])
    if not poses:
        return None

    best_error = float("inf")
    best = None
    for pose in poses:
        p2d = np.array(pose["points_2d"], dtype=np.float64)  # (17, 2)
        if p2d.shape[0] != 17:
            continue
        p2d_14 = p2d[COCO17_TO_SHELF14]
        error = np.linalg.norm(p2d_14 - gt_2d, axis=-1).mean()
        if error < best_error:
            best_error = error
            best = (p2d, np.array(pose["scores"], dtype=np.float64))
    return best


def triangulate_frame(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    proj_matrices: torch.Tensor,
) -> np.ndarray:
    """Triangulate a single frame of (V, J, 2) points.

    Args:
        points_2d: (V, J, 2) pixel keypoints.
        confidences: (V, J) confidence scores.
        proj_matrices: (1, V, 3, 4) projection matrices.

    Returns:
        (J, 3) triangulated 3D points.
    """
    V, J, _ = points_2d.shape
    p2d_t = torch.from_numpy(points_2d).float()  # (V, J, 2)
    conf_t = torch.from_numpy(confidences).float()  # (V, J)
    pred_3d = np.zeros((J, 3), dtype=np.float64)

    for j in range(J):
        w = conf_t[:, j]
        if w.sum() == 0:
            w = torch.ones_like(w)
        pred_3d[j] = triangulate_dlt_torch(
            p2d_t[:, j].unsqueeze(0), proj_matrices, weights=w.unsqueeze(0)
        ).numpy()
    return pred_3d


def evaluate_shelf_campus(
    data_root: Path,
    person_id: int = 0,
    max_frames: Optional[int] = None,
) -> Dict[str, float]:
    """Run standard Shelf/Campus D evaluation for a single person.

    Returns:
        dict with keys ``mpjpe`` and ``pa_mpjpe``.
    """
    detection_path = data_root / "detection.json"
    annotation_path = data_root / "annotation_3d.json"
    calibration_path = data_root / "calibration.json"

    with open(detection_path) as f:
        det_meta = json.load(f)
    image_wh = det_meta["image_wh"]

    cameras, camera_names = build_camera_objects(calibration_path, image_wh)
    det_lookup = load_detection_lookup(detection_path)

    with open(annotation_path) as f:
        annotations = json.load(f)

    # Pre-compute per-camera projection matrices (batch dim for triangulation).
    K = np.stack([cam.K for cam in cameras])
    R = np.stack([cam.R for cam in cameras])
    t = np.stack([cam.t for cam in cameras])
    proj_matrices = (
        torch.from_numpy(K).float().unsqueeze(0)
        @ torch.cat(
            [
                torch.from_numpy(R).float().unsqueeze(0),
                torch.from_numpy(t).float().unsqueeze(0).unsqueeze(-1),
            ],
            dim=-1,
        )
    )  # (1, V, 3, 4)

    pred_3d_list: List[np.ndarray] = []
    gt_3d_list: List[np.ndarray] = []

    for frame in annotations:
        ts = frame["timestamp"]
        det_at_ts = det_lookup.get(ts)
        if det_at_ts is None:
            continue
        if not all(cam in det_at_ts for cam in camera_names):
            continue

        # Find the requested person in the true 3D annotations.
        gt_pose = None
        for pose in frame["poses"]:
            if pose["id"] == person_id:
                gt_pose = pose
                break
        if gt_pose is None:
            continue

        gt_3d = np.array(gt_pose["points_3d"], dtype=np.float64)  # (14, 3)
        if gt_3d.shape[0] != 14:
            continue

        # Project true 3D to each view so we can pick the matching detection.
        per_cam_p2d: List[np.ndarray] = []
        per_cam_conf: List[np.ndarray] = []
        all_views_present = True
        for cam_name, camera in zip(camera_names, cameras):
            gt_2d_proj = _project_points(gt_3d, camera)  # (14, 2)
            selected = select_detection_for_person(
                camera, det_at_ts[cam_name], gt_2d_proj
            )
            if selected is None:
                all_views_present = False
                break
            per_cam_p2d.append(selected[0])
            per_cam_conf.append(selected[1])

        if not all_views_present:
            continue

        points_2d = np.stack(per_cam_p2d, axis=0)[:, COCO17_TO_SHELF14, :]  # (V, 14, 2)
        confidences = np.stack(per_cam_conf, axis=0)[:, COCO17_TO_SHELF14]  # (V, 14)

        pred_3d = triangulate_frame(points_2d, confidences, proj_matrices)

        pred_3d_list.append(pred_3d)
        gt_3d_list.append(gt_3d)

        if max_frames is not None and len(pred_3d_list) >= max_frames:
            break

    if not pred_3d_list:
        raise RuntimeError(
            f"No frames evaluated for person_id={person_id} in {data_root}"
        )

    pred_3d_arr = np.stack(pred_3d_list, axis=0)  # (T, 14, 3)
    gt_3d_arr = np.stack(gt_3d_list, axis=0)  # (T, 14, 3)

    metrics = {
        "mpjpe": mpjpe(pred_3d_arr, gt_3d_arr),
        "pa_mpjpe": pa_mpjpe(pred_3d_arr, gt_3d_arr),
        "frames_evaluated": len(pred_3d_list),
    }
    return metrics, pred_3d_arr, gt_3d_arr


def main():
    parser = argparse.ArgumentParser(
        description="Standard 3D evaluation for Shelf/Campus using true 3D GT."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/shelf_campus/Shelf_Seq1",
        help="Path to Shelf_Seq1 or Campus_Seq1 folder.",
    )
    parser.add_argument(
        "--person_id",
        type=int,
        default=0,
        help="Person ID in annotation_3d.json to evaluate.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="If given, stop after this many successfully evaluated frames.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save a .npz with pred_3d and gt_3d arrays.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    metrics, pred_3d, gt_3d = evaluate_shelf_campus(
        data_root, person_id=args.person_id, max_frames=args.max_frames
    )

    print(f"Evaluated {metrics['frames_evaluated']} frames from {data_root}")
    print(f"MPJPE:     {metrics['mpjpe']:.4f}")
    print(f"PA-MPJPE:  {metrics['pa_mpjpe']:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, pred_3d=pred_3d, gt_3d=gt_3d)
        print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
