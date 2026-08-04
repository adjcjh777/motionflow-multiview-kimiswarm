"""Helper utilities for the multi-view pose fusion pipeline."""

from itertools import product
from typing import Dict, List, Tuple
import numpy as np

from .calibration.camera import Camera
from .fusion.triangulation import triangulate_confidence_weighted


def _reprojection_error_for_group(
    points_2d: np.ndarray,
    cameras: List[Camera],
    triangulated_3d: np.ndarray,
) -> float:
    """Compute sum of per-joint reprojection errors for a matched group.

    points_2d: (V, J, 2)
    cameras: list of V Camera
    triangulated_3d: (J, 3)
    """
    total = 0.0
    X_h = np.hstack([triangulated_3d, np.ones((triangulated_3d.shape[0], 1))])
    for view_points_2d, cam in zip(points_2d, cameras):
        P = cam.projection_matrix
        x_h = (P @ X_h.T).T
        x = x_h[:, :2] / x_h[:, 2:3]
        total += np.linalg.norm(x - view_points_2d, axis=1).sum()
    return total


def select_best_person_group(
    frame_predictions: Dict[str, List[np.ndarray]],
    cameras: Dict[str, Camera],
    camera_ids: List[str],
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Select one person per camera view by minimizing multi-view reprojection error.

    Args:
        frame_predictions: maps camera_id -> list of person arrays (J, 3) or (J, 2).
        cameras: maps camera_id -> Camera.
        camera_ids: ordered list of camera ids to consider.

    Returns:
        selected_indices: list of person indices, one per view.
        points_2d: (V, J, 2)
        confidences: (V, J)
    """
    valid_cameras = []
    person_lists = []
    for cid in camera_ids:
        preds = frame_predictions.get(cid, [])
        if len(preds) == 0:
            continue
        valid_cameras.append(cid)
        person_lists.append(list(range(len(preds))))

    if not valid_cameras:
        raise ValueError("No predictions for any camera.")

    cam_objects = [cameras[cid] for cid in valid_cameras]
    best_error = float("inf")
    best_combo = None

    for combo in product(*person_lists):
        points_2d = []
        confidences = []
        for cid, pid in zip(valid_cameras, combo):
            p = frame_predictions[cid][pid]
            if p.shape[-1] == 3:
                points_2d.append(p[:, :2])
                confidences.append(p[:, 2])
            else:
                points_2d.append(p[:, :2])
                confidences.append(np.ones(p.shape[0]))
        points_2d = np.stack(points_2d, axis=0)
        confidences = np.stack(confidences, axis=0)

        # Triangulate each joint
        proj_matrices = np.stack([cam.projection_matrix for cam in cam_objects], axis=0)
        try:
            triangulated = np.array([
                triangulate_confidence_weighted(
                    points_2d[:, j, :], proj_matrices, confidences[:, j]
                )
                for j in range(points_2d.shape[1])
            ])
        except Exception:
            continue

        err = _reprojection_error_for_group(points_2d, cam_objects, triangulated)
        if err < best_error:
            best_error = err
            best_combo = combo

    if best_combo is None:
        raise ValueError("Could not triangulate any person combination.")

    # Recompute points_2d/confidences for best combo
    points_2d = []
    confidences = []
    for cid, pid in zip(valid_cameras, best_combo):
        p = frame_predictions[cid][pid]
        if p.shape[-1] == 3:
            points_2d.append(p[:, :2])
            confidences.append(p[:, 2])
        else:
            points_2d.append(p[:, :2])
            confidences.append(np.ones(p.shape[0]))
    points_2d = np.stack(points_2d, axis=0)
    confidences = np.stack(confidences, axis=0)

    return best_combo, points_2d, confidences
