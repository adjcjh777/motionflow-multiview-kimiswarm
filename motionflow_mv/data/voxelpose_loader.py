"""Loader for VoxelPose-provided Shelf/Campus calibration and 2D predictions.

The VoxelPose repo ships:
    - calibration_shelf.json / calibration_campus.json
    - pred_shelf_maskrcnn_hrnet_coco.pkl / pred_campus_maskrcnn_hrnet_coco.pkl

This loader converts them into our Camera / 2D prediction formats without
requiring the raw images or 3D GT.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List
import numpy as np

from motionflow_mv.calibration.camera import Camera


class VoxelPoseLoader:
    """Generic loader for a VoxelPose dataset (Shelf, Campus, etc.)."""

    def __init__(self, data_root: str, calibration_file: str, predictions_file: str):
        self.data_root = Path(data_root)
        self.cameras = self._load_cameras(self.data_root / calibration_file)
        self.predictions = self._load_predictions(self.data_root / predictions_file)

    def _load_cameras(self, calibration_path: Path) -> Dict[str, Camera]:
        with open(calibration_path) as f:
            raw = json.load(f)
        cameras = {}
        for cam_id, cam_dict in raw.items():
            fx = cam_dict["fx"]
            fy = cam_dict["fy"]
            cx = cam_dict["cx"]
            cy = cam_dict["cy"]
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            R = np.array(cam_dict["R"])  # 3x3
            T = np.array(cam_dict["T"]).reshape(3)  # camera center in world coords
            # VoxelPose convention: x_cam = R (X - T)
            # Our Camera projects: x = K (R_cam X + t)
            # => R_cam = R, t = -R @ T
            t = -R @ T
            cameras[str(cam_id)] = Camera(K=K, R=R, t=t)
        return cameras

    def _load_predictions(self, pred_path: Path) -> Dict[str, List[np.ndarray]]:
        """Return dict mapping 'cam_id_frame_idx' -> list of per-person (J,2/3) arrays."""
        with open(pred_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        predictions = {}
        for key, val in data.items():
            # val is a list of dicts, each dict has key "pred"
            preds = [np.array(p["pred"]) for p in val]
            predictions[key] = preds
        return predictions

    def get_frame_predictions(self, camera_id: str, frame_idx: int) -> List[np.ndarray]:
        key = f"{camera_id}_{frame_idx}"
        return self.predictions.get(key, [])

    def get_camera(self, camera_id: str) -> Camera:
        return self.cameras[str(camera_id)]


class VoxelPoseShelfLoader(VoxelPoseLoader):
    """Load calibration and 2D predictions for the Shelf dataset."""

    def __init__(self, data_root: str):
        super().__init__(
            data_root,
            calibration_file="calibration_shelf.json",
            predictions_file="pred_shelf_maskrcnn_hrnet_coco.pkl",
        )


class VoxelPoseCampusLoader(VoxelPoseLoader):
    """Load calibration and 2D predictions for the Campus dataset."""

    def __init__(self, data_root: str):
        super().__init__(
            data_root,
            calibration_file="calibration_campus.json",
            predictions_file="pred_campus_maskrcnn_hrnet_coco.pkl",
        )
