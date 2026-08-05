"""Build a pseudo-GT Campus dataset from detection.json + result_3d.json.

Output: data/shelf_campus/Campus_Seq1/pseudogt.npz
"""

import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.shelf_loader import load_cameras
from motionflow_mv.fusion.triangulation import triangulate_dlt_torch


def main():
    root = Path("data/shelf_campus/Campus_Seq1")
    det = json.loads((root / "detection.json").read_text())
    res3d = json.loads((root / "result_3d.json").read_text())
    cameras, camera_names = load_cameras(root / "calibration.json")

    image_wh = det["image_wh"]
    W, H = image_wh
    print(f"Image size: {image_wh}")

    # Convert normalized intrinsics to pixel units.
    for cam in cameras:
        cam.K = cam.K * np.array([[W, 0, 0], [0, H, 0], [0, 0, 1]])

    # Build a timestamp -> detection frame lookup per camera.
    det_lookup = {}
    for key, frame in det["frames"].items():
        parts = key.split("/")
        cam_name = parts[0]
        timestamp = float(parts[1].replace(".jpg", ""))
        det_lookup.setdefault(timestamp, {})[cam_name] = frame

    all_p2d = []
    all_conf = []
    all_j3d = []

    K_t = torch.from_numpy(np.stack([cam.K for cam in cameras])).float().unsqueeze(0)
    R_t = torch.from_numpy(np.stack([cam.R for cam in cameras])).float().unsqueeze(0)
    t_t = torch.from_numpy(np.stack([cam.t for cam in cameras])).float().unsqueeze(0)
    P = K_t @ torch.cat([R_t, t_t[..., None]], dim=-1)

    matched = 0
    for gt_frame in res3d:
        ts = gt_frame["timestamp"]
        if ts not in det_lookup:
            continue
        det_at_ts = det_lookup[ts]
        if not all(cam in det_at_ts for cam in camera_names):
            continue

        p2d_list = []
        conf_list = []
        for cam_name in camera_names:
            poses = det_at_ts[cam_name].get("poses", [])
            if not poses:
                break
            person = poses[0]
            p2d_list.append(person["points_2d"])
            conf_list.append(person["scores"])
        if len(p2d_list) != len(camera_names):
            continue

        p2d = np.array(p2d_list, dtype=np.float64)
        conf = np.array(conf_list, dtype=np.float64)

        # Build a DLT-consistent 3D target from the 2D detections.
        p2d_t = torch.from_numpy(p2d).float()
        conf_t = torch.from_numpy(conf).float()
        j3d = np.zeros((p2d.shape[1], 3), dtype=np.float64)
        for j in range(p2d.shape[1]):
            w = conf_t[:, j]
            if w.sum() == 0:
                w = torch.ones_like(w)
            j3d[j] = triangulate_dlt_torch(p2d_t[:, j], P, weights=w).numpy()

        all_p2d.append(p2d)
        all_conf.append(conf)
        all_j3d.append(j3d)
        matched += 1

    points_2d = np.stack(all_p2d, axis=0)
    confidences = np.stack(all_conf, axis=0)
    joints_3d = np.stack(all_j3d, axis=0)

    print(f"Matched frames: {matched}")
    print(f"points_2d shape: {points_2d.shape}")
    print(f"joints_3d range: {joints_3d.min():.2f} {joints_3d.max():.2f}")

    np.savez(
        root / "pseudogt.npz",
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=np.stack([cam.K for cam in cameras]),
        camera_R=np.stack([cam.R for cam in cameras]),
        camera_t=np.stack([cam.t for cam in cameras]),
    )
    print(f"Saved {root / 'pseudogt.npz'}")


if __name__ == "__main__":
    main()
