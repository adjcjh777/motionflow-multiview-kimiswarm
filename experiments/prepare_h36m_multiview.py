"""Build a small multi-view H36M dataset from the preprocessed keypoint subset.

Uses the camera parameters from:
    https://github.com/karfly/human36m-camera-parameters
and triangulates per-frame 3D joints via DLT to obtain world-coordinate targets.

Output example:
    data/h36m_hf/s_01_act_02_multiview.npz
"""

import argparse
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.triangulation import triangulate_dlt
from motionflow_mv.calibration.camera import Camera


def build_projection_matrix(K, R, t):
    Rt = np.concatenate([R, t.reshape(3, 1)], axis=1)
    return K @ Rt


def triangulate_joints(points_2d, cameras):
    """points_2d: (V, J, 2), cameras: list of Camera. Returns (J, 3)."""
    V, J, _ = points_2d.shape
    P = np.stack([cam.projection_matrix for cam in cameras], axis=0)
    X = np.zeros((J, 3), dtype=np.float64)
    for j in range(J):
        X[j] = triangulate_dlt(points_2d[:, j, :], P)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--actions", type=int, nargs="+", default=[2])
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--out_dir", type=str, default="data/h36m_hf")
    args = parser.parse_args()

    with open("data/h36m_hf/camera_params.json") as f:
        cam_params = json.load(f)

    with zipfile.ZipFile("data/h36m_hf/h36m_sh_conf_cam_source_final.pkl.zip") as z:
        with z.open("h36m_sh_conf_cam_source_final.pkl") as f:
            data = pickle.load(f)

    split_data = data[args.split]

    # Group indices by base source (handles both train ``..._cam_XX`` and test
    # ``..._ca_XX`` source formats).
    import re
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    for i, src in enumerate(split_data["source"]):
        # Camera suffix is the last token; the keyword before it is cam/ca.
        # Train: s_01_act_02_cam_01 -> base s_01_act_02, cam cam_01
        # Test:  s_09_act_02_subact_01_ca_01 -> base s_09_act_02_subact_01, cam ca_01
        m = re.match(r"(.+)_(cam|ca)_(\d+)$", src)
        if m is None:
            raise ValueError(f"Unsupported source format: {src}")
        base = m.group(1)
        cam = f"{m.group(2)}_{m.group(3)}"
        groups[base][cam].append(i)

    # Find all base groups for this subject/action.
    prefix = f"s_{args.subject:02d}_act_"
    candidate_bases = sorted([b for b in groups.keys() if b.startswith(prefix)])
    target_bases = []
    for a in args.actions:
        a_prefix = f"s_{args.subject:02d}_act_{a:02d}"
        matches = [b for b in candidate_bases if b.startswith(a_prefix)]
        if not matches:
            raise ValueError(f"No source found for subject {args.subject} action {a} in {args.split}")
        target_bases.extend(matches)

    print(f"Found {len(target_bases)} source groups for subject {args.subject} actions {args.actions}")

    # Build camera objects from the first action (same subject => same extrinsics).
    first_cams_dict = groups[target_bases[0]]
    # Preserve the original camera order (first occurrence in the source list),
    # not a lexical sort, because ca_01/ca_02/etc. may not correspond to the
    # camera_name ordering.
    cam_names = sorted(first_cams_dict.keys(), key=lambda c: first_cams_dict[c][0])
    subject_key = f"S{args.subject}"
    first_idxs = [first_cams_dict[cam][0] for cam in cam_names]
    cameras = []
    for i, cam_name in enumerate(cam_names):
        camera_name = split_data["camera_name"][first_idxs[i]]
        intr = cam_params["intrinsics"][camera_name]["calibration_matrix"]
        K = np.array(intr, dtype=np.float64)
        ext = cam_params["extrinsics"][subject_key][camera_name]
        R = np.array(ext["R"], dtype=np.float64)
        t = np.array(ext["t"], dtype=np.float64).reshape(3)
        cameras.append(Camera(K=K, R=R, t=t))

    all_p2d = []
    all_conf = []
    all_j3d = []

    for target_base in target_bases:
        cams_dict = groups[target_base]
        n_frames = len(cams_dict[cam_names[0]])
        print(f"Processing {target_base}: {n_frames} frames")
        for frame in range(n_frames):
            p2d = np.stack([split_data["joint_2d"][cams_dict[cam][frame]] for cam in cam_names], axis=0)
            conf = np.stack([split_data["confidence"][cams_dict[cam][frame]].squeeze(-1) for cam in cam_names], axis=0)
            j3d = triangulate_joints(p2d, cameras)
            all_p2d.append(p2d)
            all_conf.append(conf)
            all_j3d.append(j3d)

    points_2d = np.stack(all_p2d, axis=0)
    confidences = np.stack(all_conf, axis=0)
    joints_3d = np.stack(all_j3d, axis=0)

    print(f"points_2d shape: {points_2d.shape}")
    print(f"joints_3d range: {joints_3d.min():.2f} {joints_3d.max():.2f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actions_str = "_".join(f"{a:02d}" for a in args.actions)
    out_path = out_dir / f"s_{args.subject:02d}_acts_{actions_str}_multiview.npz"
    np.savez(
        out_path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=np.stack([cam.K for cam in cameras]),
        camera_R=np.stack([cam.R for cam in cameras]),
        camera_t=np.stack([cam.t for cam in cameras]),
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
