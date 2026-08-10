"""Build a small multi-view H36M dataset from the preprocessed keypoint subset.

DEPRECATION / REDIRECT NOTE:
    This script is largely redundant with the canonical converter in
    ``motionflow_mv.data.webbridge_loader:convert_human36m``. New code should
    use that converter (or ``experiments/batch_convert_h36m_webbridge.py``).

Uses the camera parameters from:
    https://github.com/karfly/human36m-camera-parameters

By default it triangulates per-frame 3D joints via DLT to obtain
world-coordinate targets. These labels are CIRCULAR (they are derived from the
input 2D keypoints). To emit non-circular labels, pass ``--true-gt-path``
pointing to an ``.npz`` file that already contains the true ``joints_3d`` array
of shape ``(T, J, 3)`` in the same frame order produced by this script.

Output example:
    data/h36m_hf/s_01_act_02_multiview.npz
"""

import argparse
import json
import pickle
import warnings
import zipfile
from pathlib import Path
from typing import Optional

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
    parser.add_argument(
        "--true-gt-path",
        type=Path,
        default=None,
        help=(
            "Optional path to an .npz file containing the true 3D ground-truth "
            "'joints_3d' array of shape (T, J, 3). When provided, it is used "
            "as the 3D label instead of triangulating the 2D input."
        ),
    )
    args = parser.parse_args()

    true_gt: Optional[np.ndarray] = None
    if args.true_gt_path is not None:
        if not args.true_gt_path.exists():
            raise FileNotFoundError(f"True 3D GT file not found: {args.true_gt_path}")
        true_gt = np.load(args.true_gt_path)["joints_3d"]
        print(f"Loaded true 3D GT from {args.true_gt_path}: shape {true_gt.shape}")
    else:
        warnings.warn(
            "No true 3D GT provided. Generated joints_3d labels are CIRCULAR "
            "(triangulated from the input 2D keypoints) and do not reflect true pose accuracy.",
            stacklevel=2,
        )

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

    # Determine the physical camera per slot.  In the train split the pkl's
    # per-row ``camera_name`` is consistent within each slot, but in the test
    # split it is shuffled across frames within a slot, so the first row cannot
    # be trusted.  If every slot carries a single camera_name across the first
    # base group we use it; otherwise we fall back to the fixed H36M studio
    # order (slot 01..04 -> 54138969, 55011271, 58860488, 60457274), which was
    # verified by reprojection (~3-8 px RMSE of the true mocap GT).
    H36M_FIXED_CAMERAS = ["54138969", "55011271", "58860488", "60457274"]
    per_slot_names = []
    consistent = True
    for cam in cam_names:
        names_in_slot = {
            str(split_data["camera_name"][i]) for i in first_cams_dict[cam]
        }
        if len(names_in_slot) != 1:
            consistent = False
            break
        per_slot_names.append(names_in_slot.pop())
    if consistent and len(per_slot_names) == len(cam_names):
        camera_names = per_slot_names
    else:
        if len(cam_names) != 4:
            raise ValueError(
                "Cannot infer camera names: inconsistent camera_name within "
                f"slots and {len(cam_names)} slots (expected 4)."
            )
        camera_names = H36M_FIXED_CAMERAS
        print(
            "NOTE: camera_name column is shuffled within camera slots "
            f"(test split); using fixed H36M camera order {camera_names}."
        )

    cameras = []
    for i, cam_name in enumerate(cam_names):
        camera_name = camera_names[i]
        intr = cam_params["intrinsics"][camera_name]["calibration_matrix"]
        K = np.array(intr, dtype=np.float64)
        ext = cam_params["extrinsics"][subject_key][camera_name]
        R = np.array(ext["R"], dtype=np.float64)
        t = np.array(ext["t"], dtype=np.float64).reshape(3)
        cameras.append(Camera(K=K, R=R, t=t))

    all_p2d = []
    all_conf = []
    all_j3d = []

    gt_frame_offset = 0
    for target_base in target_bases:
        cams_dict = groups[target_base]
        n_frames = len(cams_dict[cam_names[0]])
        print(f"Processing {target_base}: {n_frames} frames")
        for frame in range(n_frames):
            p2d = np.stack([split_data["joint_2d"][cams_dict[cam][frame]] for cam in cam_names], axis=0)
            conf = np.stack([split_data["confidence"][cams_dict[cam][frame]].squeeze(-1) for cam in cam_names], axis=0)
            if true_gt is not None:
                j3d = true_gt[gt_frame_offset + frame]
            else:
                j3d = triangulate_joints(p2d, cameras)
            all_p2d.append(p2d)
            all_conf.append(conf)
            all_j3d.append(j3d)
        gt_frame_offset += n_frames

    points_2d = np.stack(all_p2d, axis=0)
    confidences = np.stack(all_conf, axis=0)
    joints_3d = np.stack(all_j3d, axis=0)

    if true_gt is not None and true_gt.shape[0] != joints_3d.shape[0]:
        raise ValueError(
            f"True 3D GT frame count ({true_gt.shape[0]}) does not match "
            f"the number of frames produced ({joints_3d.shape[0]})."
        )

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
