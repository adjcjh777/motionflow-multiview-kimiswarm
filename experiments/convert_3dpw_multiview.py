"""Convert 3DPW sequence files to the MotionFlow-MultiView canonical ``.npz`` format.

3DPW is a single-camera + IMU in-the-wild dataset.  Because the project pipeline
expects calibrated multi-view input, this script can produce two kinds of output:

* ``pseudo`` (default): generate a static virtual multi-view rig around the actor
  and re-project the 3D SMPL joints into each virtual view.  The result is a
  canonical ``(T, V, J, 2)`` dataset that can be consumed directly by the existing
  ``ray_attention`` temporal model.

* ``actual``: keep the real moving camera as a single-view validation source.
  Because the camera is dynamic, the per-frame extrinsics are stored in extra
  arrays ``camera_K_frames``, ``camera_R_frames`` and ``camera_t_frames``.  The
  standard ``camera_K/R/t`` slots contain the first-frame intrinsics/extrinsics as
  a placeholder; consumers that need the true moving camera should read the
  per-frame arrays.

Usage
-----
    # pseudo multi-view (default)
    conda run -n mf python experiments/convert_3dpw_multiview.py \
        --input data/webbridge/3dpw/sequenceFiles/sequenceFiles/validation/courtyard_basketball_01.pkl \
        --output data/webbridge/3dpw/validation/courtyard_basketball_01_pseudo.npz \
        --mode pseudo --n_views 4

    # actual single-view
    conda run -n mf python experiments/convert_3dpw_multiview.py \
        --input data/webbridge/3dpw/sequenceFiles/sequenceFiles/validation/courtyard_basketball_01.pkl \
        --output data/webbridge/3dpw/validation/courtyard_basketball_01_actual.npz \
        --mode actual

Dependencies
------------
Only NumPy is required.  No SMPL runtime is needed because 3DPW already ships the
3D joint positions in ``jointPositions``.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera


def _look_at(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> tuple:
    """Build a world-to-camera (R, t) for a camera at *cam_pos* looking at *target*."""
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    # Right = up_world x forward, camera_up = forward x right => z-axis = forward.
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up_vec = np.cross(forward, right)
    up_vec /= np.linalg.norm(up_vec)
    R = np.stack([right, up_vec, forward], axis=0)
    t = -R @ cam_pos
    return R, t


def _project(points_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project (..., 3) world points through K, R, t -> (..., 2)."""
    original_shape = points_3d.shape
    X = points_3d.reshape(-1, 3)
    X_cam = (R @ X.T).T + t  # (N, 3)
    X_screen = (K @ X_cam.T).T
    depth = X_screen[:, 2:3]
    points_2d = X_screen[:, :2] / np.maximum(depth, 1e-6)
    return points_2d.reshape(original_shape[:-1] + (2,))


def _build_pseudo_rig(
    joints_3d: np.ndarray,
    actual_cam_poses: np.ndarray,
    actual_K: np.ndarray,
    n_views: int,
    noise_std: float = 0.0,
) -> tuple:
    """Build a static virtual multi-view rig around the actor.

    Returns
    -------
    cameras: list of Camera objects (length n_views)
    points_2d: (T, V, J, 2) projected 2D keypoints
    confidences: (T, V, J) visibility mask (1.0 if camera sees joint, 0.0 otherwise)
    """
    T, J, _ = joints_3d.shape

    # World-to-camera rotation/translation from 4x4 cam poses.
    R_actual = actual_cam_poses[:, :3, :3]  # (T, 3, 3)
    t_actual = actual_cam_poses[:, :3, 3]  # (T, 3)
    C_actual = -(R_actual.transpose(0, 2, 1) @ t_actual[..., None]).squeeze(-1)  # (T, 3)

    # Mean root (pelvis) and mean actual camera center define the rig.
    root = joints_3d[:, 0, :]
    mean_root = root.mean(axis=0)
    mean_cam = C_actual.mean(axis=0)
    radius = np.linalg.norm(mean_cam - mean_root)
    if radius < 0.1:
        radius = 5.0  # fallback for degenerate sequences
    height = mean_cam[2]

    cameras = []
    points_2d_views = []
    confidences_views = []

    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        cam_pos = mean_root + radius * np.array([np.cos(theta), np.sin(theta), 0.0])
        cam_pos[2] = height
        R, t = _look_at(cam_pos, mean_root)
        cameras.append(Camera(K=actual_K.copy(), R=R, t=t))

        # Project all frames with this static camera.
        p2d = _project(joints_3d, actual_K, R, t)  # (T, J, 2)
        if noise_std > 0:
            p2d = p2d + np.random.randn(*p2d.shape) * noise_std

        points_2d_views.append(p2d)

        # A joint is visible if its depth is positive.
        X_cam = (R @ joints_3d.reshape(-1, 3).T).T + t
        visible = X_cam[:, 2] > 0
        visible = visible.reshape(T, J).astype(np.float32)
        confidences_views.append(visible)

    points_2d = np.stack(points_2d_views, axis=1)  # (T, V, J, 2)
    confidences = np.stack(confidences_views, axis=1)  # (T, V, J)
    return cameras, points_2d, confidences


def _build_actual_single(
    joints_3d: np.ndarray,
    actual_cam_poses: np.ndarray,
    actual_K: np.ndarray,
) -> tuple:
    """Build a single-view output using the real moving camera.

    Returns
    -------
    cameras: list with one Camera object (first-frame extrinsics as a static placeholder)
    points_2d: (T, 1, J, 2)
    confidences: (T, 1, J)
    per_frame: dict with camera_K_frames, camera_R_frames, camera_t_frames
    """
    T, J, _ = joints_3d.shape
    R = actual_cam_poses[:, :3, :3]  # (T, 3, 3)
    t = actual_cam_poses[:, :3, 3]   # (T, 3)

    # Project each frame with its own camera.
    p2d = np.zeros((T, J, 2), dtype=np.float64)
    visible = np.zeros((T, J), dtype=np.float32)
    for frame in range(T):
        p2d[frame] = _project(joints_3d[frame], actual_K, R[frame], t[frame])
        X_cam = (R[frame] @ joints_3d[frame].T).T + t[frame]
        visible[frame] = (X_cam[:, 2] > 0).astype(np.float32)

    # Static placeholder from first frame.
    placeholder = Camera(K=actual_K.copy(), R=R[0].copy(), t=t[0].copy())
    cameras = [placeholder]

    per_frame = {
        "camera_K_frames": np.tile(actual_K.copy(), (T, 1, 1)).astype(np.float64),
        "camera_R_frames": R.astype(np.float64),
        "camera_t_frames": t.astype(np.float64),
    }

    return cameras, p2d[:, None, :, :], visible[:, None, :], per_frame


def convert_3dpw_sequence(
    pkl_path: Path,
    output_path: Path,
    mode: str = "pseudo",
    n_views: int = 4,
    person_idx: int = 0,
    noise_std: float = 0.0,
) -> Path:
    """Convert one 3DPW .pkl sequence to a canonical or single-view ``.npz``.

    Args:
        pkl_path: path to the 3DPW sequence .pkl file.
        output_path: destination path.
        mode: ``pseudo`` or ``actual``.
        n_views: number of virtual views for pseudo mode.
        person_idx: which actor to extract (3DPW sequences may contain two people).
    """
    pkl_path = Path(pkl_path)
    output_path = Path(output_path)

    with open(pkl_path, "rb") as f:
        seq = pickle.load(f, encoding="latin1")

    for key in ("jointPositions", "cam_poses", "cam_intrinsics"):
        if key not in seq:
            raise KeyError(f"Missing '{key}' in {pkl_path}")

    joint_positions = seq["jointPositions"][person_idx]  # (T, 72)
    T = joint_positions.shape[0]
    joints_3d = joint_positions.reshape(T, 24, 3).astype(np.float64)

    actual_cam_poses = seq["cam_poses"].astype(np.float64)  # (T, 4, 4)
    actual_K = seq["cam_intrinsics"].astype(np.float64)  # (3, 3)

    if joints_3d.shape[0] != actual_cam_poses.shape[0]:
        raise ValueError(
            f"Mismatched frame counts: joints {joints_3d.shape[0]} vs cam_poses {actual_cam_poses.shape[0]}"
        )

    if mode == "pseudo":
        cameras, points_2d, confidences = _build_pseudo_rig(
            joints_3d, actual_cam_poses, actual_K, n_views, noise_std=noise_std
        )
        per_frame = {}
    elif mode == "actual":
        cameras, points_2d, confidences, per_frame = _build_actual_single(
            joints_3d, actual_cam_poses, actual_K
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points_2d=points_2d.astype(np.float64),
        confidences=confidences.astype(np.float32),
        joints_3d=joints_3d.astype(np.float64),
        camera_K=np.stack([cam.K for cam in cameras], axis=0).astype(np.float64),
        camera_R=np.stack([cam.R for cam in cameras], axis=0).astype(np.float64),
        camera_t=np.stack([cam.t for cam in cameras], axis=0).astype(np.float64),
        **per_frame,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a 3DPW sequence .pkl to a multi-view / single-view .npz."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a 3DPW .pkl file or to a directory of .pkl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz path or output directory when --input is a directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pseudo",
        choices=["pseudo", "actual"],
        help="Output mode: pseudo multi-view or actual single-view.  Default: pseudo.",
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=4,
        help="Number of pseudo views (only used in pseudo mode).  Default: 4.",
    )
    parser.add_argument(
        "--person_idx",
        type=int,
        default=0,
        help="Which actor to extract (3DPW sequences may contain two people).  Default: 0.",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="Pixel std-dev of Gaussian noise added to pseudo-view 2D projections.  Default: 0.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        # Batch mode: mirror directory structure under output_path.
        pkl_files = sorted(input_path.rglob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {input_path}")
        for pkl_file in pkl_files:
            # Relative path from input dir.
            rel = pkl_file.relative_to(input_path)
            out_file = output_path / rel.with_suffix("").with_name(
                rel.stem + f"_{args.mode}"
            ).with_suffix(".npz")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                out = convert_3dpw_sequence(
                    pkl_file,
                    out_file,
                    mode=args.mode,
                    n_views=args.n_views,
                    person_idx=args.person_idx,
                    noise_std=args.noise_std,
                )
                print(f"Saved {out}")
            except Exception as exc:
                print(f"FAILED {pkl_file}: {exc}")
    else:
        out = convert_3dpw_sequence(
            input_path,
            output_path,
            mode=args.mode,
            n_views=args.n_views,
            person_idx=args.person_idx,
            noise_std=args.noise_std,
        )
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
