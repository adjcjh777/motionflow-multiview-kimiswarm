"""Generate synthetic multi-view 3D pose training data from SMPL with randomized rigs.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/generate_synthetic_multiview_dataset.py \
        --n_sequences 500 --frames_per_seq 30 --output outputs/synthetic_multiview_dataset.npz
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import smplx
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera


def make_random_cameras(rng, n_views: int = 4):
    """Return a randomized calibrated camera rig."""
    radius = rng.uniform(3.0, 6.0)
    height = rng.uniform(0.5, 2.5)
    focal = rng.uniform(600.0, 1200.0)
    cx = rng.uniform(300.0, 340.0)
    cy = rng.uniform(220.0, 260.0)
    phi_base = rng.uniform(np.pi / 6, np.pi / 3)

    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views + rng.uniform(-0.1, 0.1)
        phi = phi_base + rng.uniform(-0.1, 0.1)
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = cx
        K[1, 2] = cy

        c = radius * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        c[2] += height
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def project_points(points_3d: np.ndarray, camera: Camera):
    """Project (J, 3) to (J, 2)."""
    P = camera.projection_matrix
    X_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return x


def generate_sequence(
    smpl_model,
    betas,
    cameras,
    n_frames: int,
    rng: np.random.Generator,
    noise_std: float = 0.5,
    device: torch.device = None,
    occlusion_rate: float = 0.1,
    outlier_rate: float = 0.02,
):
    """Generate one synthetic sequence of SMPL motion."""
    n_views = len(cameras)
    J = 17

    # Smooth random pose via Brownian motion on latent angles.
    body_pose_latents = []
    global_orient_latents = []
    transl_seq = []
    for _ in range(n_frames):
        body_pose_latents.append(rng.normal(0, 0.3, size=69).astype(np.float32))
        global_orient_latents.append(rng.normal(0, 0.2, size=3).astype(np.float32))
        transl_seq.append((rng.normal(0, 0.2, size=3) + np.array([0.0, 0.0, 1.0])).astype(np.float32))

    joints_3d_list = []
    points_2d_list = []
    confidences_list = []

    for f in range(n_frames):
        body_pose = torch.from_numpy(body_pose_latents[f]).reshape(1, 69).to(device)
        global_orient = torch.from_numpy(global_orient_latents[f]).reshape(1, 3).to(device)
        transl = torch.from_numpy(transl_seq[f]).reshape(1, 3).to(device)

        with torch.no_grad():
            output = smpl_model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            )
        joints_3d = output.joints[0, :J].cpu().numpy()  # (J, 3)
        joints_3d_list.append(joints_3d)

        points_2d_v = []
        confidences_v = []
        for v, cam in enumerate(cameras):
            x = project_points(joints_3d, cam)
            x += rng.normal(0, noise_std, size=x.shape)

            # Occlusion: randomly drop some joints in this view
            if occlusion_rate > 0:
                occ_mask = rng.random(J) < occlusion_rate
                x[occ_mask] = 0.0
                conf = rng.uniform(0.8, 1.0, size=J)
                conf[occ_mask] = 0.0
            else:
                conf = rng.uniform(0.8, 1.0, size=J)

            # Outliers: occasional wild keypoints
            if outlier_rate > 0:
                outlier_mask = rng.random(J) < outlier_rate
                x[outlier_mask] += rng.normal(0, 50.0, size=(outlier_mask.sum(), 2))
                conf[outlier_mask] = 0.0

            points_2d_v.append(x)
            confidences_v.append(conf)
        points_2d_list.append(np.stack(points_2d_v, axis=0))  # (V, J, 2)
        confidences_list.append(np.stack(confidences_v, axis=0))  # (V, J)

    joints_3d = np.stack(joints_3d_list, axis=0)  # (T, J, 3)
    points_2d = np.stack(points_2d_list, axis=0)  # (T, V, J, 2)
    confidences = np.stack(confidences_list, axis=0)  # (T, V, J)
    return joints_3d, points_2d, confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sequences", type=int, default=500)
    parser.add_argument("--frames_per_seq", type=int, default=30)
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--occlusion_rate", type=float, default=0.1)
    parser.add_argument("--outlier_rate", type=float, default=0.02)
    parser.add_argument("--output", type=str, default="outputs/synthetic_multiview_dataset.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    smpl_model = smplx.SMPL("data/smpl/SMPL_NEUTRAL.pkl", batch_size=1).to(device)
    betas0 = torch.randn(1, 10).to(device) * 0.1

    rng = np.random.default_rng(2025)

    all_joints_3d = []
    all_points_2d = []
    all_confidences = []
    all_camera_K = []
    all_camera_R = []
    all_camera_t = []

    for seq_idx in range(args.n_sequences):
        cameras = make_random_cameras(rng, n_views=args.n_views)
        joints_3d, points_2d, confidences = generate_sequence(
            smpl_model, betas0, cameras, args.frames_per_seq, rng,
            args.noise_std, device, args.occlusion_rate, args.outlier_rate
        )
        n_frames = joints_3d.shape[0]
        K_arr = np.stack([cam.K for cam in cameras], axis=0)  # (V, 3, 3)
        R_arr = np.stack([cam.R for cam in cameras], axis=0)
        t_arr = np.stack([cam.t for cam in cameras], axis=0)
        all_joints_3d.append(joints_3d)
        all_points_2d.append(points_2d)
        all_confidences.append(confidences)
        all_camera_K.append(np.tile(K_arr[None], (n_frames, 1, 1, 1)))
        all_camera_R.append(np.tile(R_arr[None], (n_frames, 1, 1, 1)))
        all_camera_t.append(np.tile(t_arr[None], (n_frames, 1, 1)))
        if (seq_idx + 1) % 50 == 0:
            print(f"Generated {seq_idx + 1}/{args.n_sequences} sequences")

    data = {
        "joints_3d": np.concatenate(all_joints_3d, axis=0),  # (T_total, J, 3)
        "points_2d": np.concatenate(all_points_2d, axis=0),  # (T_total, V, J, 2)
        "confidences": np.concatenate(all_confidences, axis=0),  # (T_total, V, J)
        "camera_K": np.concatenate(all_camera_K, axis=0),  # (T_total, V, 3, 3)
        "camera_R": np.concatenate(all_camera_R, axis=0),  # (T_total, V, 3, 3)
        "camera_t": np.concatenate(all_camera_t, axis=0),  # (T_total, V, 3)
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)
    print(f"Saved synthetic dataset to {output_path}")
    print(f"Total frames: {data['joints_3d'].shape[0]}")


if __name__ == "__main__":
    main()
