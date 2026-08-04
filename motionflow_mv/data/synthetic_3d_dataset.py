"""Shared synthetic 3D multi-view dataset generator.

Produces calibrated multi-view 2D keypoints, confidences, and 3D ground truth
joints. All plugin training scripts can import from here to ensure the same
data distribution and scale.
"""

import numpy as np
import torch

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.triangulation import triangulate_confidence_weighted


def make_cameras(n_views: int = 5, rng: np.random.Generator = None):
    """Return a list of calibrated pinhole cameras on a circle."""
    if rng is None:
        rng = np.random.default_rng(123)
    cameras = []
    for i in range(n_views):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[:2, 2] = rng.uniform(300, 340, size=2)
        R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        theta = 2 * np.pi * i / n_views
        phi = np.pi / 3
        radius = 5.0
        c = radius * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        t = -R @ c
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def generate_sequence(
    n_frames: int = 10,
    n_views: int = 5,
    j: int = 17,
    rng: np.random.Generator = None,
    noise_std: float = 1.0,
):
    """Return one synthetic sequence: inputs (T, V, J, 3), baselines (T, J, 3), gt (T, J, 3)."""
    if rng is None:
        rng = np.random.default_rng(0)

    # Smooth base skeleton trajectory.
    base = rng.uniform(-1.0, 1.0, size=(j, 3)) * np.array([0.5, 0.8, 1.5])
    base[:, 2] += 3.0
    positions = [base.copy()]
    for _ in range(n_frames - 1):
        delta = rng.normal(0, 0.05, size=base.shape)
        base = base + delta
        positions.append(base.copy())
    positions = np.stack(positions, axis=0)  # (T, J, 3)

    cameras = make_cameras(n_views, rng)
    proj = [cam.projection_matrix for cam in cameras]

    inputs = []
    baselines = []
    for t in range(n_frames):
        X = positions[t]
        points_2d = []
        conf = []
        for cam in cameras:
            P = cam.projection_matrix
            X_h = np.hstack([X, np.ones((j, 1))])
            x_h = (P @ X_h.T).T
            x = x_h[:, :2] / x_h[:, 2:3]
            x += rng.normal(0, noise_std, size=x.shape)
            points_2d.append(x)
            conf.append(rng.uniform(0.5, 1.0, size=j))
        points_2d = np.stack(points_2d, axis=0)  # (V, J, 2)
        conf = np.stack(conf, axis=0)  # (V, J)
        inputs.append(np.concatenate([points_2d, conf[..., None]], axis=-1))

        # Baseline DLT from noisy points.
        baseline = np.zeros((j, 3), dtype=np.float64)
        proj_arr = np.stack(proj, axis=0)
        for joint_idx in range(j):
            baseline[joint_idx] = triangulate_confidence_weighted(
                points_2d[:, joint_idx, :],
                proj_arr,
                conf[:, joint_idx],
            )
        baselines.append(baseline)

    return (
        torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32),
        torch.tensor(np.stack(baselines, axis=0), dtype=torch.float32),
        torch.tensor(positions, dtype=torch.float32),
        cameras,
    )


def generate_dataset(
    n_seq: int,
    n_frames: int = 10,
    n_views: int = 5,
    j: int = 17,
    seed: int = 0,
    noise_std: float = 1.0,
):
    rng = np.random.default_rng(seed)
    X, B, Y = [], [], []
    for _ in range(n_seq):
        inp, base, gt, _ = generate_sequence(n_frames, n_views, j, rng, noise_std)
        X.append(inp)
        B.append(base)
        Y.append(gt)
    return torch.stack(X), torch.stack(B), torch.stack(Y)
