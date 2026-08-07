"""Focused CPU geometry test for epipolar attention bias."""

import numpy as np
import torch

from motionflow_mv.fusion.epipolar_attention_bias import compute_epipolar_distance


def test_exact_correspondences_have_zero_epipolar_distance():
    n_views = 4
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        center = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -center / np.linalg.norm(center)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rotation = np.stack([right, up, -forward])
        translation = -rotation @ center
        intrinsic = np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
        )
        cameras.append((intrinsic, rotation, translation))

    points_3d = np.array(
        [[0.1, 0.2, 0.4], [-0.3, 0.1, 0.8], [0.2, -0.2, 1.1]]
    )
    points_2d = []
    for intrinsic, rotation, translation in cameras:
        camera_points = points_3d @ rotation.T + translation
        projected = (intrinsic @ camera_points.T).T
        points_2d.append(projected[:, :2] / projected[:, 2:3])

    K = torch.tensor(np.stack([camera[0] for camera in cameras])).unsqueeze(0)
    R = torch.tensor(np.stack([camera[1] for camera in cameras])).unsqueeze(0)
    t = torch.tensor(np.stack([camera[2] for camera in cameras])).unsqueeze(0)
    points_2d = torch.tensor(np.stack(points_2d)).unsqueeze(0)

    distances = compute_epipolar_distance(K, R, t, points_2d)[0]
    off_diagonal = distances[~torch.eye(n_views, dtype=torch.bool)]

    assert float(off_diagonal.max()) < 1e-6

    corrupted = points_2d.clone()
    corrupted[0, 1, 0, 1] += 20.0
    corrupted_distances = compute_epipolar_distance(K, R, t, corrupted)

    assert float(corrupted_distances[0, 0, 1, 0]) > 1.0
