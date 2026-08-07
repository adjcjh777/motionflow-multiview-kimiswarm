#!/usr/bin/env python3
"""Pure NumPy probe for camera-centric scale translation equivariance."""

from __future__ import annotations

import json

import numpy as np


def historical_scale(point, scales):
    return scales.mean() * point


def ray_depth_scale(point, centers, scales, weights):
    rays = point[None, :] - centers
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    depth = np.sum((point[None, :] - centers) * rays, axis=-1)
    delta = (scales - 1.0)[:, None] * depth[:, None] * rays
    return point + np.sum(weights[:, None] * delta, axis=0) / weights.sum()


def run_probe() -> dict[str, object]:
    point = np.array([2.0, 3.0, 5.0])
    centers = np.array([[0.0, 0.0, 0.0], [4.0, 1.0, 0.0]])
    scales = np.array([0.8, 1.1])
    weights = np.array([1.0, 0.5])
    shift = np.array([10.0, -4.0, 2.0])

    old = historical_scale(point, scales)
    old_shifted = historical_scale(point + shift, scales)
    corrected = ray_depth_scale(point, centers, scales, weights)
    corrected_shifted = ray_depth_scale(
        point + shift,
        centers + shift,
        scales,
        weights,
    )

    first = ray_depth_scale(point, centers, np.array([0.8, 1.2]), weights)
    swapped = ray_depth_scale(point, centers, np.array([1.2, 0.8]), weights)
    return {
        "historical_translation_error": (
            old_shifted - (old + shift)
        ).tolist(),
        "ray_depth_translation_error": (
            corrected_shifted - (corrected + shift)
        ).tolist(),
        "same_mean_swapped_scale_distance": float(np.linalg.norm(first - swapped)),
        "identity_scale_output": ray_depth_scale(
            point, centers, np.ones(2), weights
        ).tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(run_probe(), indent=2))
