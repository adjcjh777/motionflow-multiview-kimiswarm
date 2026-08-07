#!/usr/bin/env python3
"""Pure NumPy diagnostic for variable-view DLT masking.

The probe isolates geometry from the learned feature/attention/residual path.
It compares an exact active-camera solve with the padded zero-observation path
used by variable-view inference, including the current DLT epsilon and caller
weight floor.
"""

from __future__ import annotations

import itertools
import json

import numpy as np


def make_four_view_rig() -> np.ndarray:
    intrinsic = np.array(
        [[1145.0, 0.0, 512.0], [0.0, 1145.0, 512.0], [0.0, 0.0, 1.0]]
    )
    projections = []
    for index in range(4):
        angle = 2.0 * np.pi * index / 4.0
        center = np.array([4.0 * np.cos(angle), 4.0 * np.sin(angle), 1.7])
        forward = -center / np.linalg.norm(center)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rotation = np.stack([right, up, -forward])
        translation = -rotation @ center
        projections.append(intrinsic @ np.column_stack([rotation, translation]))
    return np.stack(projections)


def project(point_3d: np.ndarray, projections: np.ndarray) -> np.ndarray:
    homogeneous = np.append(point_3d, 1.0)
    image = projections @ homogeneous
    return image[:, :2] / image[:, 2, None]


def weighted_dlt(
    points_2d: np.ndarray,
    projections: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    rows = []
    for point, projection, weight in zip(points_2d, projections, weights):
        x_coord, y_coord = point
        scale = np.sqrt(max(float(weight), 0.0) + epsilon)
        rows.append(scale * (x_coord * projection[2] - projection[0]))
        rows.append(scale * (y_coord * projection[2] - projection[1]))
    system = np.stack(rows)
    solution, *_ = np.linalg.lstsq(system[:, :3], -system[:, 3], rcond=None)
    return solution


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean_mm": float(array.mean()),
        "p95_mm": float(np.quantile(array, 0.95)),
        "max_mm": float(array.max()),
    }


def run_probe(seed: int = 20, samples: int = 1000) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    projections = make_four_view_rig()
    points_3d = np.column_stack(
        [
            rng.uniform(-0.7, 0.7, samples),
            rng.uniform(-0.7, 0.7, samples),
            rng.uniform(-0.5, 0.8, samples),
        ]
    )

    results: dict[str, object] = {"seed": seed, "samples": samples, "noise_std_px": 1.0}
    for active_count in (2, 3):
        errors = {
            "active_subset": [],
            "padded_exact_zero": [],
            "padded_dlt_epsilon": [],
            "padded_weight_floor": [],
            "confidence_only_leak": [],
            "all_four_views": [],
        }
        for active_indices in itertools.combinations(range(4), active_count):
            active = np.zeros(4, dtype=bool)
            active[list(active_indices)] = True
            zero_weights = active.astype(np.float64)
            floored_weights = np.maximum(zero_weights, 1e-4)

            for ground_truth in points_3d:
                observed = project(ground_truth, projections)
                observed += rng.normal(0.0, 1.0, observed.shape)
                zero_padded = observed.copy()
                zero_padded[~active] = 0.0

                estimates = {
                    "active_subset": weighted_dlt(
                        observed[active], projections[active], np.ones(active_count), 0.0
                    ),
                    "padded_exact_zero": weighted_dlt(
                        zero_padded, projections, zero_weights, 0.0
                    ),
                    "padded_dlt_epsilon": weighted_dlt(
                        zero_padded, projections, zero_weights, 1e-6
                    ),
                    "padded_weight_floor": weighted_dlt(
                        zero_padded, projections, floored_weights, 1e-6
                    ),
                    "confidence_only_leak": weighted_dlt(
                        observed, projections, floored_weights, 1e-6
                    ),
                    "all_four_views": weighted_dlt(
                        observed, projections, np.ones(4), 0.0
                    ),
                }
                for name, estimate in estimates.items():
                    errors[name].append(float(np.linalg.norm(estimate - ground_truth) * 1000.0))

        summaries = {name: summarize(values) for name, values in errors.items()}
        summaries["floor_increment_over_subset_mm"] = (
            summaries["padded_weight_floor"]["mean_mm"]
            - summaries["active_subset"]["mean_mm"]
        )
        results[str(active_count)] = summaries

    return results


if __name__ == "__main__":
    print(json.dumps(run_probe(), indent=2))
