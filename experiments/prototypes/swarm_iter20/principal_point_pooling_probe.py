#!/usr/bin/env python3
"""Pure NumPy probe for principal-point pooling mask semantics."""

from __future__ import annotations

import json

import numpy as np


def historical_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    revived = weights[:, None] + 1e-8
    return (values * revived).sum(axis=0) / revived.sum(axis=0)


def exact_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    numerator = (values * weights[:, None]).sum(axis=0)
    denominator = weights.sum()
    return numerator / (1.0 if denominator == 0 else denominator)


def run_probe() -> dict[str, object]:
    features = np.array([[2.0, 20.0], [100.0, 1000.0]])
    points = np.array([[2.0, 20.0], [10.0, 100.0]])
    zero = np.zeros(2)
    soft_confidence = np.array([0.5, 1.0])

    return {
        "all_zero_feature_weights": {
            "historical": historical_mean(features, zero).tolist(),
            "exact": exact_mean(features, zero).tolist(),
        },
        "all_zero_raw_confidence": {
            "historical": historical_mean(points, zero).tolist(),
            "exact": exact_mean(points, zero).tolist(),
        },
        "soft_confidence": {
            "historical_confidence_squared": historical_mean(
                points, soft_confidence**2
            ).tolist(),
            "exact_confidence_once": exact_mean(points, soft_confidence).tolist(),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_probe(), indent=2))
