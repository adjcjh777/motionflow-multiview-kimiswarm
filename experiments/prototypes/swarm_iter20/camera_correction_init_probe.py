#!/usr/bin/env python3
"""NumPy distribution probe for the historical random correction-head init."""

from __future__ import annotations

import json

import numpy as np


def default_linear(rng: np.random.Generator, value: np.ndarray, out_dim: int) -> np.ndarray:
    bound = 1.0 / np.sqrt(value.shape[-1])
    weight = rng.uniform(-bound, bound, (out_dim, value.shape[-1]))
    bias = rng.uniform(-bound, bound, out_dim)
    return value @ weight.T + bias


def correction_head(
    rng: np.random.Generator,
    value: np.ndarray,
    hidden: int,
    out_dim: int,
) -> np.ndarray:
    value = np.maximum(default_linear(rng, value, hidden), 0.0)
    value = np.maximum(default_linear(rng, value, hidden), 0.0)
    return np.tanh(default_linear(rng, value, out_dim))


def historical_pp_output(seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    raw_descriptor = np.array([320.0, 240.0, 1.0, 320.0, 240.0, 800.0, 800.0, 0.0])
    projected = default_linear(rng, raw_descriptor, 64)
    pp_delta = correction_head(rng, projected, hidden=64, out_dim=2) * 20.0
    focal_delta = correction_head(rng, projected, hidden=64, out_dim=1)[0] * 0.1
    return pp_delta, float(focal_delta)


def run_probe(samples: int = 1000) -> dict[str, object]:
    outputs = [historical_pp_output(seed) for seed in range(samples)]
    pp_abs = np.abs(np.concatenate([item[0] for item in outputs]))
    focal_abs = np.abs(np.array([item[1] for item in outputs]))
    return {
        "samples": samples,
        "historical_default_linear_distribution": {
            "pp_abs_mean_px": float(pp_abs.mean()),
            "pp_abs_median_px": float(np.median(pp_abs)),
            "pp_fraction_over_19px": float((pp_abs > 19.0).mean()),
            "focal_abs_mean_scale_delta": float(focal_abs.mean()),
            "focal_fraction_over_0_095": float((focal_abs > 0.095).mean()),
        },
        "zero_initialized_final_linear": {
            "pp_delta_px": [0.0, 0.0],
            "focal_scale": 1.0,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_probe(), indent=2))
