"""CPU-only frozen/oracle gate for shared principal-point residuals.

This is intentionally not a CTMF solver. It tests one necessary condition:
whether a cross-fitted shared 2-D offset can separate camera-wide principal-
point drift from sparse local keypoint outliers when the 3-D pose is fixed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _robust_shared_offset(
    residuals: np.ndarray,
    noise_std: float,
    prior_precision: float,
    huber_delta: float,
    n_iters: int = 3,
) -> tuple[np.ndarray, float]:
    """Fit a shared 2-D offset with a small fixed Huber IRLS loop."""
    theta = residuals.mean(axis=0)
    hessian = prior_precision
    for _ in range(n_iters):
        normalized = np.linalg.norm(residuals - theta, axis=-1) / noise_std
        huber_weight = np.minimum(
            1.0, huber_delta / np.maximum(normalized, 1e-12)
        )
        precision_weight = huber_weight / noise_std**2
        hessian = prior_precision + float(precision_weight.sum())
        theta = (precision_weight[:, None] * residuals).sum(axis=0) / hessian
    return theta, hessian


def pp_crossfit_predictive_scores(
    residuals: np.ndarray,
    noise_std: float = 1.0,
    prior_std: float = 25.0,
    huber_delta: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two-fold predictive residuals and Mahalanobis scores.

    Args:
        residuals: ``(N, V, 2)`` residuals defined as observed minus the
            projection of frozen oracle 3-D points through nominal cameras.

    ``N=1`` is deliberately unsupported: one observation contains no shared
    evidence independent of the point being scored.
    """
    n_points, n_views, _ = residuals.shape
    if n_points < 2:
        raise ValueError("P0a abstains when N=1")

    predictive_residual = np.empty_like(residuals)
    score = np.empty((n_points, n_views), dtype=np.float64)
    indices = np.arange(n_points)
    prior_precision = 1.0 / prior_std**2

    for view in range(n_views):
        for fold in (0, 1):
            test = indices % 2 == fold
            theta, hessian = _robust_shared_offset(
                residuals[~test, view],
                noise_std=noise_std,
                prior_precision=prior_precision,
                huber_delta=huber_delta,
            )
            error = residuals[test, view] - theta
            predictive_variance = noise_std**2 + 1.0 / hessian
            predictive_residual[test, view] = error
            score[test, view] = np.sum(error**2, axis=-1) / predictive_variance

    return predictive_residual, score


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive
    return float(
        (ranks[labels].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _rms_ratio(after: np.ndarray, before: np.ndarray) -> float:
    return float(np.sqrt(np.mean(after**2)) / np.sqrt(np.mean(before**2)))


def _local_signal(
    rng: np.random.Generator,
    n_points: int,
    rate: float,
    target_rms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sparse outliers whose realized signal RMS exactly matches target."""
    n_outliers = max(1, int(round(rate * n_points)))
    labels = np.zeros(n_points, dtype=bool)
    labels[rng.choice(n_points, size=n_outliers, replace=False)] = True
    direction = rng.normal(size=(n_outliers, 2))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    magnitude = target_rms * np.sqrt(n_points / n_outliers)
    signal = np.zeros((n_points, 2), dtype=np.float64)
    signal[labels] = magnitude * direction
    return signal, labels


def _run_setting(
    rng: np.random.Generator,
    n_points: int,
    n_clips: int,
    n_views: int,
    noise_std: float,
    outlier_rate: float,
    drift: np.ndarray,
    score_threshold: float,
) -> dict[str, float]:
    shared_ratio = []
    local_ratio = []
    shuffled_ratio = []
    independent_ratio = []
    mixed_labels = []
    mixed_raw_score = []
    mixed_predictive_score = []
    shared_raw_flags = []
    shared_predictive_flags = []
    mixed_raw_flags = []
    mixed_predictive_flags = []
    drift_rms = float(np.linalg.norm(drift))

    for _ in range(n_clips):
        noise = rng.normal(0.0, noise_std, size=(n_points, n_views, 2))

        shared = noise.copy()
        shared[:, 0] += drift
        predicted, score = pp_crossfit_predictive_scores(shared, noise_std=noise_std)
        shared_ratio.append(_rms_ratio(predicted[:, 0], shared[:, 0]))
        shared_raw_score = np.sum(shared[:, 0] ** 2, axis=-1) / noise_std**2
        shared_raw_flags.extend(shared_raw_score > score_threshold)
        shared_predictive_flags.extend(score[:, 0] > score_threshold)

        local_signal, labels = _local_signal(
            rng, n_points, outlier_rate, target_rms=drift_rms
        )
        local = noise.copy()
        local[:, 0] += local_signal
        predicted, _ = pp_crossfit_predictive_scores(local, noise_std=noise_std)
        local_ratio.append(_rms_ratio(predicted[:, 0], local[:, 0]))

        mixed = shared.copy()
        mixed[:, 0] += local_signal
        predicted, score = pp_crossfit_predictive_scores(mixed, noise_std=noise_std)
        mixed_labels.extend(labels)
        raw_score = np.sum(mixed[:, 0] ** 2, axis=-1) / noise_std**2
        mixed_raw_score.extend(raw_score)
        mixed_predictive_score.extend(score[:, 0])
        mixed_raw_flags.extend(raw_score > score_threshold)
        mixed_predictive_flags.extend(score[:, 0] > score_threshold)

        shuffled = shared.copy()
        drift_observation = np.zeros((n_points, n_views), dtype=bool)
        drift_observation[:, 0] = True
        shuffled_drift_observation = np.empty_like(drift_observation)
        for point in range(n_points):
            permutation = rng.permutation(n_views)
            shuffled[point] = shared[point, permutation]
            shuffled_drift_observation[point] = drift_observation[point, permutation]
        predicted, _ = pp_crossfit_predictive_scores(shuffled, noise_std=noise_std)
        shuffled_ratio.append(
            _rms_ratio(
                predicted[shuffled_drift_observation],
                shuffled[shuffled_drift_observation],
            )
        )

        direction = rng.normal(size=(n_points, 2))
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        independent = noise.copy()
        independent[:, 0] += drift_rms * direction
        predicted, _ = pp_crossfit_predictive_scores(
            independent, noise_std=noise_std
        )
        independent_ratio.append(
            _rms_ratio(predicted[:, 0], independent[:, 0])
        )

    labels = np.asarray(mixed_labels, dtype=bool)
    raw_flags = np.asarray(mixed_raw_flags, dtype=bool)
    predictive_flags = np.asarray(mixed_predictive_flags, dtype=bool)
    raw_auc = _auc(labels, np.asarray(mixed_raw_score))
    predictive_auc = _auc(
        labels, np.asarray(mixed_predictive_score)
    )
    return {
        "shared_ratio_median": float(np.median(shared_ratio)),
        "local_ratio_median": float(np.median(local_ratio)),
        "mixed_raw_auc": raw_auc,
        "mixed_predictive_auc": predictive_auc,
        "auc_gain": predictive_auc - raw_auc,
        "shared_raw_flag_rate": float(np.mean(shared_raw_flags)),
        "shared_predictive_flag_rate": float(np.mean(shared_predictive_flags)),
        "mixed_raw_true_positive_rate": float(np.mean(raw_flags[labels])),
        "mixed_raw_false_positive_rate": float(np.mean(raw_flags[~labels])),
        "mixed_predictive_true_positive_rate": float(
            np.mean(predictive_flags[labels])
        ),
        "mixed_predictive_false_positive_rate": float(
            np.mean(predictive_flags[~labels])
        ),
        "shuffled_group_ratio_median": float(np.median(shuffled_ratio)),
        "independent_drift_ratio_median": float(np.median(independent_ratio)),
    }


def run_probe(seed: int = 20260807, n_clips: int = 256) -> dict:
    rng = np.random.default_rng(seed)
    # Frozen numerical reference equal to the chi-square(2) 95th percentile.
    # Huber cross-fit scores are judged by measured rates, not assumed calibrated.
    score_threshold = 5.991464547107979
    metrics = {
        str(n_points): _run_setting(
            rng,
            n_points=n_points,
            n_clips=n_clips,
            n_views=4,
            noise_std=1.0,
            outlier_rate=0.2,
            drift=np.array([5.0, -5.0]),
            score_threshold=score_threshold,
        )
        for n_points in (17, 153)
    }
    decisive = metrics["153"]
    try:
        pp_crossfit_predictive_scores(np.zeros((1, 4, 2), dtype=np.float64))
        n1_abstains = False
    except ValueError:
        n1_abstains = True
    gates = {
        "n1_abstains": n1_abstains,
        "shared_ratio_le_0_30": decisive["shared_ratio_median"] <= 0.30,
        "local_ratio_ge_0_80": decisive["local_ratio_median"] >= 0.80,
        "predictive_auc_ge_0_80": decisive["mixed_predictive_auc"] >= 0.80,
        "shared_raw_flag_rate_ge_0_80": decisive["shared_raw_flag_rate"] >= 0.80,
        "shared_predictive_flag_rate_le_0_10": decisive[
            "shared_predictive_flag_rate"
        ]
        <= 0.10,
        "mixed_predictive_tpr_ge_0_80": decisive[
            "mixed_predictive_true_positive_rate"
        ]
        >= 0.80,
        "mixed_predictive_fpr_le_0_10": decisive[
            "mixed_predictive_false_positive_rate"
        ]
        <= 0.10,
        "mixed_fpr_reduction_ge_0_50": (
            decisive["mixed_raw_false_positive_rate"]
            - decisive["mixed_predictive_false_positive_rate"]
        )
        >= 0.50,
        "shuffle_ratio_ge_0_80": decisive["shuffled_group_ratio_median"] >= 0.80,
        "independent_ratio_ge_0_80": decisive["independent_drift_ratio_median"]
        >= 0.80,
    }
    return {
        "protocol": {
            "seed": seed,
            "clips": n_clips,
            "views": 4,
            "points": [1, 17, 153],
            "noise_std_px": 1.0,
            "principal_point_drift_px": [5.0, -5.0],
            "local_outlier_rate": 0.2,
            "local_score_threshold": score_threshold,
            "n1_status": "ABSTAIN",
        },
        "metrics": metrics,
        "gates": gates,
        "decision": "P0A_DIAGNOSTIC_PASS" if all(gates.values()) else "P0A_DIAGNOSTIC_FAIL",
        "incremental_utility": "INCONCLUSIVE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--clips", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run_probe(seed=args.seed, n_clips=args.clips), indent=2))


if __name__ == "__main__":
    main()
