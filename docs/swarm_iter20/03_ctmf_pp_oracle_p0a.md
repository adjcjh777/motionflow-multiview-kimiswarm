# Swarm Iteration 20 — CTMF PP-Only Oracle P0a

**Date:** 2026-08-07
**Decision:** **P0A DIAGNOSTIC PASS; INCREMENTAL UTILITY INCONCLUSIVE**
**Compute:** NumPy CPU only; no learning and no GPU.

## Claim under test

Before implementing a joint camera/pose solver, test the most favorable special
case: frozen oracle 3D and a two-dimensional principal-point tangent. A shared
camera offset should be removable from cross-fitted predictive residuals while
sparse local keypoint outliers should remain detectable.

Passing this gate would only permit a P0b. It would not establish CTMF novelty,
MPJPE improvement, unknown-3D behavior, or true marginalization.

## Protocol

The executable probe is
`experiments/prototypes/swarm_iter20/ctmf_pp_oracle_probe.py`.

- Seed `20260807`, 256 clips, four camera groups.
- `N=T*J` in `{1, 17, 153}`; `N=1` must abstain.
- Detector noise: isotropic `1 px`.
- Shared condition: camera 0 has `delta_pp=(5,-5) px` for all points.
- Local condition: `20%` sparse random-direction outliers, scaled so their
  realized signal RMS exactly matches the shared offset signal.
- Mixed condition: shared offset plus the same local outliers.
- Scoring: two-fold cross-fit with three Huber IRLS steps. A point never helps
  fit the offset used to score itself.
- Local gating uses the fixed numerical reference `5.9915`, equal to the
  chi-square(2) 95th percentile; it is not selected from these results. Huber
  cross-fit scores are evaluated by measured rates, not claimed to be strictly
  chi-square calibrated.
- Negative controls: per-point camera-group shuffle and per-point independent
  offsets. A global camera permutation is not used because it only renames the
  groups.

## Results

| `N` | Shared ratio | Matched-local ratio | Raw AUC | Predictive AUC | AUC gain | Shuffled ratio | Independent ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.1991 | 1.0033 | 0.9972 | 1.0000 | 0.0028 | 0.8671 | 1.0813 |
| 153 | 0.1973 | 1.0003 | 0.9869 | 1.0000 | **0.0131** | 0.8805 | 1.0144 |

The AUC gain is descriptive, not a pass/fail gate: raw AUC `0.9869` leaves a
theoretical maximum gain of only `0.0131`.

At the fixed threshold, the operational rates are:

| `N` | Shared raw flag | Shared predictive flag | Mixed raw TPR/FPR | Mixed predictive TPR/FPR |
|---:|---:|---:|---:|---:|
| 17 | 1.0000 | 0.0503 | 1.0000 / 1.0000 | 1.0000 / 0.0703 |
| 153 | 0.9999 | 0.0498 | 1.0000 / 1.0000 | 1.0000 / 0.0530 |

At `N=153`, all diagnostic gates pass:

- shared ratio `<=0.30`;
- local ratio `>=0.80`;
- predictive AUC and mixed TPR `>=0.80`;
- shared and mixed predictive false-positive rates `<=0.10`;
- mixed false-positive rate improves by at least `0.50` over raw scoring;
- shuffled and independent ratios `>=0.80`;
- `N=1` abstains.

## Interpretation

The shared-offset estimator behaves correctly, and its benefit disappears when
camera grouping is destroyed. Raw residual ranking already detects the sparse
outliers almost perfectly, so AUC cannot measure incremental value here. At a
fixed clean-noise threshold, however, raw scoring mistakes virtually every
shared-drift observation for a local anomaly, while predictive scoring retains
all outliers with about `5.3%` false positives.

This is a **P0a implementation diagnostic PASS**, not CTMF GO. The special case
is only robust shared 2D centering and does not establish novelty or superiority
over strong pointwise/camera-correction baselines. Keep the broader CTMF claim
stopped and do not proceed to GPU training from this result.
