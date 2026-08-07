# Swarm Iteration 20 — Epipolar Index Probe

**Date:** 2026-08-07  
**Scope:** CPU-only geometry exploration; no training and no GPU experiment.

## Question

Does the epipolar distance used by v3 attention satisfy the defining geometry
constraint on exact multi-view correspondences?

For a source view `s` and destination view `d`, the required constraint is

`x_d^T F_{d,s} x_s = 0`.

## P0 probe

A four-camera circular rig projected three exact 3D points into every view. The
probe then compared the repository implementation with the same fundamental
matrices evaluated using the defining source/destination ordering.

| Computation | Off-diagonal mean | Off-diagonal max |
|---|---:|---:|
| Previous implementation | 7.7513 px | 31.5442 px |
| Correct `x_d^T F_{d,s} x_s` ordering | 6.0e-14 px | 1.8e-13 px |

The previous implementation built `F` in `(destination, source)` order but
multiplied it by the destination point and evaluated the result at the source
point. This computes `x_s^T F_{d,s} x_d`, which is not the epipolar constraint.

## Change

`compute_epipolar_distance` now forms the destination line from the source
point and evaluates that line at the destination point. A single focused CPU
test covers both the exact-correspondence invariant and a 20 px wrong-match
negative control.

## Research consequence

| Path | How the old distance is used | Evidence boundary after this fix |
|---|---|---|
| Bayesian Tri v2 | Auxiliary training loss | The documented 9.03/8.35 mm results remain old-recipe validation numbers; they do not validate epipolar consistency. |
| OmniMultiViewFusion v2 | Auxiliary training loss | Frozen-checkpoint inference does not change, but its training recipe used the wrong geometry when the loss weight was non-zero. |
| OmniMultiViewFusion v3 | Training loss and inference-time attention bias | An old checkpoint cannot be relabelled as the corrected mechanism by evaluating it with the new forward path. |
| Epipolar weight-head variants | Inference-time view-weight bias | Their behavior changes directly and requires a controlled retrain before any corrected-mechanism claim. |

- Any earlier result using epipolar bias or a non-zero epipolar-distance loss
  does not isolate the intended geometry mechanism and must not be cited as
  evidence for it. Its reported task metric can remain an empirical result of
  the old recipe, but not evidence for correct epipolar consistency.
- Results are unaffected only when the broken distance was excluded from both
  the training objective and inference path. In v3, setting
  `use_epipolar_bias=False` alone is insufficient if
  `epipolar_loss_weight` remained non-zero during training.
- The next evaluation should compare corrected epipolar bias against the exact
  reduction `use_epipolar_bias=False`; this iteration intentionally does not
  launch that GPU experiment.

## Decision

**GO** for the corrected geometry implementation. **STOP** any positive claim
about the old epipolar mechanism until a later controlled evaluation is run.
