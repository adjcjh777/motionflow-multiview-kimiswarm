# v31: Geometry-Attention Refinement for Hierarchical Multi-View Fusion

## Problem Statement

v30 stabilised the hierarchical multi-view encoder with LayerNorm, stochastic depth, gated residuals and identity-at-init, but its cross-view attention is still *content-only*: views attend to each other based on learned feature similarity.  This is sub-optimal for the multi-view pose problem because:

1. **Feature similarity can be misleading.**  Occluded joints, noisy detections, or calibration drift can make two views look similar in feature space even when their rays are geometrically inconsistent.
2. **v25 already showed the value of geometry bias.**  The `GeometryAwareCrossViewAttention` block in `multiview_geometry_fusion_v25.py` adds epipolar-distance and ray-intersection terms to the attention scores, but it operates on a single scale and is tied to the geometry-fusion branch, not the hierarchical encoder.
3. **v29a overfits after epoch 1.**  A strong geometry prior can act as a regulariser and slow the drift into over-fitted feature patterns.

We therefore propose v31: bring v25-style geometry-aware attention into the v30 hierarchical encoder, at every scale (joint / part / body), while preserving v30's hardening and identity-at-init properties.

## Concrete Proposed Change

Add a new module, `motionflow_mv/fusion/hierarchical_multiview_v31.py`, containing:

- **`_GeometryBiasedCrossViewAttentionBlock`** – a `MultiheadAttention`-based block that accepts an additive geometry-bias tensor `(N, V, V)` and adds it to the attention scores before softmax.  It keeps v30's pre-norm, FFN, dropout, and stochastic-depth behaviour.
- **`HierarchicalViewEncoderV31`** – mirrors the v30 three-scale encoder (joint, part, body) but computes geometry biases once from the input cameras and 2-D points and feeds them to each scale:
  - **Joint scale** uses per-joint epipolar distance + ray-intersection logit.
  - **Part scale** averages the per-joint bias over each part group.
  - **Body scale** averages the bias over all joints.
  - A learnable per-block geometry gate is initialised near zero, so the block is effectively content-only at the start of training and gradually learns to use geometry.
- **Fallback path**: if cameras / 2-D points are omitted, the block behaves exactly like v30, which preserves warm-start compatibility and keeps smoke tests simple.
- **Integration plan** (one-line change in `omniview_fusion_v5.py` in a follow-up): replace the v30 call with the v31 call and pass `points_2d`, `K`, `R`, `t`.

The new module reuses existing helpers (`compute_rays`, `ray_intersection_logit`, `compute_epipolar_distance`) so no new geometric primitives are needed.

## Expected Impact

- **val_MPJPE**: modest but consistent improvement over v30 on WebBridge/H36M/MPI mixed validation, especially in the presence of outlier views and occlusion.  The geometry bias should down-weight inconsistent views before they can corrupt the fused token.
- **Overfitting**: geometry provides a strong inductive bias that is independent of the training set, so we expect slower overfitting after the first epoch (mitigating the v29a collapse from 28 mm to 47 mm in one epoch).
- **Few-view robustness**: the part/body scales already help when few views are available; adding geometric consistency should make the 2–4 view regime more stable.
- **Computational cost**: one extra ray/epipolar pass per forward and a small number of extra parameters (geometry gate + two temperature scalars).  The overhead should be <5 % of a v30 forward.

## Main Risk

**Geometry bias can dominate too early.**  If the gate or the temperature parameters are not initialised conservatively, the attention may lock onto a subset of geometrically "safe" views and ignore useful appearance cues.  This is mitigated by:
- Initialising the geometry gate to ~0.05 (sigmoid(-3.0)).
- Keeping the residual gate at sigmoid(-6.0) so the whole block is identity-at-init.
- Pairing with the v29 physical-loss warmup so new priors are ramped in gradually.
- TTE remains disabled, per current project policy.

## Smoke / Launch

A local RTX 4090 smoke is provided in `scripts/run_v31_geometry_attention_refinement_local4090.sh`.  It instantiates `HierarchicalViewEncoderV31` on synthetic tokens and cameras, checks identity-at-init, gradient flow, and view-mask handling.  Full training requires the one-line integration into `omniview_fusion_v5.py` and the new flag `--use_hierarchical_multiview_v31`, which is intentionally left for the next iteration to keep this proposal read-only with respect to existing source files.
