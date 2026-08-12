# Agent v52 Risk Analysis: Uncertainty-Weighted Triangulation

**Module:** `uncertainty_weighted_triangulation_v52`  
**Date:** 2026-08-09  
**Analyst:** design-swarm agent  
**Scope:** `motionflow_mv/fusion/uncertainty_weighted_triangulation_v52.py`, `OmniMultiViewFusionV5`, `motionflow_mv/utils/geometry.py`

## Executive summary

v52 introduces a learned per-view, per-joint uncertainty that drives a weighted triangulation. The idea is conceptually close to v33 `UncertaintyAwareTriangulationV33`, v45 adaptive geometry fusion, and v51 cross-domain sparse-view reliability, so the main risks are not architectural novelty but *stability*, *composition*, and *overhead*. Below are five concrete risks with mitigations that can be actioned during implementation.

## Risk 1: Weight collapse to one dominant view

**Description:** The precision MLP can learn to put all mass on a single view (e.g. the first visible view) and ignore the others. This is a local minimum when one view dominates the training data, and it defeats the purpose of multi-view fusion. With `min_weight=0.05` the collapse is bounded but still harmful.

**Impact:** Triangulation degrades to single-view depth from one camera; MPJPE rises, especially for joints that are occluded or noisy in the chosen view.

**Mitigation:**

1. Enforce the entropy regularization term `H_bar(weights)` described in the proposal (§3.4).
2. Initialize the precision predictor to near-zero (uniform weights) and use a small learning-rate multiplier on its parameters for the first epoch.
3. Clamp `precision_vj` to `[min_weight, 1/min_weight]` so that no view can dominate by more than a fixed factor.
4. Monitor `max_v weights / mean_v weights` as a smoke-test sanity metric; reject runs where this ratio exceeds 10 in the first 500 steps.

## Risk 2: Unstable gradients in weighted DLT

**Description:** The weighted DLT uses a pseudo-inverse of `A^T W A`. When weights are very small or when only two views are visible, the matrix can become ill-conditioned, producing `NaN/Inf` gradients or exploding updates. This is particularly likely during variable-view training with `view_mask`.

**Impact:** Training diverges or produces NaN losses; smoke tests fail with LAPACK errors.

**Mitigation:**

1. Add a learned or fixed damping term `A^T W A + lambda_damp I` before inversion, where `lambda_damp` is small (e.g. `1e-4`).
2. Use `torch.linalg.lstsq` or SVD-based pseudo-inverse rather than a direct matrix inverse.
3. Mask the loss and gradient for joints with fewer than `min_visible_views=2`; set those joints to the initial triangulation and zero their contribution to `L_uwt`.
4. Add a unit test that triangulates random 2-view, 3-view, and 4-view batches and checks finite gradients.

## Risk 3: Double-counting with v45 / v46 / v51 weights

**Description:** v52 weights may duplicate the role of v45 adaptive geometry fusion, v46 sparse-view reliability, and v51 cross-domain sparse-view reliability. If all three mechanisms down-weight the same noisy view, the combined effective weight can become too small, and the model may ignore useful but slightly noisy views.

**Impact:** Sparse-view (`MPJPE@2`) and cross-domain metrics degrade; the ensemble becomes over-conservative.

**Mitigation:**

1. Treat v52 as the *primary* triangulation weight, and use v45/v46/v51 weights only inside the downstream refinement blocks, not by multiplying them into the same DLT.
2. If multiplication is desired, add a gating MLP that learns how to combine the factors rather than multiplying naively, or freeze v45/v46/v51 weights during the first epoch of v52 training.
3. Run an ablation that trains v52 on top of v45 only, then v45+v46, then v45+v46+v51, to isolate interaction effects.

## Risk 4: Memory and latency overhead

**Description:** The precision MLP runs per view, per joint, and per frame, increasing memory and compute. For `B=4`, `T=243`, `V=4`, `J=17`, `d=64`, `hidden=64`, the MLP adds ~2M parameters and a per-batch forward cost of ~0.5M activations. This can matter on the RTX 4090 smoke budget.

**Impact:** Smoke runs OOM or exceed the 24 GB GPU budget; training throughput drops.

**Mitigation:**

1. Share the MLP weights across all joints and use a small per-joint embedding if `weight_type="per_view_joint"` is needed.
2. Default `v52_uwt_hidden=32` for smoke configs and `64` only for full A800 runs.
3. Optionally skip v52 during the first few epochs (gate with `epoch >= v52_uwt_warmup_epochs`) to preserve the baseline training speed.

## Risk 5: Identity-at-init is not strictly identity

**Description:** Even with zero-initialized last layers, the residual MLP may contain BatchNorm or LayerNorm layers that shift the output, and the precision MLP may produce tiny non-zero values due to numerical precision or the final `exp`. The module may therefore perturb the baseline before any training.

**Impact:** Smoke comparisons to v45/v46 become noisy; warm-start from a v45 checkpoint may diverge in the first steps.

**Mitigation:**

1. Do not place normalization layers after the final zero-initialized linear projection; use layer norm only inside the MLP trunk.
2. Add an unit test that instantiates `UncertaintyWeightedTriangulationV52`, passes random inputs, and asserts `||pred_3d_ref - pred_3d_init||_2 < 1e-4` before any optimizer step.
3. Expose a `disable_residual` flag for the smoke test so that the residual path can be removed from the comparison.

## Recommendation

Proceed with implementation, but prioritize Risks 2 and 3 in the smoke phase. A stable, identity-initialized weighted triangulation is a prerequisite for the later v45/v46/v51 composability experiments.
