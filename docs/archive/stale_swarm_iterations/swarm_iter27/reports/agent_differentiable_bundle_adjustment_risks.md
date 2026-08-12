# v53 Differentiable Bundle Adjustment — Risk Register

**Module:** `differentiable_bundle_adjustment_v53`
**Date:** 2026-08-09
**Status:** design-stage risk register

## R1: Camera corrections diverge in early training

**Risk:** The learned camera residual MLPs may predict large corrections before the pose estimate is reliable, causing camera extrinsics to drift and training loss to explode.
**Impact:** High in the first 1–2 epochs.
**Mitigation:**

* Keep `v53_dba_init_gate=0.0` and warm it up linearly over `v53_dba_warmup_epochs`.
* Clamp translation updates to `[-v53_dba_max_t_cm, +v53_dba_max_t_cm]` and rotation updates to a small angle (e.g., ≤ 2°).
* Initialise all correction MLP final layers to zero and use a high `v53_dba_cam_reg` in early experiments.

## R2: Unrolled optimisation is slow and memory-heavy

**Risk:** Each forward pass runs `K` differentiable LM steps over `V` views and `J` joints, storing intermediate Jacobians and increasing memory use.
**Impact:** Medium; may force smaller batch sizes on RTX 4090 / A800.
**Mitigation:**

* Default `v53_dba_num_steps=3`; ablate `1, 3, 5`.
* Stop gradients through the intermediate LM steps and back-propagate only through the final energy (detach intermediates), or use a learned single-step update as a fallback.
* Cache refined cameras for the remainder of the forward pass to avoid recomputation.

## R3: Degenerate geometry with too few views

**Risk:** Refining cameras from `min_views=2` configurations can be unstable, especially when 2-D detections are noisy and the baseline is short.
**Impact:** High for `MPJPE@2` and variable-view metrics.
**Mitigation:**

* Skip DBA when the number of active views is below `v53_dba_min_views` (default 3).
* Add a damping term (`v53_dba_damping`) to the normal equations inside the LM solver.
* Fall back to the input pose and cameras whenever the reprojection energy increases after refinement.

## R4: Double-counting with v52 UWT and v50/v51 reliability heads

**Risk:** v52 already learns per-view/joint weights, while v50 SEFH and v51 CDSVR learn view reliability. Running DBA on top may over-regularise or conflict with these signals.
**Impact:** Medium; may yield no MPJPE gain or unstable gradients.
**Mitigation:**

* Use v52 weights as soft observation covariances, not as hard masks, and make this optional via `v53_dba_use_uwt_weights`.
* Run an ablation with `v53_dba_use_uwt_weights=False` to isolate the camera-correction gain from the weighting gain.
* Gate the DBA loss so it does not dominate the total loss early in training.

## R5: Gradient bias through iterative camera updates

**Risk:** Unrolled differentiable camera updates can introduce biased gradients, exploding values, or high memory use from stored Jacobians.
**Impact:** Medium; can hurt convergence and slow training.
**Mitigation:**

* Compute the analytic Jacobian of the reprojection residual only for the final step, or use finite-difference/checkpointing.
* Apply gradient clipping around the DBA block in the trainer.
* Provide a fallback `v53_dba_learned_update=True` mode where a small MLP predicts the pose/camera update directly, avoiding explicit Jacobians.

## Action items before implementation

1. Confirm the exact tensor shape and value range of `uwt_weights` output by `UncertaintyWeightedTriangulationV52` so the DBA weighting broadcasts correctly.
2. Decide whether v53 should refine **extrinsics only** (recommended first cut) or also intrinsics.
3. Draft the smoke config `configs/benchmark_v53_dba_smoke.yaml` and add a CPU-only unit test for the identity-at-init property.
