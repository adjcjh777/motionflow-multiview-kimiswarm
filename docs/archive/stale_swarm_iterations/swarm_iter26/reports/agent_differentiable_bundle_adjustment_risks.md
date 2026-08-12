# v52 Differentiable Bundle Adjustment — Risk Register

**Module:** `differentiable_bundle_adjustment_v52`  
**Date:** 2026-08-09  
**Status:** design-stage risk register  

## R1: Unstable camera corrections in early training

**Risk:** the camera residual MLPs may predict large corrections before the pose estimate is reliable, causing the camera parameters to diverge and training loss to explode.  
**Impact:** high in first 1–2 epochs.  
**Mitigation:**

* Start with `v52_dba_init_gate=0.0` and warm it up linearly over the first few epochs (`v52_dba_gate_warmup_epochs`).
* Clamp camera translation updates to `[-v52_dba_max_t_cm, +v52_dba_max_t_cm]` and rotation updates to a small angle (e.g., 2°).
* Use the Huber robust kernel and high `v52_dba_cam_reg` in early experiments.

## R2: Slow forward pass due to unrolled optimisation

**Risk:** each forward pass runs `K` gradient steps over `V` views and `J` joints, increasing memory and runtime.  
**Impact:** medium; may reduce effective batch size on A800/RTX 4090.  
**Mitigation:**

* Keep `v52_dba_num_steps` small (default 3; ablate 1, 3, 5).
* Run the DBA block on a single representative frame per clip or use a strided temporal window rather than every frame.
* Cache the final refined cameras for the rest of the forward pass so the block is evaluated once per clip.

## R3: Redundancy with v50/v51 reliability/uncertainty heads

**Risk:** v50 SEFH and v51 CDSVR already down-weight unreliable views; jointly optimising cameras may duplicate or conflict with their learned behaviour.  
**Impact:** medium; could lead to over-regularisation and no MPJPE gain.  
**Mitigation:**

* Use v50/v51 weights only as soft priors, not hard masks.
* Add an smoke/ablation that runs v52 **without** v50/v51 to isolate the gain.
* Make `v52_dba_use_sefh_weights` a flag so the dependency can be disabled quickly.

## R4: Degenerate geometry when views are collinear or very few

**Risk:** refining cameras from `min_views=2` configurations can produce unstable / degenerate bundle adjustments, especially when the 2-D detections are noisy.  
**Impact:** high for sparse-view metrics (`MPJPE@2`).  
**Mitigation:**

* Skip DBA when the number of active views is below `v52_dba_min_views` (default 3).
* Add a damping term to the normal equations equivalent inside the unrolled solver.
* Fall back to the input pose/cameras when the reprojection energy increases after refinement.

## R5: Gradient bias through iterative camera updates

**Risk:** unrolled gradient steps through camera parameter updates can introduce bias, explode gradients, or create memory issues when storing intermediate Jacobians.  
**Impact:** medium; can hurt convergence.  **Mitigation:**

* Use `torch.autograd.functional.jacobian` only for the final energy, not every inner step.
* Apply gradient clipping around the DBA block.
* Consider a learned update (one MLP step) as a fallback if explicit Jacobians prove unstable.

## Action items before implementation

1. Confirm the exact tensor shape of `v50_sefh_reliability` in `OmniMultiViewFusionV5` so the DBA weighting is broadcast correctly.
2. Decide whether v52 should refine **extrinsics only** (recommended first cut) or intrinsics too.
3. Draft the smoke config `configs/benchmark_v52_dba_smoke.yaml` and add a CPU-only unit test for the identity-at-init property.
