# Agent Risk Report: v54 Outlier-Robust Reliability Refinement (OR2)

**Module:** `outlier_robust_reliability_v54`  
**Tracking issue:** #184  
**Date:** 2026-08-09

---

## Risk 1: Over-aggressive outlier downweighting collapses to degenerate triangulation

**Description:**  
If the learned M-estimator is too steep or the scale shrinks too fast, v54 may downweight all but one or two views for a joint. With `min_visible_views=2` the triangulation can still proceed, but the solution may become numerically unstable or overfit to a single noisy view.

**Mitigation:**
* Enforce a hard floor on the refined weight (`v54_or2_min_weight=0.05`) so every visible view retains at least a small influence.
* Initialize `scale` to a conservative pixel value (default 5 px) so early training does not reject clean views.
* Add a supervision signal that penalizes entropy collapse: the auxiliary `L_or2` should keep `γ` near 1 for inliers.
* During smoke, log the fraction of weights below `0.1` and reject configs where >30 % of weights are zeroed.

---

## Risk 2: Identity-at-init is broken by the re-triangulation step

**Description:**  
v54 is supposed to be warm-startable/identity-at-init so that loading a v53 checkpoint with v54 enabled leaves `val_MPJPE` unchanged. However, because we re-triangulate with `w^{OR2} = w^{UWT} · γ`, even if `γ` starts at 1.0, floating-point differences in the DLT path or residual MLP may shift the output pose.

**Mitigation:**
* Set `v54_or2_identity_init=True` by default, which zero-initializes the residual MLP final layer and initializes the residual gate so `sigmoid(gate) ≈ 0`.
* Initialize the `logit_γ` bias to a large positive value (e.g. +4.0) so `γ ≈ 1` at start.
* Implement a unit test that compares `pred_3d_or2` against the v52/v53 input when the module is freshly initialized; assert per-joint error < 1e-4 mm.
* Smoke-test by loading the best v53 checkpoint with v54 enabled; require `|ΔMPJPE| ≤ 0.1 mm`.

---

## Risk 3: Scale collapse / scale drift across domains

**Description:**  
The learned robust scale is data-dependent. When training on a mixture of H36M, MPI, and 3DPW, the scale may drift to a value that is appropriate for one domain but disastrous for another (e.g. very small for clean studio data, huge for noisy 3DPW).

**Mitigation:**
* Clamp the predicted scale to a sensible range, e.g. `[1.0 px, 30.0 px]`, using `scale.clamp(min=1.0, max=30.0)`.
* Make the scale prediction depend on domain label (if `v54_or2_use_domain_scale=True`) or, better, normalize residuals by the per-batch median absolute deviation (MAD) before feeding the MLP.
* Log per-domain histograms of `scale` and `γ` in tensorboard; alert if any domain median scale differs by >5 px from the others after the first epoch.

---

## Risk 4: Interaction with v46 sparse-view dropout confuses reliability learning

**Description:**  
v46 randomly drops views during training. If v54 learns that certain view indices are unreliable simply because they are sometimes missing (masked to zero), it may develop a view-index bias rather than a content-based outlier score.

**Mitigation:**
* Use the same `view_mask` that v46 produces; do not let missing views participate in the residual statistics.
* Normalize features so that masked views contribute neither to residual aggregation nor to per-view statistics.
* Include camera-conditioned features (ray direction, camera position) instead of raw view index so the outlier score is geometrically grounded.
* Verify in smoke that `γ` for a synthetically occluded view is lower than `γ` for the same view when unoccluded, independent of view index.

---

## Risk 5: Auxiliary loss `L_or2` dominates early training

**Description:**  
The binary-like inlier/outlier auxiliary loss may be noisy in the first epochs while the 3D pose estimate is still poor. If the loss weight is too high, the module may overfit to incorrect residual thresholds and suppress useful views.

**Mitigation:**
* Default `v54_or2_loss_weight=0.01` and apply warmup (`v54_or2_warmup_epochs=1`) so the auxiliary loss is added only after the main pose estimate has stabilized.
* Make the inlier threshold dynamic: define inliers as views whose residual is below the 25th percentile of the *current* residual distribution, rather than a fixed pixel value.
* In the smoke script, test both `v54_or2_loss_weight=0` and `0.01`; only enable the loss if it does not increase the first-epoch `val_MPJPE` by >2 mm.
