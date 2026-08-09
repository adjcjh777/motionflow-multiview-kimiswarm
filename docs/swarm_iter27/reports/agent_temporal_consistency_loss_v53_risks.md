# v53 Temporal Consistency Loss — Risk Register

## 1. Over-smoothing of fast motion

**Risk:** The multi-scale smoothness and velocity losses may penalise legitimate high-velocity motion (e.g., rapid limb movements, sports actions), collapsing dynamic poses toward a temporally smoothed mean and increasing MPJPE on fast-motion sequences.

**Mitigation:**
- Use a scale-dependent Huber delta `δ_τ = v53_tcl_huber_delta · τ` so large residuals at coarse scales are not over-penalised.
- Ramp the loss weight linearly from zero over `v53_tcl_warmup_epochs`, giving the pose head time to stabilise before strong temporal regularisation takes effect.
- Clamp the per-joint confidence `conf(t, j) ≥ v53_tcl_min_conf` so the loss is active only up to a bounded level.
- Ablate `v53_tcl_scales`; if coarse scales harm fast motion, keep only `τ = 1`.

## 2. Coupling with v52 UWT weights

**Risk:** v53 reuses the v52 UWT weights as a proxy for temporal uncertainty. If v52 weights collapse to a degenerate distribution or are not enabled, the confidence signal becomes uninformative and the temporal loss is applied uniformly, potentially amplifying v52 failures.

**Mitigation:**
- Guard `use_v53_temporal_consistency_loss` so it requires `use_v52_uncertainty_weighted_triangulation = True`; fail fast with a clear error if the dependency is missing.
- Fallback to a uniform confidence when all v52 weights are below `v53_tcl_min_conf`, avoiding division-by-zero and NaN gradients.
- Monitor the correlation between v52 weight entropy and temporal loss magnitude; if the correlation breaks, disable learned scale weights.

## 3. Conflict with v28/v40 physical losses and v47/v49 temporal heads

**Risk:** v53 adds temporal regularisation on top of v47/v49 temporal aggregation and v28/v40 physical-space losses. The combined regularisation may push the model toward a physically plausible but geometrically incorrect local minimum, or it may dominate the supervised pose loss.

**Mitigation:**
- Start with `v53_tcl_weight = 0.001` and increase only after the physical and temporal refinement losses have stabilised.
- Ensure the total auxiliary loss from v53 stays below `0.1 · L_reproj` by tuning `λ`; add a hard clipping bound of `10.0` on the v53 loss during warmup.
- Log `loss_v53_smooth`, `loss_v53_velocity`, and `loss_v53_bone` separately so their relative magnitudes are visible.

## 4. Sparse-view degradation

**Risk:** With only 2–3 active views, triangulation is already noisy and v52 weights may be unreliable. A strong temporal consistency loss may amplify bias from the dominant view and hide true motion, hurting `MPJPE@2` and `MPJPE@3`.

**Mitigation:**
- Only apply v53 when at least `v53_tcl_min_views_for_loss` active views are present; skip the loss entirely for sparser clips.
- Weight each term by per-joint confidence `conf(t, j)` so joints with low v52 precision do not contribute strongly.
- Evaluate `MPJPE@2`, `MPJPE@3`, and `MPJPE@full` separately; if sparse-view metrics regress, reduce `v53_tcl_weight` or disable coarse scales.

## 5. Warm-up interaction with early stopping and EMA

**Risk:** Because the loss is ramped from zero, early stopping may trigger before the loss reaches full strength, and EMA may track a model that has not yet felt the temporal regularisation. This can lead to selecting a sub-optimal checkpoint.

**Mitigation:**
- Set `early_stopping_patience` to at least `v53_tcl_warmup_epochs + 2`.
- Use the post-warmup validation metric for the early-stopping decision, not the first epoch.
- Apply EMA updates only after the warmup epoch has completed, or increase EMA momentum after warmup.
