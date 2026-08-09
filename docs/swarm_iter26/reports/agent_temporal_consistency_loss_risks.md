# v52 Temporal Consistency Loss — Risk Register

## 1. Over-smoothing fast motion

**Risk:** The multi-scale smoothness loss may penalise legitimate high-velocity motion (e.g., jumping, fast arm swings), collapsing dynamic poses toward a static mean.

**Mitigation:**
- Use a Huber loss with scale-dependent delta `δ_τ = v52_tcl_huber_delta · τ` so large residuals at coarse scales are not over-penalised.
- Initialise the learned scale weights to zero and ramp the loss weight over one epoch, giving the pose head time to stabilise before strong smoothing.
- Ablate `v52_tcl_scales`: keep only `τ=1` if larger scales harm fast-motion sequences.

## 2. Conflict with v28/v40 physical losses and v47/v49 temporal heads

**Risk:** v52 adds an temporal loss on top of v47/v49 temporal aggregation and v28/v40 physical-space losses. The combined regularisation may push the model toward a physically plausible but geometrically incorrect local minimum.

**Mitigation:**
- Start with `v52_tcl_weight = 0.001` and increase only after the physical losses have stabilised.
- Treat v52 as mutually exclusive with v32/v33 trajectory refiners; the v47/v49 heads can remain enabled because they refine features, not the final pose loss.
- Monitor the per-term loss magnitudes in TensorBoard/WandB and keep `L_v52 < 0.1 · L_reproj` by tuning `λ`.

## 3. Sparse-view degradation

**Risk:** With only 2–3 active views, triangulation is already noisy. A strong temporal consistency loss may amplify bias from the dominant view and hide the true motion.

**Mitigation:**
- Only apply v52 when `v52_tcl_min_views_for_loss` is satisfied; skip the loss entirely for clips below the threshold.
- Weight each term by per-joint confidence `c(t, j)` so occluded joints do not contribute.
- Evaluate `MPJPE@2`, `MPJPE@3`, and `MPJPE@full` separately; if sparse-view metrics regress, reduce `v52_tcl_weight` or drop coarse scales.

## 4. Learned scale weights may collapse to zero or explode

**Risk:** The per-joint MLP that predicts `w_τ` is small and may either learn to zero out the loss or saturate, effectively making the loss scale non-stationary across epochs.

**Mitigation:**
- Clamp `w_τ ∈ [0.1, 2.0]` so the loss is always active but bounded.
- Add a tiny L2 regulariser on the MLP weights (`1e-5`).
- Provide a deterministic fallback: when `v52_tcl_learned_scale_weights = False`, set `w_τ = 1` for all joints and scales.

## 5. Warm-up interaction with early stopping and EMA

**Risk:** Because the loss is ramped from zero, early stopping may trigger before the loss reaches full strength, and EMA may track a model that has not yet felt the temporal regularisation.

**Mitigation:**
- Set `early_stopping_patience` to at least the number of warmup epochs plus two.
- Use the post-warmup validation metric for the early-stopping decision, not the first epoch.
- Apply EMA only after the warmup epoch has completed.
