# v54 Learned View Selection Policy — Risk Register

## Risk 1: Straight-Through Top-K Causes Gradient Instability or NaN/Inf

- **Likelihood:** Medium  
- **Impact:** Training diverges; smoke shows `val_MPJPE > 120 mm` or NaN loss.  
- **Mitigation:**  
  - Default to soft sigmoid selection for the first epoch (`v54_lvsp_hard=False`).  
  - Clip policy logits to `[-5, 5]` before applying temperature.  
  - Detach the residual reference used for the reprojection reward.  
  - Fall back to continuous relaxation if any NaN appears in the selection weights.

## Risk 2: Policy Drops Too Many Views and Under-Constrains Triangulation

- **Likelihood:** Medium-High  
- **Impact:** Sparse-view metrics (`MPJPE@2`, `MPJPE@3`) degrade; overall pose becomes unstable.  
- **Mitigation:**  
  - Enforce `v54_lvsp_min_weight = 0.05` so every visible view retains some influence.  
  - When `v54_lvsp_top_k > 0`, clamp the per-joint selected count to `max(2, min_visible_views)`.  
  - Add an entropy bonus that penalizes degenerate selections that select zero or one view only.  
  - Validate smoke on `MPJPE@2/3/4` in addition to `MPJPE@full`.

## Risk 3: Early-Training Noisy Residuals Mislead the Policy

- **Likelihood:** High  
- **Impact:** The policy latches onto coincidental low-residual views and never recovers; full-run MPJPE is worse than v53.  
- **Mitigation:**  
  - Use a detached, baseline `pred_3d` from v53 as the residual reference.  
  - Ramp `v54_lvsp_loss_weight` from `0.0` to its target over the first epoch (warm-up schedule).  
  - Use a small entropy weight initially and increase it only after the policy has seen stable gradients.  
  - Evaluate identity-at-init with a pretrained v53 checkpoint: `MPJPE` should not shift by more than 0.1 mm.

## Risk 4: Inference-Time Top-K Sensitivity Across Datasets

- **Likelihood:** Medium  
- **Impact:** A single `v54_lvsp_top_k` works well for H36M/WebBridge but fails on 3DPW or other unseen camera layouts.  
- **Mitigation:**  
  - Default `v54_lvsp_top_k = 0` (continuous) for the first smoke; enable top-K only after dataset-specific tuning.  
  - Expose a dataset override in the YAML config (e.g., `v54_lvsp_top_k_3dpw`).  
  - Report per-domain selection probability histograms in evaluation to detect domain-specific collapse.

## Risk 5: Interaction with v51/v50/CDSVR Auxiliary Losses

- **Likelihood:** Medium  
- **Impact:** Multiple auxiliary losses (v50 SEFH, v51 CDSVR, v52 UWT, v53 PSC, v54 LVSP) compete for gradients and over-regularize the pose head.  
- **Mitigation:**  
  - Keep `v54_lvsp_loss_weight = 0.01` or lower; increase only on smoke evidence.  
  - Run an ablation stack: `v53 only` → `v53 + v54 (soft)` → `v53 + v54 (top-K)` → `v51/v52/v53/v54`.  
  - Stop the A800 full run if the smoke comparison against the v53 baseline is not positive within 2 epochs.
