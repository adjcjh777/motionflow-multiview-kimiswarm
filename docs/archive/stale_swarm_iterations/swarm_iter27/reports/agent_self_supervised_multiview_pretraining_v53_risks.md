# v53 Self-Supervised Multi-View Pre-Training — Risk Report

**Module:** `SelfSupervisedMultiViewPretrainingV53`  
**Tracking issue:** #201  
**Depends on:** v52 Uncertainty-Weighted Triangulation

## Risk matrix

| # | Risk | Symptom / Failure mode | Mitigation |
|---|------|----------------------|------------|
| 1 | **Degenerate masked triangulation** | `pred_3d_mask` contains NaN/Inf or very large values when too many views are masked, especially in sparse-view samples. | Enforce `v53_ssmvp_min_visible_views >= 2` and `v53_ssmvp_min_visible_ratio <= 0.5`. Skip joints/time steps with fewer than two visible views. Use the same SVD pseudo-inverse + damping already present in the v52 DLT path. |
| 2 | **Noisy pseudo-target drift** | The full-view `pred_3d_full` is itself imperfect; optimising toward it can amplify systematic triangulation errors (e.g., near coplanar views). | Detach `pred_3d_full` (`stop_grad`). Keep the re-projection term `L_reproj` so the masked prediction is still anchored to real 2-D evidence. Optionally decay `λ_cons` relative to `λ_reproj` during the first epochs. |
| 3 | **Double-counting / conflict with v52 UWT** | The v52 UWT precision network may overfit to the synthetic mask distribution, hurting full-view performance or sparse-view v46/v51 heads. | Treat v53 as a pure auxiliary loss; do not change the main `pred_3d_full` output. Warm-start from a trained v52 checkpoint and freeze all non-UWT parameters for the first epoch. Monitor `MPJPE@full` and `MPJPE@2/3` jointly. |
| 4 | **Training cost / memory blow-up** | Running an second masked DLT pass on every sample increases forward time and GPU memory. | Only compute the masked triangulation on a random half of the batch (`v53_ssmvp_apply_ratio=0.5`). Reuse the existing `triangulate_dlt_batched_lstsq` implementation; the extra pass is a single batched pseudo-inverse, not a full forward. |
| 5 | **Mask distribution mismatch** | The synthetic view/joint/time corruptions may not resemble real test-time dropouts, so the learned weights fail to transfer to actual missing views. | Mix all three corruption modes and randomise `v53_ssmvp_mask_ratio` uniformly in `[0.15, 0.35]` per batch. Evaluate on the real v46 sparse-view protocol (`MPJPE@2/3/4`) and fall back to lower loss weight if only full-view metrics improve. |

## Acceptance thresholds

* Identity-at-init: loading the best v52 checkpoint with v53 enabled changes `val_MPJPE@full` by less than 0.1 mm before any training step.
* Smoke: no NaN/Inf/OOM for at least one full epoch; auxiliary loss magnitude < 5 % of total loss.
* Ablation: v52+v53 must not regress `MPJPE@full`; `MPJPE@2/3` should improve by ≥ 0.5 mm or stay within 0.3 mm to proceed to A800.
