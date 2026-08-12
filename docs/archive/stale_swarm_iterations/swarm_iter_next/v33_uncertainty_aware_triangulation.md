# v33 Proposal: Uncertainty-Aware Depth / Triangulation

**Slug:** `uncertainty_aware_triangulation`  
**Target iteration:** v33 (post-v32)  
**Status:** Design proposal — ready for swarm review

---

## 1. Problem Statement and Motivation

Current OmniMultiViewFusionV5 triangulates 3-D poses via confidence-weighted DLT, optionally refined by the v25 geometry-fusion block and v27 uncertainty depth proposals (UDP).  However, the *confidence* channel coming from the 2-D detector is only a scalar per keypoint and is often miscalibrated or corrupted by occlusion.  The model therefore has no learned, joint-conditioned estimate of **which views are trustworthy for which joint** beyond the static detector confidence and the heuristic outlier-view detector.

Prior art exists but is not wired into the v5/v25 mainline:

* `motionflow_mv/models/crossview_residual_uncertainty.py` and `motionflow_mv/fusion/ray_attention_temporal_uncertainty_v2_model.py` showed that a learned per-view log-variance improves triangulation robustness.
* `motionflow_mv/fusion/uncertainty_weighted_triangulation.py` implements differentiable covariance-weighted DLT but is not integrated.
* v27 `UncertaintyDepthProposalTriangulation` models depth uncertainty per ray, but does not estimate per-view 2-D observation uncertainty for the DLT itself.

**Goal for v33:** add a learned, end-to-end uncertainty-aware triangulation head inside `OmniMultiViewFusionV5` that predicts per-view, per-joint 2-D uncertainty from the spatio-temporal features, re-weights the DLT triangulation with that uncertainty, and is supervised by a reprojection negative-log-likelihood (NLL) loss.  The module should be optional, identity-at-init, and compatible with variable-view training and the existing v25/v27/v32 pipeline.

---

## 2. Proposed Architecture Changes

### 2.1 New module

**File:** `motionflow_mv/fusion/uncertainty_aware_triangulation_v33.py`

**Class:** `UncertaintyAwareTriangulationV33`

Inputs (same shapes as v25 geometry-fusion block):

* `points_2d`: `(B, T, V, J, 2)`
* `confidences`: `(B, T, V, J)`
* `features`: `(B, T, V, J, d)` — per-view spatio-temporal tokens from the ST transformer
* `proj_matrices`: `(B, T, V, 3, 4)`
* `pred_3d_init`: `(B, T, J, 3)` — initial triangulated estimate
* `view_mask`: optional `(B, T, V)`

Outputs:

* `pred_3d_ref`: `(B, T, J, 3)` — uncertainty-refined 3-D estimate
* `uncertainty_loss`: scalar auxiliary NLL loss

**Internal design:**

1. **Uncertainty head.**  A small 2-layer MLP on the per-view feature token `features` predicts a per-view, per-joint log-variance:
   ```
   log_var_u, log_var_v = MLP(features)  # (B, T, V, J, 2)
   ```
   The diagonal covariance is `Σ = diag(exp(log_var_u), exp(log_var_v))`.  `log_var` is clamped to `[-10, 10]` for stability.

2. **DLT re-weighting.**  The 2-D uncertainty is converted to a precision-weight DLT system using the existing `uncertainty_weighted_triangulation.triangulate_uncertainty_weighted_batched` helper.  Each view's rows in the DLT matrix are multiplied by the inverse Cholesky factor of `Σ`, so noisier views automatically contribute less.

3. **Residual refinement (identity at init).**  The precision-weighted DLT output is refined by a tiny MLP residual around `pred_3d_init`:
   ```
   pred_3d_ref = pred_3d_init + scale * MLP(pred_3d_weighted - pred_3d_init)
   scale = nn.Parameter(torch.tensor(0.0))  # no-op at init
   ```

4. **Supervision.**  A reprojection NLL loss encourages predicted uncertainty to match actual reprojection errors:
   ```
   L_uat = 0.5 * (r^T Σ^{-1} r + log det Σ)
   ```
   where `r` is the 2-D reprojection residual.  This is computed only on views with positive confidence.

### 2.2 Integration into `OmniMultiViewFusionV5`

In `motionflow_mv/fusion/omniview_fusion_v5.py`, add a new optional stage **after** the v25 geometry-fusion block and **before** the residual/diffusion refiner:

```python
if self.use_uncertainty_aware_triangulation_v33 and self.uncertainty_aware_triangulation_v33 is not None:
    pred_3d_gn, uat_loss = self.uncertainty_aware_triangulation_v33(
        points_2d=points_2d.view(B, T, V, J, 2),
        confidences=confidences.view(B, T, V, J),
        features=feat.view(B, T, V, J, d),
        proj_matrices=P.view(B, T, V, 3, 4),
        pred_3d_init=pred_3d_gn.view(B, T, J, 3),
        view_mask=view_mask_flat.view(B, T, V),
    )
    pred_3d_gn = pred_3d_gn.view(B * T, J, 3)
    geom_loss_v25 = geom_loss_v25 + self.v33_uat_loss_weight * uat_loss
```

The uncertainty loss is folded into `geom_loss_v25`, so no training-script changes are required beyond the flag plumbing.

### 2.3 Training-script flags

In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, add:

```python
parser.add_argument("--use_uncertainty_aware_triangulation_v33", action="store_true",
                    help="Use v33 uncertainty-aware triangulation head")
parser.add_argument("--v33_uat_loss_weight", type=float, default=0.01,
                    help="Weight for v33 uncertainty NLL loss")
parser.add_argument("--v33_uat_log_var_min", type=float, default=-10.0,
                    help="Min log-variance clamp")
parser.add_argument("--v33_uat_log_var_max", type=float, default=10.0,
                    help="Max log-variance clamp")
parser.add_argument("--v33_uat_covariance_hidden", type=int, default=64,
                    help="Hidden dim of the uncertainty-prediction MLP")
```

Pass-through in `build_model_from_args`:

```python
"use_uncertainty_aware_triangulation_v33": getattr(args, "use_uncertainty_aware_triangulation_v33", False),
"v33_uat_loss_weight": getattr(args, "v33_uat_loss_weight", 0.01),
"v33_uat_log_var_min": getattr(args, "v33_uat_log_var_min", -10.0),
"v33_uat_log_var_max": getattr(args, "v33_uat_log_var_max", 10.0),
"v33_uat_covariance_hidden": getattr(args, "v33_uat_covariance_hidden", 64),
```

In `OmniMultiViewFusionV5.__init__`, instantiate the new head when the flag is on and wire the new hyperparameters.

### 2.4 Data / preprocessing needs

No new dataset is required.  The module consumes the same `(points_2d, confidences, K, R, t, view_mask)` tensors already produced by the mixed loader.  It is designed to be robust to:

* variable number of active views (`view_mask`)
* outlier views injected by `inject_outlier_views`
* joint-level occlusion / confidence dropout
* mixed H36M (4 views) and MPI-INF-3DHP (14 views) domains via the mixed loader

---

## 3. Training Command / Ablation Flags

### Smoke test (RTX 4090)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --use_multiview_geometry_fusion_v25 \
  --v25_geom_loss_weight 0.1 \
  --use_uncertainty_aware_triangulation_v33 \
  --v33_uat_loss_weight 0.01
```

### Full ablation (H36M + MPI mixed, matches v32 baseline)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
  --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding \
  --use_deformable_cross_view_attention_v18 \
  --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
  --v25_use_geometry_attention --v25_use_learned_depth_triangulation \
  --use_uncertainty_aware_triangulation_v33 --v33_uat_loss_weight 0.01 \
  --num_workers 4 --d 64 --residual_hidden 128 --n_st_layers 2 \
  --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
  --clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 \
  --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 \
  --early_stopping_patience 5 --early_stopping_min_delta 0.001 \
  --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
  --outlier_view_prob 0.3 --outlier_view_max_views 1 \
  --output outputs/omniview_fusion_v33_uncertainty_aware_tri.pth
```

### Ablations to run

1. `v33_uat_only` — enable only the new uncertainty head, keep v25 geometry attention and depth proposals on.
2. `v33_uat_no_v27` — enable the new head but disable `use_uncertainty_depth_proposals_v27` to isolate 2-D uncertainty from depth uncertainty.
3. `v33_uat_low_weight` — set `v33_uat_loss_weight=0.001` to verify the loss does not dominate early training.
4. `v33_uat_high_weight` — set `v33_uat_loss_weight=0.1` to test whether stronger uncertainty supervision helps.
5. `v33_uat_variable_views` — run with aggressive outlier augmentation to measure robustness.

---

## 4. Expected Metrics and Baseline to Beat

### Primary metric

* **val_MPJPE** on the mixed H36M + MPI-INF-3DHP validation set.

### Baselines

Based on the current status table in `AGENTS.md`:

* v25 full (v18 + geometry fusion): best reported ~21.54 mm (v29o hierarchical-only).
* v26/v27 full: ~40–44 mm before overfitting; v27 UDP did not clearly beat v25.
* v32 combined (domain-aware + trajectory consistency + physical alignment): under evaluation; use whatever the v32 queue reports as its best.

**Success criteria for v33:**

| Metric | Target |
|--------|--------|
| val_MPJPE on mixed H36M+MPI | ≤ best v32/v25 baseline by ≥ 1 mm (e.g., ≤ 20.5 mm if v25 baseline is 21.5 mm) |
| val_MPJPE under outlier-view augmentation (`outlier_view_prob=0.3`) | ≥ 5% relative improvement over baseline with same augmentation |
| Variable-view (2–4 active views) MPJPE | ≥ 5% relative improvement over baseline |
| Training stability | No divergence in first 3 epochs with default `v33_uat_loss_weight=0.01` |

### Evaluation protocol

Run the standard `TrainerV2` validation loop.  For robustness reporting, add a **post-hoc** evaluation pass with:

* fixed 2-view, 3-view, 4-view subsets on H36M validation
* outlier injection at inference (`outlier_view_prob=0.3`, no training change)
* per-joint MPJPE breakdown to check if uncertainty helps wrists/ankles (joints with frequent self-occlusion)

---

## 5. Risks / Unknowns

| Risk | Mitigation |
|------|------------|
| **NLL loss dominates early training** and destabilises the geometry-fusion block. | Start with `v33_uat_loss_weight=0.01`; add a warmup so the loss ramps up over the first 3 epochs; clamp `log_var` aggressively. |
| **Uncertainty head collapses** to a trivial constant (e.g., all views equally uncertain). | Supervise with reprojection NLL on every view; regularise with an entropy term on predicted precisions; monitor per-view variance histograms. |
| **Redundancy with v27 UDP** — both model uncertainty, so gains may be small. | Run ablation `v33_uat_no_v27` explicitly; target the *complementarity* of 2-D (this proposal) vs. depth (v27) uncertainty. |
| **Variable-view training masks cause degenerate covariance** when very few views remain. | Apply a minimum-views guard; if active views < 2 for a joint, fall back to the v25 estimate. |
| **Computational overhead** from the Cholesky-based weighted DLT. | The DLT system is small (`2V × 4`); overhead should be < 5%.  If it is too slow, cache the DLT matrix and only re-weight rows. |
| **Conflicting gradient path** because the uncertainty head and residual MLP both refine the same 3-D estimate. | Identity-at-init residual scale keeps gradients well-behaved at start; use `scale` parameter initialised to 0. |
| **Calibration of predicted uncertainty** may not correlate with actual errors. | Report reprojection-NLL on validation as a calibration diagnostic, not just MPJPE. |

---

## 6. Implementation Sketch (for the swarm coder)

1. Create `motionflow_mv/fusion/uncertainty_aware_triangulation_v33.py` with:
   * `_UncertaintyMLP` (predicts 2-D log-variance from features)
   * `_precision_weighted_dlt(...)` wrapper around `uncertainty_weighted_triangulation.triangulate_uncertainty_weighted_batched`
   * `UncertaintyAwareTriangulationV33` module
2. Add flags to `OmniMultiViewFusionV5.__init__` and instantiate the module.
3. Insert the call site in `OmniMultiViewFusionV5.forward` after the v25 block.
4. Add CLI args in `experiments/train_omniview_fusion_v5_webbridge_multi.py` and plumb through `build_model_from_args`.
5. Smoke test: `python experiments/train_omniview_fusion_v5_webbridge_multi.py --smoke --use_uncertainty_aware_triangulation_v33`.
6. Run ablations and report val_MPJPE / robustness metrics.

---

## 7. Relation to Prior Iterations

* **v25:** extends the geometry-fusion block with explicit 2-D uncertainty.
* **v27:** complements `UncertaintyDepthProposalTriangulation` (depth uncertainty) rather than replacing it.
* **v31/v32:** can be stacked with hierarchical encoders (`use_hierarchical_multiview_v31`) and trajectory-consistency refiners (`use_trajectory_consistency_v32`).
* **Earlier uncertainty models (crossview_residual_uncertainty.py):** re-uses the same log-variance / reprojection-NLL idea, but applied to the current v5 mainline instead of the legacy ray-attention model.
