# v33: Robust Outlier-View Rejection with Adaptive Thresholds

**Slug:** `outlier_view_rejection`  
**Date:** 2026-08-08  
**Target stack:** `OmniMultiViewFusionV5` (WebBridge / H36M / MPI-INF-3DHP mixed training)  
**Baseline to beat:** v32 combined stack with `--use_multiview_geometry_fusion_v25 --v25_use_outlier_view_detector`, anchor ~21.5 mm MPJPE on mixed validation.

---

## 1. Problem statement and motivation

The codebase already contains a differentiable `OutlierViewDetector`, but it is only wired inside the v25 geometry-fusion block and relies on a hand-tuned z-score threshold (`v25_outlier_z_thresh=3.0`, `v25_outlier_soft_beta=1.0`). As a result:

1. **Main DLT is not outlier-aware.** The primary triangulation path in `OmniMultiViewFusionV5` multiplies `weights = confidence * precision * visibility` but does not use the detector output. A single corrupted view can therefore still bias the initial 3-D estimate.
2. **Thresholds are static.** A fixed `z_thresh=3.0` may be too lenient for WebBridge clips with calibration noise and too aggressive for MPI clips with motion blur. The per-joint adaptive scales exist but are not learned from richer cues.
3. **No explicit supervision.** The detector is trained only indirectly through downstream MPJPE. The augmentation mask from `inject_outlier_views` is available but unused as a direct target.
4. **Redundancy with selector.** `AdaptiveViewSelector` chooses views, but without outlier information it may select corrupted views. Combining the two should yield both fewer and cleaner views.

**Goal for v33:** upgrade the existing `OutlierViewDetector` into a learned, feature-aware outlier-rejection head that operates in the main DLT path, adapts its thresholds per joint / body part / domain, and is optionally supervised by the augmentation mask.

---

## 2. Proposed architecture changes

### 2.1 Refactor / extend `OutlierViewDetector`

Create a v33-enhanced version in `motionflow_mv/fusion/outlier_view_detector_v33.py`:

```
OutlierViewDetectorV33
    ├── __init__(
    │       z_thresh=3.0, soft_beta=1.0, min_mad=0.5,
    │       adaptive=True, num_joints=J, num_parts=P,
    │       feature_dim=d, use_feature_gate=True, use_part_scale=True,
    │       use_domain_scale=True, supervised_gate=True
    │   )
    └── forward(
            pred_3d, points_2d, K, R, t,
            features=None,     # (B,T,V,J,d) per-view spatio-temporal tokens
            domains=None,      # (B,T) domain id for H36M/MPI/WebBridge
            view_mask=None,
            outlier_label=None # (B,T,V,J) optional augmentation mask target
        ) -> weights, aux_loss
```

**Internal design:**

1. **Residual computation.** Re-use `compute_reprojection_residual` to obtain per-view, per-joint L2 residuals `r ∈ R^(B,T,V,J)`.
2. **Robust statistics.** Compute view-wise median `med` and MAD over active views.
3. **Feature-aware residual adjustment.** A 2-layer MLP on the per-view token `features` predicts a residual offset `Δr` and a scalar attention `α`:
   ```
   Δr, α = MLP(features)  # (B,T,V,J,1) each
   r_adj = r + σ(α) · Δr
   ```
   This lets the model learn that a view with high 2-D confidence but large epipolar disagreement is still suspicious. The offset is initialized to zero and `α` to a negative value so the block starts as identity.
4. **Adaptive thresholds.** Replace fixed `z_thresh` and `soft_beta` with:
   - Global learnable scales.
   - Per-joint scales (existing).
   - **Per-part scales** (`num_parts=5`: torso, arms, legs, head, hands/feet).
   - **Per-domain scales** via a small embedding (`nn.Embedding(3, 2)` for H36M/MPI/WebBridge).
   The effective threshold for a joint `j` is:
   ```
   z_eff[j]    = z_thresh    · z_scale_global · z_scale_joint[j] · z_scale_part[p(j)] · z_scale_domain[d]
   beta_eff[j] = soft_beta   · β_scale_global · β_scale_joint[j] · β_scale_part[p(j)] · β_scale_domain[d]
   ```
5. **Soft down-weighting.** Same exponential gate as v25, but applied to `r_adj`:
   ```
   margin   = max(0, z_score - z_eff)
   weight   = 1 - gate · (1 - exp(-beta_eff · margin))
   gate     = sigmoid(residual_scale), residual_scale init = 0
   ```
6. **Optional supervised loss.** When `outlier_label` is provided (augmented outlier views), add a small BCE:
   ```
   L_sup = BCE(1 - weight, outlier_label)   # 0 = clean, 1 = outlier
   ```
   This is weighted by `v33_outlier_supervised_weight`.

### 2.2 Wire into the main DLT path

In `motionflow_mv/fusion/omniview_fusion_v5.py`, around the existing triangulation (currently `weights = weights * confidences * precision * visibility`):

```python
if self.use_outlier_view_rejection_v33:
    outlier_weights, outlier_loss = self.outlier_view_detector_v33(
        pred_3d_init.detach(),  # use current estimate, stop gradient to keep it stable
        points_2d=points_2d.view(B, T, V, J, 2),
        K=K_corrected.view(B, T, V, 3, 3),
        R=R.view(B, T, V, 3, 3),
        t=t.view(B, T, V, 3),
        features=feat.view(B, T, V, J, d),
        domains=domain_ids,
        view_mask=view_mask_flat.view(B, T, V),
        outlier_label=outlier_labels,  # from augmentation
    )
    weights = weights * outlier_weights
    geom_loss_v25 = geom_loss_v25 + self.v33_outlier_supervised_weight * outlier_loss
```

The same `outlier_weights` are also passed into the v25 geometry-fusion block so it does not recompute them.

### 2.3 Constructor / CLI flags

Add to `OmniMultiViewFusionV5`:

```python
use_outlier_view_rejection_v33: bool = False
v33_outlier_z_thresh: float = 3.0
v33_outlier_soft_beta: float = 1.0
v33_outlier_use_feature_gate: bool = True
v33_outlier_use_part_scale: bool = True
v33_outlier_use_domain_scale: bool = True
v33_outlier_supervised_weight: float = 0.1
v33_outlier_feature_hidden: int = 64
```

Add to `experiments/train_omniview_fusion_v5_webbridge_multi.py` / `build_model_from_args`:

```bash
--use_outlier_view_rejection_v33
--v33_outlier_z_thresh 3.0
--v33_outlier_soft_beta 1.0
--v33_outlier_use_feature_gate
--v33_outlier_use_part_scale
--v33_outlier_use_domain_scale
--v33_outlier_supervised_weight 0.1
--v33_outlier_feature_hidden 64
```

### 2.4 Optional: feed into `AdaptiveViewSelector`

If `--use_adaptive_view_selection` is also on, multiply the selector logits by `outlier_weights` before Gumbel-softmax top-k so outlier views are not even candidates.

---

## 3. Data and preprocessing

No new dataset is required. The module consumes:

- `points_2d`: `(B, T, V, J, 2)`
- `K, R, t`: camera parameters
- `features`: per-view spatio-temporal tokens from the ST transformer
- `domain_ids`: `(B,)` or `(B,T)` enumerating H36M/MPI/WebBridge
- `view_mask`: `(B, T, V)`
- `outlier_labels`: `(B, T, V, J)` boolean mask from `inject_outlier_views` (available in the training script)

The augmentation `inject_outlier_views` already produces corrupted views with probability `outlier_view_prob`; we re-use its mask as optional supervision. For validation, `outlier_labels=None` and the module acts in fully unsupervised mode.

---

## 4. Training command / ablation flags

### Smoke test (RTX 4090)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --smoke \
    --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 \
    --v25_use_outlier_view_detector \
    --use_outlier_view_rejection_v33 \
    --v33_outlier_z_thresh 3.0 \
    --v33_outlier_soft_beta 1.0 \
    --v33_outlier_supervised_weight 0.1
```

### Full ablation (matches v32 baseline)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
    --v25_use_geometry_attention --v25_use_learned_depth_triangulation \
    --v25_use_outlier_view_detector \
    --use_outlier_view_rejection_v33 \
    --v33_outlier_z_thresh 3.0 --v33_outlier_soft_beta 1.0 \
    --v33_outlier_use_feature_gate --v33_outlier_use_part_scale --v33_outlier_use_domain_scale \
    --v33_outlier_supervised_weight 0.1 \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --num_workers 4 --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --early_stopping_patience 5 --early_stopping_min_delta 0.001 \
    --output outputs/omniview_fusion_v33_outlier_view_rejection.pth
```

### Ablations to run

| Flag set | Purpose |
|----------|---------|
| `v33_outlier_main_dlt_only` | Enable v33 detector in main DLT, disable inside v25 to isolate main-path impact. |
| `v33_outlier_no_supervised_loss` | Set `v33_outlier_supervised_weight=0.0` to test unsupervised adaptation. |
| `v33_outlier_static_thresh` | Disable adaptive scales (`use_feature_gate=False, use_part_scale=False, use_domain_scale=False`). |
| `v33_outlier_aggressive` | `z_thresh=2.0, soft_beta=2.0` — test robustness vs. false positives. |
| `v33_outlier_plus_selector` | Combine with `--use_adaptive_view_selection` and multiply selector logits by outlier weights. |

---

## 5. Expected metrics and baseline to beat

Primary evaluation on the mixed WebBridge/H36M/MPI validation split in `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`.

| Metric | Baseline (v32 + v25 outlier detector) | v33 target |
|--------|--------------------------------------|------------|
| `val_MPJPE` (mixed) | best v32 combined checkpoint | ≤ baseline − 1 mm |
| `val_MPJPE` under `outlier_view_prob=0.3` at inference | baseline with same augmentation | ≥ 5% relative improvement |
| Variable-view MPJPE (2–4 active views) | v32 curve | ≥ 5% relative improvement |
| Outlier recall @ 0.5 | — | ≥ 0.80 on augmented validation clips |
| Clean-view precision @ 0.5 | — | ≥ 0.95 on clean validation clips |

Secondary metrics:

- **Robustness sweep:** evaluate at `outlier_view_max_views = 1, 2, 3` with `outlier_view_offset_std = 10, 20, 30` and report MPJPE delta vs. baseline.
- **View-count robustness:** fix active views to `{2, 3, 4, 8, 14}` and report per-view-count MPJPE.
- **Calibration:** report mean predicted outlier weight vs. actual reprojection residual on validation.

---

## 6. Risks / unknowns

| Risk | Why | Mitigation |
|------|-----|------------|
| **Chicken-and-egg in main DLT.** The detector uses `pred_3d_init`, which is itself computed from potentially corrupted views. | If initial triangulation is already biased, residuals are misleading. | Detach the input 3-D estimate; use the v25 robust-DLT/IRLS output as the seed; add a second refinement pass. |
| **Over-aggressive rejection.** Per-part/domain scales may suppress clean-but-noisy views (e.g., MPI motion blur). | Lower `val_MPJPE` on clean clips. | Start with gate initialized near identity; clamp scales; run `v33_outlier_static_thresh` ablation. |
| **Supervised loss conflicts with augmentation.** The augmentation mask labels entire views as outliers, but a view may have only one bad joint. | BCE at joint level may over-penalize. | Use view-level labels only as a weak target; weight the loss by `1 - confidence`. |
| **Gradient instability from feature gate.** `Δr` can push residuals negative or unbounded. | Training divergence. | Clamp `Δr` to `[-r, r]`; use layer norm; initialize `α` to a large negative value. |
| **Redundancy with v25 detector.** Running two outlier detectors may not improve metrics enough to justify complexity. | Ablation shows only marginal gain. | Make the v33 detector the single source of truth and remove / disable the v25-internal detector when v33 is on. |
| **Domain-scale overfit.** Per-domain embeddings can memorize dataset-specific noise rather than learn robust thresholds. | MPI scale collapses / explodes. | Use 3 domains max; regularize with small L2 on domain scales; share scales across similar domains. |

---

## 7. Definition of done

- [ ] `motionflow_mv/fusion/outlier_view_detector_v33.py` created with forward/backward smoke test.
- [ ] Flags `--use_outlier_view_rejection_v33` and ablation flags plumbed through `experiments/train_omniview_fusion_v5_webbridge_multi.py` and `build_model_from_args`.
- [ ] Detector wired into the main DLT path in `omniview_fusion_v5.py`.
- [ ] Optional supervised loss consumes the augmentation mask from `inject_outlier_views`.
- [ ] Smoke run completes with `val_MPJPE` lower than the corresponding v32-only smoke run.
- [ ] Outlier recall / precision measured on augmented validation clips.
- [ ] Full run queued on A800 or local RTX 4090 with a clear comparison checkpoint.
