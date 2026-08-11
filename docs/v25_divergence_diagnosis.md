# v25 H36M True-GT Divergence Diagnosis

**Date:** 2026-08-11  
**Run:** `scripts/run_v25_h36m_true_gt_medium_local_4090.sh`  
**Log:** `outputs/omniview_fusion_v25_h36m_true_gt_medium.log`  
**Checkpoint:** `outputs/omniview_fusion_v25_h36m_true_gt_medium.pth` (best epoch 2)  
**Status:** Divergence after epoch 2; training loss falls while validation MPJPE explodes.

## 1. Symptom Summary

The local RTX 4090 medium run completed 8 epochs. Validation MPJPE improves until epoch 2 and then monotonically degrades:

| Epoch | val MPJPE (mm) | train_loss | val_loss   |
|------:|---------------:|-----------:|-----------:|
| 1     | 83.19          | 6.462      | 0.002519   |
| 2     | **72.80**      | 6.475      | 0.001976   |
| 3     | 73.12          | 6.148      | 0.002067   |
| 4     | 78.33          | 5.930      | 0.002451   |
| 5     | 88.38          | 5.861      | 0.003625   |
| 6     | 115.98         | 5.855      | 0.007086   |
| 7     | 159.43         | 5.742      | 0.013444   |
| 8     | 207.62         | 5.782      | 0.021829   |

* The best checkpoint is **epoch 2** (`val_MPJPE = 72.80 mm`).
* `train_loss` keeps decreasing, but `val_loss` and `val_MPJPE` rise after epoch 2.
* No NaN/Inf was emitted; the run finished normally.

This is classic **overfitting** of a high-capacity model to a small per-epoch training set, identical in shape to the v80 true-GT collapse documented in `docs/results_v80_h36m_true_gt.md` and the v25-small overfit in `docs/swarm_iter_next/v25_small_overfit_analysis.md`.

## 2. Root-Cause Analysis

### 2.1 Too few training samples per epoch

The script uses:

```bash
--epochs 8 --batch_size 16 --train_samples 1024 --val_stride 20
```

That is only **64 gradient steps per epoch** for a model with **2.73 M parameters** (`OmniMultiViewFusionV5` with v25 geometry fusion). For comparison, full-scale A800 v25/v31 recipes use `--train_samples 4000` and often `--train_samples 10000` (`scripts/launch_v31_paper_story_multiview_video_pipeline_a800.sh`).

With only 1024 unique samples per epoch, the model quickly memorises the small training batch distribution; the learned v25 depth-proposal and geometry-attention weights over-fit to the training views and fail to generalise to S9/S11.

### 2.2 Early stopping is disabled

The saved config shows:

```json
"early_stopping_patience": 0,
"early_stopping_min_delta": 0.0,
```

Because `early_stopping_patience=0`, the trainer never stops before the requested 8 epochs, so it continues training long after the best epoch and overwrites good generalisation with severe overfitting. The A800 v25/v31 recipes consistently set `--early_stopping_patience 3 --early_stopping_min_delta 0.001`.

### 2.3 No weight decay

```json
"weight_decay": 0.0
```

v80 true-GT showed the same pattern: every recipe overfits after epoch 2, and the only ones that even reached ~40 mm used non-zero weight decay (`1e-4` to `2e-4`). The current v25 medium run has no L2 regularisation at all, leaving the large residual MLPs and attention layers free to over-fit.

### 2.4 Strong augmentation on a small dataset

The script enables:

```bash
--outlier_view_prob 0.3 --outlier_view_max_views 1 \
--outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
--use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4
```

Injecting corrupted views 30 % of the time is useful with ample data, but with only 1024 training samples per epoch it dilutes the already-limited signal and pushes the model toward memorising the augmentation pattern rather than the true geometry.

### 2.5 v25 geometry module has weak regularisation

In `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`:

* `GeometryAwareCrossViewAttention` and `DepthProposalTriangulation` use dropout, but the depth head’s residual gate `self.residual_scale` is initialised to `0.0` and has **no bound or decay**.
* The geometry loss inside v25 is a reprojection loss (`_reprojection_loss`) scaled by `v25_geom_loss_weight=0.1`. With no weight decay, the module can drive this loss down by distorting 3-D geometry to fit training views, hurting validation MPJPE.

### 2.6 Learning-rate schedule is too aggressive for fine-tuning true-GT

```bash
--lr 1e-3 --lr_cosine --lr_warmup_epochs 1 --lr_min 1e-6
```

A warm-up of only 1 epoch followed by cosine decay is fine for a full mixed-dataset run, but on the smaller true-GT medium set it means the model spends most of its training at a relatively high learning rate after epoch 2, amplifying any overfitting signal. The v80 recipe sweep found that lowering `lr` to `5e-4` or `2e-4` delayed (but did not eliminate) the collapse; combining lower `lr` with weight decay and early stopping is more robust.

### 2.7 Config-bookkeeping inconsistency

The saved `config.json` lists:

```json
"manifest": ["configs/splits/webbridge_all_train.yaml"],
"mixed_manifest": "configs/splits/h36m_true_gt_standard.yaml",
```

Because the script passes `--use_mixed_loader --mixed_manifest configs/splits/h36m_true_gt_standard.yaml`, the actual training used the true-GT protocol. However, the `manifest` field being set to `webbridge_all_train.yaml` is misleading for reproducibility and should be cleaned up so the config reflects the data actually used.

## 3. Actionable Fixes

### Immediate (preserve the best result now)

1. **Use the epoch-2 checkpoint for evaluation.**
   The current `outputs/omniview_fusion_v25_h36m_true_gt_medium.pth` is already the best epoch (72.80 mm) because the trainer saves the best `val_MPJPE`. Do not use `*_final.pth`, which was produced at the end of epoch 8.

2. **Disable the stale `*_final.pth` confusion.**
   The `*_final.pth` at epoch 8 (207.62 mm) is not the checkpoint to report. Rename or remove it after deciding on the canonical checkpoint naming.

### Short-term (re-run on local 4090)

Update `scripts/run_v25_h36m_true_gt_medium_local_4090.sh` with:

```bash
# 1. More samples per epoch
--train_samples 4096 \

# 2. Early stopping -- stop as soon as validation stops improving
--early_stopping_patience 3 --early_stopping_min_delta 0.001 \

# 3. Weight decay
--weight_decay 1e-4 \

# 4. Slightly lower learning rate or longer warmup
--lr 5e-4 --lr_warmup_epochs 2 \

# 5. Reduce outlier augmentation intensity on the smaller set
--outlier_view_prob 0.15 \

# 6. Increase v25 dropout
--v25_dropout 0.2 \
```

These values mirror the A800 v31/v25 recipes and the v80 best-practice recipe (v2: `lr=5e-4`, `weight_decay=1e-4`).

### Medium-term (if divergence persists)

1. **Progressive unfreezing / head-only warm-up.**
   Freeze the base `OmniMultiViewFusionV5` feature extractor for the first 1–2 epochs and train only the v25 geometry-fusion head, then unfreeze. This prevents the large base network from overfitting to the small true-GT set before the geometry head has stabilised.

2. **Add explicit geometry-regularisation losses.**
   * Bone-length consistency loss (already available via `--bone_loss_weight`).
   * Joint-limit loss (via `--joint_limit_weight`).
   * Temporal bone-length loss (via `--temporal_bone_weight`).
   These regularise the v25 depth-proposal head and reduce implausible poses.

3. **Mixed-loader with MPI-INF-3DHP or AIST++.**
   The true-GT H36M training set (~390 k frames) is small compared with the model capacity. Using `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml` or the AIST++ smoke/full split increases diversity and was the intended use for the v25 recipes on A800.

4. **Reduce v25 geometry loss weight or add warmup for it.**
   The v25 internal reprojection loss is added to `epi_loss` with weight `0.1`. If the geometry head overfits, try `--v25_geom_loss_weight 0.05` or ramp it from 0 over the first 2 epochs.

5. **Bound the v25 residual gate.**
   In `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`, the `residual_scale` parameter is unbounded. Adding a soft clamp (e.g., `tanh(residual_scale) * max_delta`) or a small L2 penalty on `residual_scale` would prevent the depth-proposal head from drifting too far from the initial DLT estimate.

## 4. Recommended Re-run Command

A conservative, evidence-based re-run for local RTX 4090 is:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_standard.yaml \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.05 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --num_workers 0 \
    --d 128 --residual_hidden 256 --n_st_layers 3 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 20 --batch_size 16 --train_samples 4096 --val_stride 20 \
    --lr 5e-4 --lr_cosine --lr_warmup_epochs 2 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --weight_decay 1e-4 \
    --early_stopping_patience 3 --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/omniview_fusion_v25_h36m_true_gt_medium_v2.pth \
    > outputs/omniview_fusion_v25_h36m_true_gt_medium_v2.log 2>&1
```

**GPU concurrency check required before launching.** Confirm `nvidia-smi` shows no other training process; the RTX 4090 can run at most one training task at a time per project rules.

## 5. Comparison with v80 and v25-small

| Run | Best val MPJPE | Best epoch | Post-best behaviour | Key regularisation |
|---|---:|---:|---|---|
| v25 true-GT medium | 72.80 mm | 2 | Monotone explosion | None (this run) |
| v80 true-GT v2 | 39.70 mm | 2 | Diverges | lr=5e-4, wd=1e-4 |
| v80 true-GT v3 | 42.60 mm | 2 | Diverges | lr=2e-4, wd=5e-5, early stop |
| v25 small | 18.31 mm* | 1 | Overfits | early stop added |

\* v25 small was on a different (circular-label) small split and is not directly comparable in absolute MPJPE, but the shape of the curve is identical.

The consistent pattern is: **best generalisation is reached at epoch 1–2**, and every run without early stopping and weight decay degrades afterward. The fixes above address exactly that.

## 6. Files to Modify

1. `scripts/run_v25_h36m_true_gt_medium_local_4090.sh` — apply the short-term hyperparameter changes.
2. `motionflow_mv/fusion/multiview_geometry_fusion_v25.py` — optional: bound `residual_scale` and/or add warmup for `v25_geom_loss_weight`.
3. `docs/results_true_gt_h36m.md` — update the v25 row when the re-run completes.

## 7. Verification Checklist

- [ ] Re-run uses `--train_samples 4096` (or higher) and `--early_stopping_patience 3`.
- [ ] `weight_decay > 0` is set.
- [ ] `nvidia-smi` shows the GPU is free before starting the re-run.
- [ ] Best checkpoint is selected by `val_MPJPE`, not the final epoch file.
- [ ] If divergence persists after the fixes above, consider mixed-dataset training (`configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`) or progressive unfreezing.
