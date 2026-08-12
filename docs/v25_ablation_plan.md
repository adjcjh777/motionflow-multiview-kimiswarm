# v25 H36M True-GT Ablation Plan

**Date:** 2026-08-11  
**Related:** `docs/v25_divergence_diagnosis.md`

## Background

The v25 H36M true-GT medium run diverged after epoch 2 (best val MPJPE **72.80 mm**, final epoch 8 **207.62 mm**). `docs/v25_divergence_diagnosis.md` identifies the root causes as:

1. Too few unique samples per epoch (`--train_samples 1024`).
2. Early stopping disabled (`--early_stopping_patience 0`).
3. No weight decay (`--weight_decay 0.0`).
4. Aggressive learning-rate schedule (`--lr 1e-3`, 1-epoch warmup).
5. Strong outlier augmentation (`--outlier_view_prob 0.3`) on a small per-epoch set.
6. Weak v25 geometry regularisation (no bone/joint-limit losses, unbounded depth-proposal head).
7. v25 internal geometry loss weight possibly too high for a small true-GT set.

This plan defines three concrete ablations that test the proposed fixes in isolation and in combination. All three are designed for the local RTX 4090 and must be run one at a time.

## Ablations

| # | Config | Hypothesis | Key changes vs. diverged baseline |
|---|--------|------------|-----------------------------------|
| 1 | `configs/ablations/v25_true_gt_baseline_fix.yaml` | The divergence is purely hyperparameteric. | `train_samples 4096`, `early_stopping_patience 3`, `weight_decay 1e-4`, `lr 5e-4`, `lr_warmup_epochs 2`, `outlier_view_prob 0.15`, `v25_dropout 0.2`, `v25_geom_loss_weight 0.05` |
| 2 | `configs/ablations/v25_true_gt_geometry_regularization.yaml` | The v25 geometry head needs explicit 3-D constraints. | All of ablation 1 + `bone_loss_weight 0.05`, `joint_limit_weight 0.01`, `temporal_bone_weight 0.005` |
| 3 | `configs/ablations/v25_true_gt_mixed_dataset.yaml` | The true-GT H36M training set is too small for the model capacity. | All of ablation 1, but train on `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml` (H36M + MPI-INF-3DHP) |

## Config files

- `configs/ablations/v25_true_gt_baseline_fix.yaml`
- `configs/ablations/v25_true_gt_geometry_regularization.yaml`
- `configs/ablations/v25_true_gt_mixed_dataset.yaml`

Each config is a single-run reference YAML. The trainer currently consumes command-line flags, so each YAML contains a `training:` block that maps 1:1 to CLI arguments. A convenience script can be generated from any of them.

## Launch scripts

### Ablations 1 and 2 (H36M true-GT only)

```bash
#!/usr/bin/env bash
set -euo pipefail

# Ablation 1
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
    --v25_dropout 0.2 \
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
    --output outputs/ablations/v25_true_gt_baseline_fix.pth \
    > outputs/ablations/v25_true_gt_baseline_fix.log 2>&1
```

Ablation 2 uses the same command plus:

```bash
    --bone_loss_weight 0.05 \
    --joint_limit_weight 0.01 \
    --temporal_bone_weight 0.005 \
    --output outputs/ablations/v25_true_gt_geometry_regularization.pth \
    > outputs/ablations/v25_true_gt_geometry_regularization.log 2>&1
```

### Ablation 3 (mixed dataset)

```bash
#!/usr/bin/env bash
set -euo pipefail

python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.05 \
    --v25_dropout 0.2 \
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
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v25_true_gt_mixed_dataset.pth \
    > outputs/ablations/v25_true_gt_mixed_dataset.log 2>&1
```

## Expected outcomes and decision logic

1. **If ablation 1 reaches a stable best val MPJPE <= 70 mm and does not explode after the best epoch**, then the short-term hyperparameter fixes are sufficient. Use ablation 1 as the new v25 true-GT recipe and update `scripts/run_v25_h36m_true_gt_medium_local_4090.sh` accordingly.

2. **If ablation 1 still diverges but ablation 2 stabilises**, then the v25 geometry head needs explicit 3-D constraints. Adopt ablation 2 (or a lighter variant) as the recipe.

3. **If ablation 1 diverges but ablation 3 stabilises**, then the true-GT H36M training set alone is too small. Use the mixed-loader recipe and report val MPJPE on the H36M true-GT val split.

4. **If all three still diverge**, the problem is structural and requires the medium-term fixes in `docs/v25_divergence_diagnosis.md` Section 3:
   - Progressive unfreezing (freeze base feature extractor, warm up the v25 head).
   - Bound the v25 `residual_scale` or add an L2 penalty on it.
   - Reduce `v25_geom_loss_weight` further or warm it up.
   - Consider AIST++ mixed data (`configs/splits/h36m_true_gt_aist_mixed_smoke.yaml`) once the AIST++ integration is ready.

## Evaluation protocol

For each completed run, record:

- Best val MPJPE (mm) and the epoch at which it occurred.
- Final epoch val MPJPE (mm) to detect post-best collapse.
- Whether the best checkpoint is selected by `val_MPJPE` (standard trainer behaviour).

Report results in `docs/results_true_gt_h36m.md` under a new section, or create `docs/results_v25_true_gt_ablations.md` if the table grows.

## Safety / constraints

- The local RTX 4090 can run **at most one training task at a time**.
- Before launching any ablation, verify the GPU is free with `nvidia-smi` and confirm no other `python.exe` training process is active.
- A800-D and the Docker `motionflow` service are **read-only**; do not start or modify anything there.
- Do not start a GPU task while another agent's run is in progress.

## Checklist before running

- [ ] `nvidia-smi` shows the GPU is free.
- [ ] `outputs/ablations/` directory exists (create with `mkdir -p outputs/ablations` if not).
- [ ] The run uses the true-GT val protocol for evaluation (S9/S11).
- [ ] Best checkpoint is selected by `val_MPJPE`, not the final epoch file.
