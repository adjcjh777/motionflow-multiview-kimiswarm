# H36M True-GT v2 Leaderboard Re-run Plan

> **Goal:** Re-train and re-evaluate the key MotionFlow-MultiView variants on the corrected, non-circular H36M true-GT **v2** labels, producing an internally consistent CVPR 2027 leaderboard.
> **Models covered:** v25, v46, v52, v57, v80, v81, v82, v85.
> **Data split:** `configs/splits/h36m_true_gt_v2_standard.yaml` (S1,5,6,7,8 → S9/S11).
> **Last updated:** 2026-08-12

## 1. Why a v2 re-run is needed

The existing `data/h36m_true_gt/` files were generated before the camera-alignment fix in `scripts/convert_h36m_true_gt_v2.py`.  Their stored 3D mocap coordinates are **physically inconsistent** with the stored 2D keypoints and camera parameters (direct MJE ≈ 16.7 m).  The v2 labels (`data/h36m_true_gt_v2/`) re-project the official mocap 3D GT into a consistent 2D/camera frame, giving direct MJE in the tens of millimetres.

Consequences:
- All numbers in `docs/results_true_gt_h36m.md` labelled *historical true-GT v1* are on the misaligned data set and must be replaced.
- The geometric baselines (DLT, RANSAC) have already been re-run on v2 and are stable (see Section 3).
- Every learned model from v25 through v85 must be re-trained **from scratch** on the v2 split using the exact same hyperparameters, then re-evaluated on the v2 S9/S11 test files.

## 2. Hard constraints

- **A800 GPUs:** only GPU 6 and GPU 7 may be used.  `CUDA_VISIBLE_DEVICES` must be `6` or `7`.
- **Do not stop or interfere** with currently running jobs:
  - GPU 7: `v85_random_view_dropout_medium_a800` (training)
  - GPU 6: `v86_no_count_embedding_medium_a800` (training)
- `/mnt/nvme0n1p1/zhangzy/projects` and the A800 Docker `motionflow` service are **read-only**.
- A800 disk is ~99 % full; run the safe cleanup dry-run before any large write.
- WSL/local RTX 4090 is for smoke tests only; full medium/long runs stay on A800.

## 3. Pre-requisites

### 3.1 Generate and verify the v2 labels

```bash
# 1. Local WSL: regenerate all v2 .npz files
bash scripts/convert_all_h36m_true_gt_v2.sh

# 2. Audit a few files (direct MJE should be tens of mm, not 0 or thousands)
python scripts/diagnose_circular_labels.py data/h36m_true_gt_v2/s_01_acts_*.npz
python scripts/diagnose_circular_labels.py data/h36m_true_gt_v2/s_09_acts_*.npz

# 3. Sync to A800 (run from WSL)
bash scripts/sync_h36m_true_gt_v2_to_a800.sh
```

- [ ] `data/h36m_true_gt_v2/` contains 7 train `.npz` files (S1,5,6,7,8) and 2 test `.npz` files (S9, S11).
- [ ] `configs/splits/h36m_true_gt_v2_standard.yaml` resolves every path.
- [ ] Direct MJE audit is in the tens of millimetres on the test files.

### 3.2 Re-run v2 geometric baselines

These are fast and provide the reference anchors for the leaderboard.

```bash
bash scripts/run_h36m_true_gt_v2_baselines.sh
```

Expected outputs:
- `outputs/h36m_true_gt_v2_baselines/dlt_baseline_h36m_true_gt_v2.json`
- `outputs/h36m_true_gt_v2_baselines/ransac_baseline_h36m_true_gt_v2.json`

- [ ] Confidence-weighted DLT v2 result recorded.
- [ ] RANSAC/conf-DLT v2 result recorded.

### 3.3 Re-run Iskakov learnable triangulation on v2

The Iskakov script does not use the WebBridge manifest; it uses the `data_3d_h36m.npz` release.  Verify it still points at the same true mocap source and re-run on v2-equivalent data.

```bash
python experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m --epochs 10 --batch_size 8 --lr 1e-3 --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --log_path outputs/iskakov_h36m_true_gt_v2.log \
    --ckpt_path outputs/iskakov_h36m_true_gt_v2.pth
```

- [ ] Iskakov v2 result recorded.

### 3.4 Create the v2 variable-view manifest

Variable-view eval scripts need a manifest that points at the v2 test files:

```bash
cat > tmp/h36m_true_gt_v2_val_manifest.txt <<'EOF'
S9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
S11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
EOF
```

- [ ] `tmp/h36m_true_gt_v2_val_manifest.txt` exists and points to v2 test files.

### 3.5 Free disk and queue resources

- [ ] Run `scripts/cleanup_a800_safe.sh` dry-run and inspect output.
- [ ] Confirm GPU 6 and GPU 7 are free (do not disturb the running v85/v86 jobs).
- [ ] Decide serial vs. queue order: v85/v86 finish first, then schedule the other models.

## 4. Common adaptation for every learned model

Every training script currently points at the v1 manifest:

```bash
--mixed_manifest configs/splits/h36m_true_gt_standard.yaml
```

For the v2 re-run, change it to:

```bash
--mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml
```

And update output paths so v1 and v2 results do not collide, e.g.:

```bash
--output outputs/ablations/v25_true_gt_v2_stability_a800.pth
> outputs/ablations/v25_true_gt_v2_stability_a800.log
```

Recommended convention: append `_v2` to the existing output name.

Evaluation scripts also need the v2 paths.  Each `eval_v*.py` defaults to `data/h36m_true_gt/`; pass:

```bash
--s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
--s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
```

## 5. Per-model re-run checklist

The sections below give the **exact A800 command** for each model.  All commands are written for GPU 7; swap to GPU 6 when GPU 7 is busy.  All commands assume the A800 working directory `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20` and the project venv at `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python`.

### 5.1 v25 stability (baseline)

**Source script:** `scripts/run_v25_ablation_true_gt_stability_a800.sh`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \
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
    --num_workers 4 \
    --d 128 --residual_hidden 256 --n_st_layers 3 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 20 --batch_size 16 --train_samples 4096 --val_stride 20 \
    --lr 1e-4 --lr_cosine --lr_warmup_epochs 4 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --weight_decay 1e-4 \
    --early_stopping_patience 3 --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v25_true_gt_v2_stability_a800.pth \
    > outputs/ablations/v25_true_gt_v2_stability_a800.log 2>&1
```

**Test eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
python scripts/eval_v25_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v25_true_gt_v2_stability_a800.pth \
    --config_json outputs/ablations/v25_true_gt_v2_stability_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 1 \
    --out_json outputs/eval_v25_true_gt_v2_stability_h36m_test.json
```

**Sparse-view no-fallback eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/ablations/v25_true_gt_v2_stability_a800.pth \
    --config outputs/ablations/v25_true_gt_v2_stability_a800.config.json \
    --dataset_manifest tmp/h36m_true_gt_v2_val_manifest.txt \
    --clip_len 13 --min_views 2 --max_views 4 --num_subsets_per_k 50 --seed 42 \
    --output_csv outputs/variable_view_v25_true_gt_v2_stability_a800.csv \
    --output_json outputs/variable_view_v25_true_gt_v2_stability_a800.json
```

**Sparse-view DLT-fallback eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/ablations/v25_true_gt_v2_stability_a800.pth \
    --config outputs/ablations/v25_true_gt_v2_stability_a800.config.json \
    --dataset_manifest tmp/h36m_true_gt_v2_val_manifest.txt \
    --clip_len 13 --min_views 2 --max_views 4 --num_subsets_per_k 50 --seed 42 \
    --var_view_dlt_fallback \
    --output_csv outputs/variable_view_fix/variable_view_v25_true_gt_v2_stability_a800_dlt_fallback.csv \
    --output_json outputs/variable_view_fix/variable_view_v25_true_gt_v2_stability_a800_dlt_fallback.json
```

- [ ] v25 v2 training finished.
- [ ] v25 v2 S9/S11 test MPJPE/PA-MPJPE recorded.
- [ ] v25 v2 variable-view k=2/3/4 MPJPE recorded (with and without DLT fallback).

### 5.2 v46 sparse-view generalization (SVG)

**Source script:** `scripts/run_v46_true_gt_h36m_a800.sh`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_v45_adaptive_geometry_fusion \
    --v45_adaptive_weight_type per_view_joint \
    --v45_adaptive_weight_hidden 32 \
    --v45_adaptive_weight_n_layers 1 \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 \
    --v46_svg_min_views 2 \
    --v46_svg_hidden 64 \
    --v46_svg_use_curriculum \
    --use_hierarchical_multiview_v30 \
    --v30_n_part_layers 2 \
    --v30_stochastic_depth_prob 0.1 \
    --use_physical_space_temporal_loss_v29 \
    --v29_floor_loss_weight 0.01 \
    --v29_bone_temporal_weight 0.01 \
    --v29_com_jitter_weight 0.001 \
    --v29_physical_loss_warmup_epochs 1 \
    --num_workers 0 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 13 \
    --epochs 8 \
    --batch_size 16 \
    --train_samples 1024 \
    --val_stride 20 \
    --lr 1e-3 \
    --lr_cosine \
    --lr_warmup_epochs 1 \
    --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --ema_decay 0.999 \
    --early_stopping_patience 3 \
    --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true \
    --use_camera_conditioning true \
    --use_epipolar_bias true \
    --use_context_visibility true \
    --use_skeleton_residual true \
    --use_rotation_correction true \
    --use_entropy_regularization true \
    --attention_entropy_weight 0.01 \
    --use_camera_view_embedding \
    --use_set_view_aggregator \
    --use_variable_view_training \
    --variable_view_min_views 2 \
    --variable_view_max_views 4 \
    --variable_view_max_views_start 4 \
    --variable_view_curriculum_alpha 2.0 \
    --variable_view_permute \
    --pa_loss_weight 0.5 \
    --monotonic_loss_weight 0.1 \
    --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 \
    --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 \
    --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 \
    --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v46_true_gt_v2_h36m_a800.pth \
    > outputs/ablations/v46_true_gt_v2_h36m_a800.log 2>&1
```

**Test eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
python scripts/eval_v46_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v46_true_gt_v2_h36m_a800.pth \
    --config_json outputs/ablations/v46_true_gt_v2_h36m_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v46_true_gt_v2_h36m_test_a800.json
```

- [ ] v46 v2 training finished.
- [ ] v46 v2 S9/S11 test MPJPE/PA-MPJPE recorded.

### 5.3 v52 uncertainty-weighted triangulation (UWT)

**Source script:** `scripts/run_v52_true_gt_h36m_a800.sh`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_v45_adaptive_geometry_fusion \
    --v45_adaptive_weight_type per_view_joint \
    --v45_adaptive_weight_hidden 32 \
    --v45_adaptive_weight_n_layers 1 \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 \
    --v46_svg_min_views 2 \
    --v46_svg_hidden 64 \
    --v46_svg_use_curriculum \
    --use_v50_self_evolution_feedback_head \
    --v50_sefh_hidden 64 \
    --v50_sefh_num_layers 2 \
    --v50_sefh_dropout 0.1 \
    --v50_sefh_loss_weight 0.0 \
    --v50_sefh_aleatoric_weight 0.0 \
    --v50_sefh_identity_init_gate \
    --use_v51_cross_domain_sparse_view_reliability \
    --v51_cdsvr_hidden 64 \
    --v51_cdsvr_num_heads 4 \
    --v51_cdsvr_dropout 0.1 \
    --v51_cdsvr_offset_min 0.05 \
    --v51_cdsvr_use_domain_label \
    --v51_cdsvr_uncertainty_temperature 1.0 \
    --v51_cdsvr_identity_init_gate \
    --v51_cdsvr_loss_weight 0.0 \
    --use_v52_uncertainty_weighted_triangulation \
    --v52_uwt_hidden 64 \
    --v52_uwt_n_layers 2 \
    --v52_uwt_weight_type per_view_joint \
    --v52_uwt_temperature 1.0 \
    --v52_uwt_use_geometry_bias \
    --v52_uwt_use_feature_bias \
    --v52_uwt_identity_init \
    --v52_uwt_min_weight 0.05 \
    --v52_uwt_loss_weight 0.01 \
    --v52_uwt_damping 1e-4 \
    --num_workers 4 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 13 \
    --epochs 8 \
    --batch_size 16 \
    --train_samples 1024 \
    --val_stride 20 \
    --lr 1e-3 \
    --lr_cosine \
    --lr_warmup_epochs 1 \
    --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --ema_decay 0.999 \
    --early_stopping_patience 3 \
    --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true \
    --use_camera_conditioning true \
    --use_epipolar_bias true \
    --use_context_visibility true \
    --use_skeleton_residual true \
    --use_rotation_correction true \
    --use_entropy_regularization true \
    --attention_entropy_weight 0.01 \
    --use_camera_view_embedding \
    --use_set_view_aggregator \
    --use_hierarchical_multiview_v30 \
    --v30_n_part_layers 2 \
    --v30_stochastic_depth_prob 0.1 \
    --use_physical_space_temporal_loss_v29 \
    --v29_floor_loss_weight 0.01 \
    --v29_bone_temporal_weight 0.01 \
    --v29_com_jitter_weight 0.001 \
    --v29_physical_loss_warmup_epochs 1 \
    --use_variable_view_training \
    --variable_view_min_views 2 \
    --variable_view_max_views 4 \
    --variable_view_max_views_start 4 \
    --variable_view_curriculum_alpha 2.0 \
    --variable_view_permute \
    --pa_loss_weight 0.5 \
    --monotonic_loss_weight 0.1 \
    --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 \
    --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 \
    --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 \
    --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v52_true_gt_v2_h36m_a800.pth \
    > outputs/ablations/v52_true_gt_v2_h36m_a800.log 2>&1
```

**Test eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
python scripts/eval_v52_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v52_true_gt_v2_h36m_a800.pth \
    --config_json outputs/ablations/v52_true_gt_v2_h36m_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v52_true_gt_v2_h36m_test_a800.json
```

- [ ] v52 v2 training finished.
- [ ] v52 v2 S9/S11 test MPJPE/PA-MPJPE recorded.

### 5.4 v57 domain-conditional physical-space calibration (DC-PSC)

**Source script:** `scripts/run_v57_true_gt_medium_a800.sh`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_v45_adaptive_geometry_fusion \
    --v45_adaptive_weight_type per_view_joint \
    --v45_adaptive_weight_hidden 32 \
    --v45_adaptive_weight_n_layers 1 \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 \
    --v46_svg_min_views 2 \
    --v46_svg_hidden 64 \
    --v46_svg_use_curriculum \
    --use_v50_self_evolution_feedback_head \
    --v50_sefh_hidden 64 \
    --v50_sefh_num_layers 2 \
    --v50_sefh_dropout 0.1 \
    --v50_sefh_loss_weight 0.0 \
    --v50_sefh_aleatoric_weight 0.0 \
    --v50_sefh_identity_init_gate \
    --use_v51_cross_domain_sparse_view_reliability \
    --v51_cdsvr_hidden 64 \
    --v51_cdsvr_num_heads 4 \
    --v51_cdsvr_dropout 0.1 \
    --v51_cdsvr_offset_min 0.05 \
    --v51_cdsvr_use_domain_label \
    --v51_cdsvr_uncertainty_temperature 1.0 \
    --v51_cdsvr_identity_init_gate \
    --v51_cdsvr_loss_weight 0.0 \
    --use_v52_uncertainty_weighted_triangulation \
    --v52_uwt_hidden 64 \
    --v52_uwt_n_layers 2 \
    --v52_uwt_weight_type per_view_joint \
    --v52_uwt_temperature 1.0 \
    --v52_uwt_use_geometry_bias \
    --v52_uwt_use_feature_bias \
    --v52_uwt_identity_init \
    --v52_uwt_min_weight 0.05 \
    --v52_uwt_loss_weight 0.01 \
    --v52_uwt_damping 1e-4 \
    --use_v57_domain_conditional_psc \
    --v57_dcpsc_hidden 64 \
    --v57_dcpsc_n_layers 2 \
    --v57_dcpsc_num_domains 8 \
    --v57_dcpsc_use_floor \
    --v57_dcpsc_use_bone_scale \
    --v57_dcpsc_use_uwt_weights \
    --v57_dcpsc_identity_init \
    --v57_dcpsc_residual_gate_init -6.0 \
    --v57_dcpsc_loss_weight 0.1 \
    --v57_dcpsc_floor_weight 0.01 \
    --v57_dcpsc_bone_weight 0.1 \
    --v57_dcpsc_reproj_weight 0.1 \
    --v57_dcpsc_warmup_epochs 1 \
    --v57_dcpsc_min_visible_views 2 \
    --num_workers 0 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 13 \
    --epochs 8 \
    --batch_size 16 \
    --train_samples 1024 \
    --val_stride 20 \
    --lr 1e-3 \
    --lr_cosine \
    --lr_warmup_epochs 1 \
    --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --ema_decay 0.999 \
    --early_stopping_patience 3 \
    --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true \
    --use_camera_conditioning true \
    --use_epipolar_bias true \
    --use_context_visibility true \
    --use_skeleton_residual true \
    --use_rotation_correction true \
    --use_entropy_regularization true \
    --attention_entropy_weight 0.01 \
    --use_camera_view_embedding \
    --use_set_view_aggregator \
    --use_hierarchical_multiview_v30 \
    --v30_n_part_layers 2 \
    --v30_stochastic_depth_prob 0.1 \
    --use_physical_space_temporal_loss_v29 \
    --v29_floor_loss_weight 0.01 \
    --v29_bone_temporal_weight 0.01 \
    --v29_com_jitter_weight 0.001 \
    --v29_physical_loss_warmup_epochs 1 \
    --use_variable_view_training \
    --variable_view_min_views 2 \
    --variable_view_max_views 4 \
    --variable_view_max_views_start 4 \
    --variable_view_curriculum_alpha 2.0 \
    --variable_view_permute \
    --pa_loss_weight 0.5 \
    --monotonic_loss_weight 0.1 \
    --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 \
    --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 \
    --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 \
    --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v57_true_gt_v2_medium_a800.pth \
    > outputs/ablations/v57_true_gt_v2_medium_a800.log 2>&1
```

**Test eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
python scripts/eval_v57_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v57_true_gt_v2_medium_a800.pth \
    --config_json outputs/ablations/v57_true_gt_v2_medium_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v57_true_gt_v2_h36m_test_a800.json
```

- [ ] v57 v2 training finished.
- [ ] v57 v2 S9/S11 test MPJPE/PA-MPJPE recorded.

### 5.5 v80 view-reliability weighting

**Source script:** `scripts/run_v80_ablation_true_gt_regularization_a800.sh`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.05 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_v45_adaptive_geometry_fusion \
    --v45_adaptive_weight_type per_view_joint \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 \
    --v46_svg_use_curriculum \
    --use_v50_self_evolution_feedback_head \
    --v50_sefh_hidden 64 --v50_sefh_num_layers 2 --v50_sefh_dropout 0.1 \
    --v50_sefh_identity_init_gate \
    --v50_sefh_loss_weight 0.0 --v50_sefh_aleatoric_weight 0.0 \
    --use_v51_cross_domain_sparse_view_reliability \
    --v51_cdsvr_hidden 64 --v51_cdsvr_num_heads 4 --v51_cdsvr_dropout 0.1 \
    --v51_cdsvr_use_domain_label --v51_cdsvr_identity_init_gate \
    --v51_cdsvr_loss_weight 0.0 \
    --use_v52_uncertainty_weighted_triangulation \
    --v52_uwt_hidden 64 --v52_uwt_n_layers 2 --v52_uwt_weight_type per_view_joint \
    --v52_uwt_use_geometry_bias --v52_uwt_use_feature_bias \
    --v52_uwt_identity_init --v52_uwt_min_weight 0.05 --v52_uwt_loss_weight 0.01 \
    --v52_uwt_damping 0.0001 \
    --use_v80_view_reliability \
    --v80_vrbt_hidden 64 --v80_vrbt_n_layers 2 \
    --v80_vrbt_weight_type per_view_joint \
    --v80_vrbt_use_geometry_bias --v80_vrbt_use_feature_bias \
    --v80_vrbt_identity_init --v80_vrbt_min_weight 0.05 \
    --bone_loss_weight 0.05 \
    --joint_limit_weight 0.01 \
    --temporal_bone_weight 0.005 \
    --num_workers 4 \
    --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 20 --batch_size 16 --train_samples 4096 --val_stride 20 \
    --lr 5e-4 --lr_cosine --lr_warmup_epochs 2 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --weight_decay 2e-4 \
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
    --output outputs/ablations/v80_true_gt_v2_regularization_a800.pth \
    > outputs/ablations/v80_true_gt_v2_regularization_a800.log 2>&1
```

**Test eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
python scripts/eval_v80_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v80_true_gt_v2_regularization_a800.pth \
    --config_json outputs/ablations/v80_true_gt_v2_regularization_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v80_true_gt_v2_h36m_test_a800.json
```

- [ ] v80 v2 training finished.
- [ ] v80 v2 S9/S11 test MPJPE/PA-MPJPE recorded.

### 5.6 v81 temporal-pose-attention

**Source script:** `scripts/run_v81_true_gt_h36m_medium_a800.sh`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \
    --num_domains 1 \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_temporal_pose_attention_v81 \
    --v81_temporal_window 9 \
    --v81_temporal_residual_gate_init -6.0 \
    --v81_temporal_dropout 0.1 \
    --num_workers 4 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 13 \
    --epochs 8 \
    --batch_size 16 \
    --train_samples 1024 \
    --val_stride 20 \
    --lr 1e-3 \
    --lr_cosine \
    --lr_warmup_epochs 1 \
    --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --ema_decay 0.999 \
    --early_stopping_patience 3 \
    --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true \
    --use_camera_conditioning true \
    --use_epipolar_bias true \
    --use_context_visibility true \
    --use_skeleton_residual true \
    --use_rotation_correction true \
    --use_entropy_regularization true \
    --attention_entropy_weight 0.01 \
    --use_camera_view_embedding \
    --use_set_view_aggregator \
    --use_variable_view_training \
    --variable_view_min_views 2 \
    --variable_view_max_views 4 \
    --variable_view_max_views_start 4 \
    --variable_view_curriculum_alpha 2.0 \
    --variable_view_permute \
    --pa_loss_weight 0.5 \
    --monotonic_loss_weight 0.1 \
    --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 \
    --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 \
    --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 \
    --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v81_true_gt_v2_h36m_medium_a800.pth \
    > outputs/ablations/v81_true_gt_v2_h36m_medium_a800.log 2>&1
```

**Test eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
python scripts/eval_v81_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v81_true_gt_v2_h36m_medium_a800.pth \
    --config_json outputs/ablations/v81_true_gt_v2_h36m_medium_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v81_true_gt_v2_h36m_test_a800.json
```

**Sparse-view DLT-fallback eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/ablations/v81_true_gt_v2_h36m_medium_a800.pth \
    --config outputs/ablations/v81_true_gt_v2_h36m_medium_a800.config.json \
    --dataset_manifest tmp/h36m_true_gt_v2_val_manifest.txt \
    --clip_len 13 --min_views 2 --max_views 4 --num_subsets_per_k 50 --seed 42 \
    --var_view_dlt_fallback \
    --output_csv outputs/variable_view_fix/variable_view_v81_true_gt_v2_medium_a800_dlt_fallback.csv \
    --output_json outputs/variable_view_fix/variable_view_v81_true_gt_v2_medium_a800_dlt_fallback.json
```

- [ ] v81 v2 training finished.
- [ ] v81 v2 S9/S11 test MPJPE/PA-MPJPE recorded.
- [ ] v81 v2 DLT-fallback variable-view k=2/3 MPJPE recorded.

### 5.7 v82 multi-scale temporal-pose-attention

v82 does not have a dedicated A800 training script.  Use the v81 A800 recipe and replace the v81 temporal module with the v82 multi-scale variant (`--use_temporal_pose_attention_v82`, `--v82_temporal_windows`).

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \
    --num_domains 1 \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_temporal_pose_attention_v82 \
    --v82_temporal_windows 5 13 -1 \
    --v82_hidden_dim 16 \
    --v82_temporal_residual_gate_init -6.0 \
    --v82_temporal_dropout 0.1 \
    --num_workers 4 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 13 \
    --epochs 8 \
    --batch_size 16 \
    --train_samples 1024 \
    --val_stride 20 \
    --lr 1e-3 \
    --lr_cosine \
    --lr_warmup_epochs 1 \
    --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --ema_decay 0.999 \
    --early_stopping_patience 3 \
    --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true \
    --use_camera_conditioning true \
    --use_epipolar_bias true \
    --use_context_visibility true \
    --use_skeleton_residual true \
    --use_rotation_correction true \
    --use_entropy_regularization true \
    --attention_entropy_weight 0.01 \
    --use_camera_view_embedding \
    --use_set_view_aggregator \
    --use_variable_view_training \
    --variable_view_min_views 2 \
    --variable_view_max_views 4 \
    --variable_view_max_views_start 4 \
    --variable_view_curriculum_alpha 2.0 \
    --variable_view_permute \
    --pa_loss_weight 0.5 \
    --monotonic_loss_weight 0.1 \
    --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 \
    --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 \
    --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 \
    --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v82_true_gt_v2_h36m_medium_a800.pth \
    > outputs/ablations/v82_true_gt_v2_h36m_medium_a800.log 2>&1
```

**Test eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
python scripts/eval_v81_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v82_true_gt_v2_h36m_medium_a800.pth \
    --config_json outputs/ablations/v82_true_gt_v2_h36m_medium_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v82_true_gt_v2_h36m_test_a800.json
```

**Sparse-view DLT-fallback eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/ablations/v82_true_gt_v2_h36m_medium_a800.pth \
    --config outputs/ablations/v82_true_gt_v2_h36m_medium_a800.config.json \
    --dataset_manifest tmp/h36m_true_gt_v2_val_manifest.txt \
    --clip_len 13 --min_views 2 --max_views 4 --num_subsets_per_k 50 --seed 42 \
    --var_view_dlt_fallback \
    --output_csv outputs/variable_view_fix/variable_view_v82_true_gt_v2_medium_a800_dlt_fallback.csv \
    --output_json outputs/variable_view_fix/variable_view_v82_true_gt_v2_medium_a800_dlt_fallback.json
```

- [ ] v82 v2 training script created / command recorded.
- [ ] v82 v2 training finished.
- [ ] v82 v2 S9/S11 test MPJPE/PA-MPJPE recorded.
- [ ] v82 v2 DLT-fallback variable-view k=2/3/4 MPJPE recorded.

### 5.8 v85 random view dropout (sparse-view robustness)

**Source script:** `scripts/run_v85_random_view_dropout_medium_a800.sh`

v85 is currently training on GPU 7 against the v1 manifest.  Do not touch that job.  Once it finishes, launch the identical recipe against the v2 split.

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \
    --num_domains 1 \
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
    --use_random_view_dropout_v85 \
    --v85_dropout_prob 0.3 \
    --v85_min_views 2 \
    --v85_use_count_embedding \
    --num_workers 4 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 13 \
    --epochs 20 \
    --batch_size 16 \
    --train_samples 4096 \
    --val_stride 20 \
    --lr 1e-4 \
    --lr_cosine \
    --lr_warmup_epochs 4 \
    --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --ema_decay 0.999 \
    --weight_decay 1e-4 \
    --early_stopping_patience 3 \
    --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true \
    --use_camera_conditioning true \
    --use_epipolar_bias true \
    --use_context_visibility true \
    --use_skeleton_residual true \
    --use_rotation_correction true \
    --use_entropy_regularization true \
    --attention_entropy_weight 0.01 \
    --use_camera_view_embedding \
    --use_set_view_aggregator \
    --use_variable_view_training \
    --variable_view_min_views 2 \
    --variable_view_max_views 4 \
    --variable_view_max_views_start 4 \
    --variable_view_curriculum_alpha 2.0 \
    --pa_loss_weight 0.5 \
    --monotonic_loss_weight 0.1 \
    --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 \
    --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 \
    --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 \
    --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v85_random_view_dropout_v2_medium_a800.pth \
    > outputs/ablations/v85_random_view_dropout_v2_medium_a800.log 2>&1
```

**Test eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
python scripts/eval_v85_random_view_dropout_h36m_test.py \
    --checkpoint outputs/ablations/v85_random_view_dropout_v2_medium_a800.pth \
    --config_json outputs/ablations/v85_random_view_dropout_v2_medium_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v85_true_gt_v2_h36m_test_a800.json
```

**Sparse-view no-fallback eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/ablations/v85_random_view_dropout_v2_medium_a800.pth \
    --config outputs/ablations/v85_random_view_dropout_v2_medium_a800.config.json \
    --dataset_manifest tmp/h36m_true_gt_v2_val_manifest.txt \
    --clip_len 13 --min_views 2 --max_views 4 --num_subsets_per_k 50 --seed 42 \
    --output_csv outputs/variable_view_v85_random_view_dropout_v2_medium_a800.csv \
    --output_json outputs/variable_view_v85_random_view_dropout_v2_medium_a800.json
```

**Sparse-view DLT-fallback eval:**
```bash
CUDA_VISIBLE_DEVICES=7 \
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/ablations/v85_random_view_dropout_v2_medium_a800.pth \
    --config outputs/ablations/v85_random_view_dropout_v2_medium_a800.config.json \
    --dataset_manifest tmp/h36m_true_gt_v2_val_manifest.txt \
    --clip_len 13 --min_views 2 --max_views 4 --num_subsets_per_k 50 --seed 42 \
    --var_view_dlt_fallback \
    --output_csv outputs/variable_view_fix/variable_view_v85_random_view_dropout_v2_medium_a800_dlt_fallback.csv \
    --output_json outputs/variable_view_fix/variable_view_v85_random_view_dropout_v2_medium_a800_dlt_fallback.json
```

- [ ] v85 v2 training launched **after** the current v85 v1 job finishes.
- [ ] v85 v2 full-view test MPJPE/PA-MPJPE recorded.
- [ ] v85 v2 no-fallback variable-view k=2/3/4 MPJPE recorded.
- [ ] v85 v2 DLT-fallback variable-view k=2/3/4 MPJPE recorded.

## 6. Result aggregation

After each model finishes, update `docs/results_true_gt_h36m.md`:

- Replace the *historical true-GT v1* table with the v2 table.
- Keep v1 numbers in a clearly marked “Historical v1 (misaligned) results” section for reference.
- Add a new section “True-GT v2 Leaderboard” with the v2 numbers.

Suggested table columns:

| Method | S9 direct (mm) | S11 direct (mm) | Combined direct (mm) | Combined PA-MPJPE (mm) | Source |
|---|---:|---:|---:|---:|---|
| Iskakov ICCV 2019 | — | — | — | — | — |
| DLT (conf-weighted) | — | — | — | — | `outputs/h36m_true_gt_v2_baselines/dlt_baseline_h36m_true_gt_v2.json` |
| RANSAC/conf-DLT | — | — | — | — | `outputs/h36m_true_gt_v2_baselines/ransac_baseline_h36m_true_gt_v2.json` |
| v25 stability | — | — | — | — | `outputs/eval_v25_true_gt_v2_stability_h36m_test.json` |
| v46 SVG | — | — | — | — | `outputs/eval_v46_true_gt_v2_h36m_test_a800.json` |
| v52 UWT | — | — | — | — | `outputs/eval_v52_true_gt_v2_h36m_test_a800.json` |
| v57 DC-PSC | — | — | — | — | `outputs/eval_v57_true_gt_v2_h36m_test_a800.json` |
| v80 regularization | — | — | — | — | `outputs/eval_v80_true_gt_v2_h36m_test_a800.json` |
| v81 temporal-pose-attention | — | — | — | — | `outputs/eval_v81_true_gt_v2_h36m_test_a800.json` |
| v82 multi-scale temporal-pose-attention | — | — | — | — | `outputs/eval_v82_true_gt_v2_h36m_test_a800.json` |
| v85 random view dropout | — | — | — | — | `outputs/eval_v85_true_gt_v2_h36m_test_a800.json` |

- [ ] `docs/results_true_gt_h36m.md` v2 table populated.
- [ ] Sparse-view v2 tables added for v25/v81/v82/v85.

## 7. Risks and watch-outs

| Risk | Mitigation |
|---|---|
| v85 currently occupies GPU 7 and v86 occupies GPU 6. | Queue the v2 re-runs; do not stop the current jobs. |
| A800 disk is ~99 % full. | Run `scripts/cleanup_a800_safe.sh` dry-run before each new run; delete failed/abandoned runs first. |
| v82 has no A800 training script. | Use the exact v82 command in Section 5.7. |
| Eval scripts default to v1 `.npz` paths. | Always pass the explicit `--s9` / `--s11` v2 paths or edit the script defaults. |
| Training scripts default to v1 manifest. | Always swap to `configs/splits/h36m_true_gt_v2_standard.yaml`. |
| Variable-view manifest defaults to v1 paths. | Create and use `tmp/h36m_true_gt_v2_val_manifest.txt`. |
| Re-running 8 medium A800 jobs is a long queue. | Prioritise v25, v85, v81, v82 (best current models); v46/v52/v57/v80 can follow. |

## 8. Quick command summary

```bash
# 1. Generate v2 labels (local)
bash scripts/convert_all_h36m_true_gt_v2.sh
bash scripts/sync_h36m_true_gt_v2_to_a800.sh

# 2. Create v2 variable-view manifest
cat > tmp/h36m_true_gt_v2_val_manifest.txt <<'EOF'
S9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
S11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
EOF

# 3. Baselines
bash scripts/run_h36m_true_gt_v2_baselines.sh

# 4. Training (queue on free GPU 6/7; full commands in Section 5)
# v25
CUDA_VISIBLE_DEVICES=7 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.05 --v25_dropout 0.2 --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment \
    --num_workers 4 --d 128 --residual_hidden 256 --n_st_layers 3 --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 20 --batch_size 16 --train_samples 4096 --val_stride 20 \
    --lr 1e-4 --lr_cosine --lr_warmup_epochs 4 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 --weight_decay 1e-4 \
    --early_stopping_patience 3 --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v25_true_gt_v2_stability_a800.pth \
    > outputs/ablations/v25_true_gt_v2_stability_a800.log 2>&1

# v46 / v52 / v57 / v80 / v81 / v82 / v85: see Section 5 for full exact commands.

# 5. Test eval (replace checkpoint/config paths with v2 variants)
python scripts/eval_v25_true_gt_h36m_test.py --checkpoint outputs/ablations/v25_true_gt_v2_stability_a800.pth --config_json outputs/ablations/v25_true_gt_v2_stability_a800.config.json --s9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --val_stride 1 --out_json outputs/eval_v25_true_gt_v2_stability_h36m_test.json

python scripts/eval_v46_true_gt_h36m_test.py --checkpoint outputs/ablations/v46_true_gt_v2_h36m_a800.pth --config_json outputs/ablations/v46_true_gt_v2_h36m_a800.config.json --s9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --val_stride 13 --out_json outputs/eval_v46_true_gt_v2_h36m_test_a800.json

python scripts/eval_v52_true_gt_h36m_test.py --checkpoint outputs/ablations/v52_true_gt_v2_h36m_a800.pth --config_json outputs/ablations/v52_true_gt_v2_h36m_a800.config.json --s9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --val_stride 13 --out_json outputs/eval_v52_true_gt_v2_h36m_test_a800.json

python scripts/eval_v57_true_gt_h36m_test.py --checkpoint outputs/ablations/v57_true_gt_v2_medium_a800.pth --config_json outputs/ablations/v57_true_gt_v2_medium_a800.config.json --s9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --val_stride 13 --out_json outputs/eval_v57_true_gt_v2_h36m_test_a800.json

python scripts/eval_v80_true_gt_h36m_test.py --checkpoint outputs/ablations/v80_true_gt_v2_regularization_a800.pth --config_json outputs/ablations/v80_true_gt_v2_regularization_a800.config.json --s9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --val_stride 13 --out_json outputs/eval_v80_true_gt_v2_h36m_test_a800.json

python scripts/eval_v81_true_gt_h36m_test.py --checkpoint outputs/ablations/v81_true_gt_v2_h36m_medium_a800.pth --config_json outputs/ablations/v81_true_gt_v2_h36m_medium_a800.config.json --s9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --val_stride 13 --out_json outputs/eval_v81_true_gt_v2_h36m_test_a800.json

# v82 uses the v81 eval helper (same architecture family)
python scripts/eval_v81_true_gt_h36m_test.py --checkpoint outputs/ablations/v82_true_gt_v2_h36m_medium_a800.pth --config_json outputs/ablations/v82_true_gt_v2_h36m_medium_a800.config.json --s9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --val_stride 13 --out_json outputs/eval_v82_true_gt_v2_h36m_test_a800.json

python scripts/eval_v85_random_view_dropout_h36m_test.py --checkpoint outputs/ablations/v85_random_view_dropout_v2_medium_a800.pth --config_json outputs/ablations/v85_random_view_dropout_v2_medium_a800.config.json --s9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz --val_stride 13 --out_json outputs/eval_v85_true_gt_v2_h36m_test_a800.json

# 6. Update docs/results_true_gt_h36m.md with v2 numbers
```

## 9. Definition of done

- [ ] All v2 `.npz` files generated, synced, and audited.
- [ ] DLT/RANSAC/Iskakov v2 baselines re-run and recorded.
- [ ] v25, v46, v52, v57, v80, v81, v82, v85 each trained once on `configs/splits/h36m_true_gt_v2_standard.yaml`.
- [ ] Each trained model evaluated on v2 S9/S11 test data.
- [ ] Sparse-view / DLT-fallback evals completed for v25, v81, v82, v85 on v2.
- [ ] `docs/results_true_gt_h36m.md` updated with a v2 leaderboard table.
- [ ] No running A800 job was interrupted or modified.
