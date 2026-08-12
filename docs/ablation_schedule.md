# v25 H36M True-GT Ablation Schedule

**Date:** 2026-08-11 (updated)  
**Status:** Ablations 1 and 2 completed on A800. Both reached best val MPJPE ~46.5 mm at epoch 1, then diverged and early-stopped at epoch 4 (final val 323.35 mm / 281.22 mm). Per decision criteria, **Ablation 3 (mixed dataset)** is now the next step. GPU-local RTX 4090 reserved for quick smoke / verification only.
**Applies to:** A800 GPUs 4/6 (read-only inspection of `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`) plus local WSL for prep and smoke.
**Related docs:**
- `docs/v25_divergence_diagnosis.md` — root-cause analysis
- `docs/v25_ablation_plan.md` — detailed ablation definitions
- `docs/results_true_gt_h36m.md` — leaderboard for recording results

## 1. Goal

Order and resource-plan the v25 true-GT H36M ablations and related follow-up runs on A800 GPUs. The local RTX 4090 is no longer used for medium/long training; it is reserved for smoke tests under 30 min. All A800 runs are read-only from the local repo perspective: prepare configs/scripts here, then launch only after confirming GPU availability on A800.

Baseline reference (H36M true-GT, S9/S11 direct):

| Run | Best val/test MPJPE | Notes |
|-----|--------------------:|-------|
| Iskakov ICCV 2019 | **23.35 mm** | Current true-GT leader |
| Conf-weighted DLT | **25.67 mm** | Strong baseline |
| RANSAC/conf-DLT | **26.47 mm** | Reproducible robust baseline |
| v80 medium | **39.98 mm** @ epoch 4 | Stronger regularisation, then overfit to 133.71 mm |
| v25 test | **43.93 mm** | Val inflated due to missing `view_mask` (fixed) |
| v57 final | **78.76 mm**; true best 75.16 mm @ epoch 3 not saved | Trainer checkpoint bug now fixed; best saved by `mpjpe` |

## 2. A800 queue and trigger conditions

Run the following jobs on A800, never exceeding the GPUs actually available. GPU 4 and GPU 6 are the current training slots. CPU-only preparation (configs, scripts, dry-runs, log parsing) is allowed while GPUs are busy.

| # | Name | GPU | Status | Trigger condition / launch rule |
|---|------|-----|--------|---------------------------------|
| 1 | **v25_true_gt_baseline_fix** | **GPU 4** | **DONE** — best val 46.53 mm @ epoch 1; final val 323.35 mm @ epoch 4 (early stopped). | Hyperparameter-only fix does not prevent divergence. |
| 2 | **v25_true_gt_geometry_regularization_a800** | **GPU 6** | **DONE** — best val 46.75 mm @ epoch 1; final val 281.22 mm @ epoch 4 (early stopped). | Geometry regularization does not prevent divergence; nearly identical to Abl. 1. |
| 3 | **v25_true_gt_mixed_dataset** | GPU 4 or 6 (whichever frees first) | **PENDING** | Launch only after **both** Abl. 1 and Abl. 2 have finished and been analyzed. Decision gate: if neither Abl. 1 nor Abl. 2 achieves ≤ 50 mm and stable val MPJPE, run mixed dataset; otherwise optional. |
| 4 | **v57_true_gt_rerun** | GPU 4 or 6 (whichever frees first) | **PENDING** | Launch after checkpoint bug fix is deployed and at least one GPU is free. Target: beat stale saved best (82.19 mm) and match the lost true best (75.16 mm @ epoch 3). Can run in parallel with Abl. 3 if two GPUs available. |
| 5 | **AIST++ medium (v25/v80/v57)** | GPU 4 or 6 | **PENDING** | Launch when a GPU is free and H36M true-GT baselines are stable. Lower priority than v57 re-run; can run concurrently with Abl. 3 or v57 re-run if GPUs allow. |

### Concurrency rules

- **Max two A800 training jobs at once** (one per GPU 4 and GPU 6).
- **No new A800 job launches until current GPU is free** (check `nvidia-smi` or A800 tmux pane).
- **Local RTX 4090:** single-GPU smoke only. Do not start a local medium/long run.
- A800-D projects/ and Docker services are **read-only / inspection-only**; never modify them from here.

## 3. Time and resource estimates

| Phase | GPU/CPU | A800 wall time | Notes |
|-------|---------|---------------:|-------|
| Abl. 1 — baseline fix | GPU 4 | **~4–6 h** | 20 epochs, `train_samples=4096/seq`, early stopping may cut it short |
| Abl. 2 — geometry regularization | GPU 6 | **~4–6 h** | Same as Abl. 1 plus bone / joint-limit / temporal-bone losses |
| Abl. 3 — mixed dataset | GPU 4/6 | **~6–10 h** | Larger effective training set (H36M + MPI), variable views up to 14 |
| v57 re-run | GPU 4/6 | **~4–6 h** | True best should now be saved thanks to `mpjpe` checkpointing |
| AIST++ medium | GPU 4/6 | **~6–10 h** | Full AIST++ medium for whichever variant is selected (v25/v80/v57) |
| Smoke / dry-run | CPU/GPU | **5–30 min** | Validate CLI, imports, paths, output dirs; local RTX 4090 only |
| Log parsing + table update | CPU | **10–15 min/run** | Extract best MPJPE, S9/S11 split, final MPJPE; update `docs/results_true_gt_h36m.md` |

## 4. Decision criteria

Primary metric: **best H36M true-GT val MPJPE** (S9+S11 combined direct). Secondary metric: final-epoch val MPJPE; a large rise indicates overfitting.

```text
Start
  │
  ▼
Ablation 1 (baseline fix) ── GPU 4
  │
  ├── Best ≤ 50 mm AND final ≤ best × 1.10? ──► Adopt Abl. 1 as new v25 true-GT recipe
  │
  ├── Best ≤ 72.80 mm but still unstable? ──► Consider Abl. 2 / Abl. 3
  │
  └── Best > 72.80 mm or diverges? ──► Investigate config/data; still run Abl. 2
        │
        ▼
Ablation 2 (geometry regularization) ── GPU 6
        │
        ├── Best ≤ 50 mm AND stable? ──► Adopt Abl. 2 (or lighter loss-weight variant)
        │
        ├── Better than Abl. 1 but still > 55 mm? ──► Run Abl. 3
        │
        └── No improvement vs. Abl. 1? ──► Run Abl. 3
                 │
                 ▼
          Ablation 3 (mixed dataset)
                 │
                 ├── Best ≤ 45 mm AND val MPJPE on H36M split is stable? ──► Adopt mixed-loader recipe
                 │
                 ├── Better than Abl. 1/2 but still > 55 mm? ──► Structural fixes required
                 │      (progressive unfreezing, bound residual_scale, lower v25_geom_loss_weight, AIST++ mix)
                 │
                 └── Worse than Abl. 1/2? ──► Mixed loader is not the right fix; revert to best ablation
                        and investigate data loader / domain embedding / MPI labels
```

### Numerical thresholds

| Decision | Condition |
|----------|-----------|
| **Adopt** ablation as recipe | Best val MPJPE ≤ 55 mm and final ≤ best × 1.10 |
| **Keep searching** | 55 mm < best ≤ 72.80 mm, or final > best × 1.20 |
| **Structural intervention** | Best > 72.80 mm in all three ablations, or any run NaNs/Inf |

### Recording requirements

After every ablation, record in `docs/results_true_gt_h36m.md`:

- Best val MPJPE and epoch.
- Final val MPJPE.
- S9 direct / S11 direct if available.
- Whether the best checkpoint was saved by `val_MPJPE`.
- Whether early stopping fired and, if so, at which epoch.

## 5. Pre-run checklist

- [ ] Confirm A800 GPU 4 is free before launching Abl. 1. (Currently running — no action.)
- [ ] Confirm A800 GPU 6 is free before launching Abl. 2. (Currently running — no action.)
- [ ] Confirm no other agent has started a GPU task by checking `tmux ls` or `nvidia-smi` on A800.
- [ ] Before launching Abl. 3: Abl. 1 and Abl. 2 are finished and analyzed.
- [ ] Before launching v57 re-run: trainer checkpoint bug fix is deployed on A800 branch and GPU is free.
- [ ] Before launching AIST++ medium: smoke manifests and DLT baseline are verified; GPU is free.
- [ ] `outputs/ablations/` exists on A800 (read-only from local; ensure via SSH if launching there).

## 6. Launch commands

> **Local repo:** prepare and store the commands below. Actual launch on A800 after GPU check.

### Ablation 1 — baseline fix (GPU 4)

```bash
nohup python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_standard.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
    --use_domain_embedding --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.05 --v25_dropout 0.2 \
    --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment \
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
    > outputs/ablations/v25_true_gt_baseline_fix.log 2>&1 &
```

### Ablation 2 — geometry regularization (GPU 6)

Same as Abl. 1 plus the three geometry-loss flags and a different output path.

```bash
nohup python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_standard.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
    --use_domain_embedding --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.05 --v25_dropout 0.2 \
    --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment \
    --bone_loss_weight 0.05 --joint_limit_weight 0.01 --temporal_bone_weight 0.005 \
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
    --output outputs/ablations/v25_true_gt_geometry_regularization.pth \
    > outputs/ablations/v25_true_gt_geometry_regularization.log 2>&1 &
```

### Ablation 3 — mixed dataset

Same as Abl. 1 but with the H36M+MPI mixed loader and `variable_view_max_views 14`.

```bash
nohup python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
    --use_domain_embedding --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.05 --v25_dropout 0.2 \
    --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment \
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
    > outputs/ablations/v25_true_gt_mixed_dataset.log 2>&1 &
```

### v57 re-run

Use the same v57 config as before, but ensure the trainer monitors `mpjpe` for best-checkpoint selection. Launch when a GPU is free and the checkpoint fix is confirmed on A800.

```bash
# Example path; replace with actual v57 config when launching
nohup python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --config configs/ablations/v57_true_gt_rerun.yaml \
    --output outputs/ablations/v57_true_gt_rerun.pth \
    > outputs/ablations/v57_true_gt_rerun.log 2>&1 &
```

### AIST++ medium

Run full AIST++ medium for the selected variant (v25/v80/v57) using the existing smoke manifests. Lower priority than H36M ablations and v57 re-run.

```bash
# Example; replace with the chosen variant's script/config
nohup python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --config configs/ablations/aist_v25_medium.yaml \
    --output outputs/ablations/aist_v25_medium.pth \
    > outputs/ablations/aist_v25_medium.log 2>&1 &
```

## 7. Monitoring

After launching any run on A800:

```bash
# A800 remote
tail -f outputs/ablations/v25_true_gt_<name>.log
nvidia-smi d -m 5  # or `watch -n 5 nvidia-smi`

# From local WSL (read-only tail via SSH)
ssh a800-D "tmux capture-pane -pt <session> -S -100"
```

Stop criteria:

- NaN/Inf in loss → stop immediately, investigate.
- Val MPJPE rises for 3 consecutive epochs → early stopping will handle it; no manual stop needed.
- GPU memory exceeds available → reduce batch size or `d` for that ablation.

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| GPU 4/6 still busy with Abl. 1/2 | Current | Wait; do CPU prep only; monitor `nvidia-smi` |
| Abl. 1/2 each take longer than 8 h | Medium | Early stopping should truncate; if not, reduce `train_samples` to 2048 or epochs to 12 |
| Mixed-loader MPI labels are not real-detected | Medium | Abl. 3 still tests data diversity; true MPI protocol is blocked until `imageSequence/` is available |
| v57 re-run still saves stale checkpoint | Low | Verify trainer monitors `mpjpe` before launching |
| Overfitting persists despite all three ablations | Medium | Fall back to structural fixes (progressive unfreezing, bound `residual_scale`, AIST++ mix) |
| Another agent launches a GPU job concurrently | Low | Confirm in handoff channel before each `nohup` |

## 9. Deliverables

- `docs/ablation_schedule.md` (this file)
- `outputs/ablations/v25_true_gt_baseline_fix.{log,pth}`
- `outputs/ablations/v25_true_gt_geometry_regularization.{log,pth}`
- `outputs/ablations/v25_true_gt_mixed_dataset.{log,pth}`
- `outputs/ablations/v57_true_gt_rerun.{log,pth}` (pending)
- `outputs/ablations/aist_<variant>_medium.{log,pth}` (pending)
- Updated `docs/results_true_gt_h36m.md` with the ablation results

## 10. Summary

1. **Abl. 1** (`v25_true_gt_baseline_fix`) is **running on A800 GPU 4**.
2. **Abl. 2** (`v25_true_gt_geometry_regularization_a800`) is **running on A800 GPU 6**.
3. **Abl. 3** (`v25_true_gt_mixed_dataset`) is **pending**: launch when GPU 4 or GPU 6 frees and Abl. 1/2 have been analyzed. Decision gate: run only if neither Abl. 1 nor Abl. 2 stabilises ≤ 50 mm.
4. **v57 re-run** is **pending**: launch after the checkpoint bug fix (best saved by `mpjpe`) is deployed and a GPU is free. Target: beat stale 82.19 mm and match the lost true best 75.16 mm.
5. **AIST++ medium** is **pending**: launch when a GPU is free and H36M true-GT baselines are stable. Lower priority than v57 re-run.
6. **Local RTX 4090** is reserved for quick smoke/verification only (< 30 min).
7. **Record** every result and update the leaderboard (`docs/results_true_gt_h36m.md`).
8. If none of the three H36M ablations work, move to the structural interventions in `docs/v25_divergence_diagnosis.md` Section 3.
