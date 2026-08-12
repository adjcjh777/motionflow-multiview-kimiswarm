# A800 `outputs/` Cleanup Analysis (read-only)

**Scope:** Inspect `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs` for safe-to-delete files. No files were deleted; this report only identifies candidates with sizes and reasoning.

## Disk situation

```text
Filesystem      Size  Used Avail Use%
/dev/nvme0n1p1  3.5T  3.3T   58G  99%

outputs/ total: 558 MB
ablations/:     522 MB
```

The `outputs/` directory is **not** the main disk consumer, but a focused cleanup can still recover ~70 MB of failed-run / smoke / duplicate data.

## Safe-to-delete candidates

### 1. Old smoke checkpoints directories (Aug 7) — ~30.2 MB

These are completed/legacy smoke runs from early August and contain both an intermediate `checkpoint.pt` and a final `checkpoint_final.pth`. Only the directories (not individual files) are listed for clarity.

| Path | Size |
|------|------|
| `outputs/mixed_smoke_v5/` | 3.7 MB |
| `outputs/mixed_smoke_v7/` | 3.7 MB |
| `outputs/domain_smoke/` | 4.4 MB |
| `outputs/curriculum_smoke/` | 4.5 MB |
| `outputs/curriculum_smoke2/` | 4.5 MB |
| `outputs/pa_smoke3/` | 4.5 MB |
| `outputs/v6_smoke_test/` | 4.5 MB |
| **Total** | **30.2 MB** |

**Why safe:** Smoke outputs from completed exploratory runs; not referenced by current experiments.

### 2. v83 failed ablation — ~20 KB

v83 plateaued at ~100 mm val MPJPE and was killed. No useful checkpoint exists.

| Path | Size | Reason |
|------|------|--------|
| `outputs/ablations/v83_true_gt_h36m_medium_a800.config.json` | 16 KB | Failed run config |
| `outputs/ablations/v83_true_gt_h36m_medium_a800.log` | 242 B | Failed run log |
| `outputs/ablations/v83_true_gt_h36m_medium_a800_nohup.log` | 0 B | Empty nohup |

### 3. v25 mixed-dataset failed/killed runs — ~64 KB

These are abandoned mixed-dataset attempts (one killed, one diverged). No final checkpoint is present.

| Path | Size | Reason |
|------|------|--------|
| `outputs/ablations/v25_true_gt_mixed_dataset_a800.config.json` | 14 KB | Abandoned run config |
| `outputs/ablations/v25_true_gt_mixed_dataset_a800_gpu5.config.json` | 14 KB | Abandoned run config |
| `outputs/ablations/v25_true_gt_mixed_dataset_a800_gpu5.log` | 35 B | Abandoned run log |
| `outputs/ablations/v25_true_gt_mixed_dataset_a800.log` | 12.6 KB | Abandoned run log |
| `outputs/ablations/v25_true_gt_mixed_dataset_a800_monitor_test_eval.log` | 330 B | Transient monitor log |
| `outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.config.json` | 15 KB | Killed run config |
| `outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.log` | 13 KB | Killed run log |

### 4. Duplicate / redundant logs in `sota_baselines/` — ~32 KB

Several nohup logs are byte-for-byte duplicates of their non-nohup counterparts or of each other.

| Path | Size | Reason |
|------|------|--------|
| `outputs/sota_baselines/monitor_v85_evalsuite_then_launch_voxelpose_nohup.log` | 9.0 KB | Identical to `monitor_v85_evalsuite_then_launch_voxelpose.log` |
| `outputs/sota_baselines/monitor_v85_then_run_evals_nohup.log` | 11.5 KB | Identical to `monitor_v85_then_run_evals.log` |
| `outputs/sota_baselines/voxelpose_h36m_true_gt_a800_train_nohup3.log` | 374 B | Identical to `...nohup2.log` and `...nohup4.log` |
| `outputs/sota_baselines/voxelpose_h36m_true_gt_a800_train_nohup4.log` | 374 B | Identical to `...nohup2.log` and `...nohup3.log` |

### 5. Duplicate Iskakov baseline result — ~28 KB

The GPU 4 Iskakov run duplicates the GPU 6 result (different file hashes, but the GPU 6 run is the canonical/latest one).

| Path | Size | Reason |
|------|------|--------|
| `outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_gpu4.config.json` | 5.2 KB | Duplicate of GPU 6 config |
| `outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_gpu4.log` | 6.8 KB | Duplicate of GPU 6 log |
| `outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_gpu4.pth` | 10.6 KB | Duplicate of GPU 6 checkpoint |

### 6. v85 redundant epoch-5 backup — 39.7 MB

| Path | Size | Reason |
|------|------|--------|
| `outputs/ablations/v85_random_view_dropout_medium_a800_backup_epoch5.pth` | 39.7 MB | Superseded by `v85_random_view_dropout_medium_a800_final.pth` (Aug 12 14:19). |

**Caution:** Do **not** delete `v85_random_view_dropout_medium_a800.pth` (best checkpoint) or `..._final.pth` while the post-training eval suite monitor (PID 2072251) may still need them. The `_backup_epoch5.pth` is safe to remove once v85 training/eval is fully complete.

### 7. Zero-byte logs (safe but no space)

The following files are empty and can be removed for tidiness, but they free no space:

- `outputs/ablations/aistpp_only_medium_a800_gpu5.log`
- `outputs/ablations/aistpp_only_medium_a800_gpu5_fast.log`
- `outputs/ablations/v25_true_gt_geometry_regularization_a800.nohup`
- `outputs/ablations/v46_true_gt_h36m_a800_gpu4_nohup.log`
- `outputs/ablations/v52_true_gt_h36m_a800_queued.log`
- `outputs/ablations/v57_aistpp_full_medium_a800_gpu6_nohup.log`
- `outputs/ablations/v57_true_gt_medium_a800_gpu5_nohup.log`
- `outputs/ablations/v80_true_gt_regularization_a800.nohup.out`
- `outputs/ablations/v82_true_gt_h36m_medium_a800_nohup.log`
- `outputs/ablations/v83_true_gt_h36m_medium_a800_nohup.log`
- `outputs/ablations/v85_random_view_dropout_medium_a800_nohup.log`
- `outputs/ablations/v85_random_view_dropout_medium_a800_restart2.log`
- `outputs/ablations/v25_true_gt_mixed_dataset_a800_monitor_nohup.log`
- `outputs/eval_v57_true_gt_h36m_test_local.log`
- `outputs/eval_v80_true_gt_h36m_test_local.log`
- `outputs/aistpp_fast_abs.log`
- `outputs/aistpp_fast_debug.log`
- `outputs/aistpp_fast_v2_nohup.log`
- `outputs/mixed_smoke_v5_debug.log`
- `outputs/webbridge_mixed_smoke.log`
- `outputs/variable_view_v81_true_gt_medium_a800_nohup.log`
- `outputs/variable_view_v85_random_view_dropout_medium_a800.log`
- `outputs/variable_view_v85_random_view_dropout_medium_a800_k2.log`
- `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.log`
- `outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_S11.log`
- `outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_S9.log`
- `outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_nohup.log`

## Files to keep (running jobs / canonical results)

- `outputs/ablations/v85_random_view_dropout_medium_a800.pth` — best checkpoint for running v85
- `outputs/ablations/v85_random_view_dropout_medium_a800_final.pth` — final v85 checkpoint
- `outputs/ablations/v86_no_count_embedding_medium_a800.*` — active v86 ablation
- All `variable_view_fix/*.json` / `*.csv` result files
- `outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_gpu6.*` — canonical latest baseline
- `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json` — canonical MPI baseline

## Estimated reclaimable space

| Category | Size |
|----------|------|
| Old smoke directories | ~30.2 MB |
| v83 failed run | ~20 KB |
| v25 mixed-dataset abandoned runs | ~64 KB |
| Duplicate sota_baselines logs | ~32 KB |
| Duplicate Iskakov GPU 4 baseline | ~28 KB |
| v85 epoch-5 backup (after v85 finishes) | ~39.7 MB |
| **Total** | **~70.0 MB** |

## Notes / next steps

- **No files were modified or deleted on A800.** This was a read-only inspection.
- The `outputs/` directory is only ~558 MB; the overall A800 disk pressure (99% full, ~58 GB free) is largely outside this repository.
- If you want a dry-run cleanup script, I can generate one that `rm`s only the items above (excluding the v85 backup until you confirm the run is complete).
