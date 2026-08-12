# Handoff: next session (qwen3.8max)

> Date: 2026-08-11 (updated end-of-session)  
> Repo: `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> Read-only: A800-D `/mnt/nvme0n1p1/zhangzy/projects` and the `motionflow` Docker container.

## Currently running

| GPU | Job | Log / output | Latest status |
|-----|-----|--------------|---------------|
| A800 GPU 4 | `v46_true_gt_h36m_a800` | `outputs/ablations/v46_true_gt_h36m_a800.log` | Epoch 1 in progress (~step 200, loss falling 16.6 → 8.5). |
| A800 GPU 5 | `v25_true_gt_mixed_dataset_a800` | `outputs/ablations/v25_true_gt_mixed_dataset_a800.log` | Epoch 1 in progress (~step 2150, loss falling 14.8 → 6.0). |
| A800 GPU 6 | `v25_true_gt_stability_a800` | `outputs/ablations/v25_true_gt_stability_a800.log` | Epoch 2 in progress. Epoch 1 val **60.74 mm**. |
| A800 GPU 7 | MPI RTMPose detection + v57 variable-view eval | `outputs/mpi_rtmpose_detected_2d/generate_20260811_180024.log`, `outputs/variable_view_v57_true_gt_medium_a800.log` | RTMPose regeneration running; 16 files queued, **no output .npz yet**. v57 variable-view eval running (CSV/JSON not yet written). |

- Local RTX 4090 is **idle** — reserved for quick smoke / diagnostics only (`< 30 min`).
- A800 `/mnt/nvme0n1p1` is **99 % full** (~45 GB free).

## Completed since last handoff

| Run | Best val MPJPE | Final / late epoch | GPU | Notes |
|-----|---------------:|-------------------:|-----|-------|
| `v80_true_gt_regularization_a800` | **54.46 mm** @ Epoch 1 | Early-stopped @ Epoch 4 | 4 | Geometry-regularised v80. Reached 58.11 mm @ Epoch 2, 72.68 mm @ Epoch 3, 75.09 mm @ Epoch 4. Checkpoint: `outputs/ablations/v80_true_gt_regularization_a800.pth`. |
| `v57_true_gt_medium_a800` | **57.81 mm** @ Epoch 4 | Early-stopped @ Epoch 7 | 5 (previously) | Fixed `mpjpe` checkpoint monitor saved the true best. Test MPJPE **57.10 mm** (S9 61.09 / S11 53.11, PA-MPJPE 37.30 mm). Checkpoint: `outputs/ablations/v57_true_gt_medium_a800.pth`. |

- GPU 4 is now running `v46_true_gt_h36m_a800`.
- GPU 5 is now running `v25_true_gt_mixed_dataset_a800`.
- GPU 6 is now running `v25_true_gt_stability_a800`.

## Key numbers (H36M true-GT, S1/5/6/7/8 → S9/S11)

| Method | MPJPE (mm) | Notes |
|--------|-----------:|-------|
| Iskakov ICCV 2019 | **23.35** | Current leader |
| Conf-weighted DLT | **25.67** | Frozen reference |
| RANSAC/conf-DLT | **26.47** | Reproducible reference |
| v80 (earlier best) | **39.98** / test 62.32 | Best learned baseline so far (earlier run) |
| v25 (test) | **43.93** | Corrected-val ablations **45.80 / 46.75 mm** @ epoch 1 |
| v80 regularization (A800) | **54.46** @ Epoch 1 | Geometry-regularised; diverged after Epoch 1 |
| v57 (A800 re-run) | **57.81** @ Epoch 4 | Completed; test **57.10 mm** |

## Blockers

1. **v25/v80 overfitting on true-GT H36M**
   - v80 regularization reached 54.46 mm @ Epoch 1, then diverged to 75.09 mm by Epoch 4.
   - `v25_true_gt_stability_a800` (GPU 6) is testing lower LR (`1e-4`), 4-epoch warmup, and `variable_view_permute=false`. Epoch 1 val **60.74 mm**.
   - `v25_true_gt_mixed_dataset_a800` (GPU 5) is testing H36M + AIST++ mixing.
   - `v46_true_gt_h36m_a800` (GPU 4) is testing sparse-view generalization on true-GT H36M.

2. **MPI-INF-3DHP real detected-2D quality**
   - RTMPose regeneration running on GPU 7; no `.npz` produced yet.
   - Do not run learned-model MPI benchmarks until DLT is ~20–30 mm.

3. **A800 disk space**
   - `/mnt/nvme0n1p1` is **99 % full** (~45 GB free). Avoid extra checkpoint dumps or frame extraction until cleanup is done.

## Next actions

1. **Continue monitoring the four running jobs**; next key check is first validation epoch for v46, v25 mixed, and v25 stability.
2. **v25 stability / mixed** (GPU 5/6): decide next recipe based on whether Epoch 1/2 val stays stable or diverges.
3. **v46 sparse-view** (GPU 4): watch whether the v46 recipe generalises better than v25/v80.
4. **MPI RTMPose** (GPU 7): confirm output files appear and re-run DLT baseline once complete.
5. **v57 variable-view eval** (GPU 7): check `outputs/variable_view_v57_true_gt_medium_a800.csv` / `.json` once complete.
6. **Avoid disk-heavy work** until A800 cleanup is done.

## Quick status commands

```bash
# A800 GPU status (read-only inspection)
ssh a800-D "nvidia-smi"

# Tail current runs
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v46_true_gt_h36m_a800.log"
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v25_true_gt_mixed_dataset_a800.log"
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v25_true_gt_stability_a800.log"
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/mpi_rtmpose_detected_2d/generate_20260811_180024.log"
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_v57_true_gt_medium_a800.log"

# Check true-GT leaderboard
cat docs/results_true_gt_h36m.md

# Disk status
ssh a800-D "df -h /mnt/nvme0n1p1"
```
