# A800 Status Snapshot — 2026-08-12

> Captured: 2026-08-12 12:56:56 UTC  
> Local WSL repo: `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> A800 repo: `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`

## GPU Status (A800)

| GPU | Name | Util.GPU | Util.Mem | Mem.Total | Mem.Used | Mem.Free | Temp | P-State |
|-----|------|----------|----------|-----------|----------|----------|------|---------|
| 0 | NVIDIA A800-SXM4-80GB | 0 % | 0 % | 81920 MiB | 76135 MiB | 5017 MiB | 35 °C | P0 |
| 1 | NVIDIA A800-SXM4-80GB | 0 % | 0 % | 81920 MiB | 76135 MiB | 5017 MiB | 31 °C | P0 |
| 2 | NVIDIA A800-SXM4-80GB | 0 % | 0 % | 81920 MiB | 76135 MiB | 5017 MiB | 31 °C | P0 |
| 3 | NVIDIA A800-SXM4-80GB | 0 % | 0 % | 81920 MiB | 76135 MiB | 5017 MiB | 35 °C | P0 |
| 4 | NVIDIA A800-SXM4-80GB | 0 % | 0 % | 81920 MiB | 673 MiB | 80479 MiB | 34 °C | P0 |
| 5 | NVIDIA A800-SXM4-80GB | 0 % | 0 % | 81920 MiB | 57847 MiB | 23306 MiB | 31 °C | P0 |
| **6** | **NVIDIA A800-SXM4-80GB** | **34 %** | **0 %** | **81920 MiB** | **1481 MiB** | **79672 MiB** | **31 °C** | **P0** |
| **7** | **NVIDIA A800-SXM4-80GB** | **86 %** | **9 %** | **81920 MiB** | **46146 MiB** | **35006 MiB** | **41 °C** | **P0** |

- **GPU 6**: used by v85 split-k variable-view eval (PID 2148515, currently k=2).
- **GPU 7**: used by v85 random view dropout training (PID 2058225, Epoch 3 in progress).

## Disk Usage

| Filesystem | Size | Used | Avail | Use% | Mounted on |
|------------|------|------|-------|------|------------|
| `/dev/nvme0n1p1` (project/data) | 3.5T | 3.3T | 58G | **99%** | `/mnt/nvme0n1p1` |
| `/dev/mapper/ubuntu--vg-ubuntu--lv` (root) | 437G | 381G | 38G | 91% | `/` |
| `/dev/nvme1n1p1` | 3.5T | 2.7T | 641G | 81% | `/mnt/nvme1n1p1` |

- A800 project repo size: ~2.0 GB.
- `/mnt/nvme0n1p1` remains at **99% capacity**; large writes should be avoided until cleanup is run.

## Active Project PIDs

| PID | PPID | GPU | %CPU | %MEM | ELAPSED | State | Task |
|-----|------|-----|------|------|---------|-------|------|
| 2058225 | 2058223 | 7 | 100 | 0.3 | 01:21:27 | Rl | v85 random view dropout training |
| 2148510 | 1 | — | 0.0 | 0.0 | 00:12:00 | S | split-k eval launcher / session leader |
| 2148515 | 2148511 | 6 | 101 | 0.1 | 00:12:00 | Rl | v85 split-k variable-view eval (k=2) |
| 2072251 | 1 | — | 0 | — | — | S | v85 post-training eval suite monitor |
| 2072252 | 2072251 | — | 0 | — | — | S | eval suite monitor (child) |
| 2146696 | 1 | — | 0 | — | — | S | VoxelPose launch-after-eval monitor |
| 2146699 | 2146696 | — | 0 | — | — | S | VoxelPose monitor (child) |

### Training details
- **PID 2058225** is running `experiments/train_omniview_fusion_v5_webbridge_multi.py` with `CUDA_VISIBLE_DEVICES=7`.
- Flags include: `--use_random_view_dropout_v85`, `--v85_dropout_prob 0.3`, `--v85_min_views 2`, `--v85_use_count_embedding`, and many v25/v80/v81/v82 architecture flags.
- 4 DataLoader worker children (PIDs 2158410–2158413) spawned by PID 2058225.

### Eval details
- **PID 2148515** is running `experiments/eval_variable_views.py` on GPU 6 with `--k_values 2`.
- Outputs: `outputs/variable_view_v85_random_view_dropout_medium_a800_k2.{csv,json}`.
- At capture time, no k=2 output files contained data yet.

## Latest 30 Lines of v85 Training Log

File: `outputs/ablations/v85_random_view_dropout_medium_a800.log`

```text
  train step 1100: loss=15.449843
  train step 1150: loss=15.304775
  train step 1200: loss=15.151788
  train step 1250: loss=15.002606
Epoch 2: train_loss=14.910560, val_loss=0.000690, val_MPJPE=36.48mm
  train step 50: loss=11.107350
  train step 100: loss=11.080187
  train step 150: loss=10.986425
  train step 200: loss=10.844447
  train step 250: loss=10.711629
  train step 300: loss=10.551463
  train step 350: loss=10.434668
  train step 400: loss=10.269525
  train step 450: loss=10.128754
  train step 500: loss=9.971600
  train step 550: loss=9.823399
  train step 600: loss=9.686951
  train step 650: loss=9.597321
  train step 700: loss=9.484853
  train step 750: loss=9.391644
  train step 800: loss=9.294049
  train step 850: loss=9.205652
  train step 900: loss=9.118785
  train step 950: loss=9.031479
  train step 1000: loss=8.943329
  train step 1050: loss=8.866835
  train step 1100: loss=8.783078
  train step 1150: loss=8.716976
  train step 1200: loss=8.647389
  train step 1250: loss=8.577945
```

- **Status**: Epoch 1 finished at `train_loss=17.48`, `val_MPJPE=62.53 mm`.  
  **Epoch 2** finished at `train_loss=14.91`, `val_MPJPE=36.48 mm`.  
  Now in **Epoch 3**, train step loss is falling steadily (last logged: step 1250 loss = 8.58).

## Latest 30 Lines of v85 Split-k Eval Log

Files: `outputs/variable_view_v85_random_view_dropout_medium_a800.log` and `outputs/variable_view_v85_random_view_dropout_medium_a800_k2.log`

```text
# Log file is currently empty (0 bytes) at the time of capture.
# The split-k eval was launched at 12:45 UTC and is still running k=2.
```

- The per-k log (`_k2.log`) and the split-k nohup log are also empty/quiet at capture time.
- No CSV/JSON results for k=2 have been written yet.

## Warnings / Watch-outs

1. **GPU 7 busy training**: do not kill or restart PID 2058225. v85 is the only in-flight training run.
2. **GPU 6 busy with split-k eval**: do not kill or restart PID 2148515 / its session leader 2148510. Currently on k=2.
3. **Disk pressure**: `/mnt/nvme0n1p1` is 99% full (~58 GB free). Avoid large writes; consider running `scripts/cleanup_a800_safe.sh` dry-run before any new large jobs.
4. **GPU policy**: project only uses GPUs 6 and 7. GPUs 0–5 are reserved for other workloads.

## Next Steps

- Wait for v85 training to finish; monitor `outputs/ablations/v85_random_view_dropout_medium_a800.log`.
- Wait for the split-k eval to progress through k=2/3/4; check `outputs/variable_view_v85_random_view_dropout_medium_a800_k*.json`.
- The post-training eval monitor (`scripts/monitor_v85_then_run_evals.sh`, PID 2072251) will launch test-set eval, no-fallback eval, and DLT-fallback eval once GPU 6 or 7 is free.
- Consider cleanup of old failed runs (v83/v84) to free disk before scheduling VoxelPose/MVPose SOTA baselines.
