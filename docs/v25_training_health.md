# v25 H36M True-GT Training Health Report

**Generated:** 2026-08-11 10:56 UTC (during inspection)  
**Run:** `scripts/run_v25_h36m_true_gt_medium_local_4090.sh` on local RTX 4090  
**Output prefix:** `outputs/omniview_fusion_v25_h36m_true_gt_medium`

## 1. Process Status

| Item | Value |
|------|-------|
| Shell PID | `50033` (`bash scripts/run_v25_h36m_true_gt_medium_local_4090.sh`) |
| Python PID | `50037` (`python -u experiments/train_omniview_fusion_v5_webbridge_multi.py ...`) |
| Started | 2026-08-11 10:46:54 (≈10 min elapsed at inspection) |
| State | `R` (runnable), 2 threads, resident memory ≈2.65 GB |
| Parent/child | Started directly under PID 1; Python is child of the bash wrapper |

## 2. GPU Usage

```
2026/08/11 10:55:47, NVIDIA GeForce RTX 4090, 60 °C
GPU Utilization: 100 %
Memory Utilization: 3 %
Memory Used: 12320 MiB / 24564 MiB total
```

- The RTX 4090 is fully occupied by the single v25 training process.
- No other GPU processes were detected.

## 3. Log Inspection

**Log file:** `outputs/omniview_fusion_v25_h36m_true_gt_medium.log`

- **Not truncated.** The file was actively being appended during inspection (mtime `10:55`, size 469 B and growing).
- Current content (last lines):

```
Device: cuda
Model params: 2731695
  train step 50: loss=16.800499
  train step 100: loss=12.940161
  train step 150: loss=10.290557
  train step 200: loss=8.611329
  train step 250: loss=7.494233
  train step 300: loss=6.709237
Epoch 1: train_loss=6.462229, val_loss=0.002519, val_MPJPE=83.19mm
  train step 50: loss=6.678796
  train step 100: loss=6.668445
  train step 150: loss=6.721706
  train step 200: loss=6.639841
  train step 250: loss=6.550868
  train step 300: loss=6.492619
```

- Loss is decreasing across the first epoch and remained stable entering the second epoch.
- No exceptions, CUDA errors, or NaNs observed in the log.

## 4. Checkpoint Timestamps

| File | Mtime | Size | Notes |
|------|-------|------|-------|
| `.config.json` | 2026-08-11 10:47 | 15 KB | Written at start of current run |
| `.pth` | 2026-08-11 10:52 | 39 MB | Current run checkpoint (epoch 1 saved) |
| `.log` | 2026-08-11 10:55 | 469 B (growing) | Active log |
| `.nohup` | 2026-08-10 23:04 | 0 B | Empty / unused for this launch |
| `.pth_final` | 2026-08-10 23:58 | 39 MB | **Stale artifact from a previous run; not produced by the current process.** |

The most recent checkpoint (`.pth`) is from the current run and was saved only a few minutes before inspection, indicating the training loop is progressing normally.

## 5. Training Configuration Summary

Key settings from the active run (from `omniview_fusion_v25_h36m_true_gt_medium.config.json`):

- **Manifest:** `configs/splits/h36m_true_gt_standard.yaml`
- **Epochs:** 8
- **Samples per epoch:** 1024 (batch size 16 → 64 steps/epoch)
- **Model:** v25 geometry fusion, deformable cross-view attention (v18), robust DLT reweight, variable-view training (2–4 views)
- **Optimizer:** Adam, lr=1e-3 with cosine decay and 1-epoch warmup
- **Output target:** `outputs/omniview_fusion_v25_h36m_true_gt_medium.pth`

## 6. Health Verdict

- **Status:** `HEALTHY / RUNNING`
- **Progress:** Completed Epoch 1; Epoch 2 in progress at inspection time.
- **Concerns:**
  - The stale `*_final.pth` from 2026-08-10 23:58 could be mistaken for a completed run; it is **not** from the current process.
  - Log verbosity is low (only one line per 50 global steps), which is normal for this script but makes fine-grained progress hard to see. This is not a truncation issue.

## 7. Recommendations

1. **Do not start another GPU task** until this run finishes; the RTX 4090 is fully utilized.
2. **Remove or rename the stale `*_final.pth`** after the current run completes to avoid confusion during evaluation.
3. **Continue monitoring** the log for the Epoch 2 validation MPJPE; a value near or below the Epoch 1 83.19 mm would confirm convergence is on track.
4. If the run finishes successfully, verify the new `*_final.pth` timestamp matches the run end time before using it for evaluation.
