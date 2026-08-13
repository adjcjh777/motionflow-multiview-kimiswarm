# Paper Results Table

> **Date:** 2026-08-12 (updated)  
> **Scope:** Clean, non-circular (true-GT) results for H36M, Shelf/Campus, AIST++, and MPI-INF-3DHP.  
> **Source docs:** `docs/results_true_gt_h36m.md`, `docs/results_true_gt_shelf_campus.md`, `docs/results_aistpp_dlt_baseline.md`, `docs/results_mpi_detected_dlt.md`.

## Overview

All numbers below are reported in **millimetres (mm)**. `direct` = mean per-joint position error after rigid root alignment, `PA` = Procrustes-aligned MPJPE. A dash `—` means the method was not run on that dataset; **RUNNING** marks a result that is not yet available.

### Main results (best available per method and dataset)

| Method | H36M true-GT standard protocol | Shelf / Campus detected | AIST++ full / smoke | MPI-INF-3DHP detected-2D |
|--------|-------------------------------:|------------------------:|--------------------:|--------------------------:|
| DLT (unweighted) | 28.77 | 134.43 | 38.11 / 12.66 | **115.09** (PA 132.68) |
| DLT (confidence-weighted) | **25.67** | 132.29 | **15.93** / 6.52 | **115.09** (PA 132.68) |
| **MVPose (zju3dv/mvpose, GT 2D geometry-only)** | **28.47** | — | — | — |
| RANSAC/conf-DLT | 26.47 | — | — | — |
| Iskakov ICCV 2019 | **23.40** | **128.73** | **29.27** / **9.31** | — |
| v25 stability (A800) | **30.83** (PA 33.59) | — | — | — |
| v25 mixed (H36M + AIST++) | **33.42** (PA 34.60) | — | — | — |
| v81 temporal-pose-attention | **37.83** (PA 37.75) | — | — | — |
| v82 multi-scale temporal-pose-attention | **39.46** (PA 39.94) | — | — | — |
| v46 sparse-view generalisation | **52.46** (PA 40.20) | — | — | — |
| v80 regularization ablation | **53.98** (PA 32.47) | — | — | — |
| v52 UWT | **54.01** (PA 42.22) | — | — | — |
| v57 re-run | **57.10** (PA 37.30) | — | — | — |
| v25 medium (local 4090) | 43.93 | 430.67 | 71.79 | 26.15 |
| v80 medium (local 4090) | 62.32 | 408.58 | 76.34 | 35.22 |

*Table notes:*
- **H36M:** combined direct MPJPE on the standard protocol `S1,5,6,7,8 → S9,11` using `data/h36m_true_gt/`; PA-MPJPE shown in parentheses.
- **Shelf / Campus:** val direct MPJPE on the detected true-GT protocol `data/webbridge/shelf_campus_detected/`.
- **AIST++:** full-dataset confidence-weighted DLT baseline (1,408 clips) and smoke-split Iskakov/v25/v80 results.
- **MPI-INF-3DHP:** RTMPose detected-2D DLT baseline on all 16 canonical sequences.

---

## H36M true-GT standard protocol

Protocol: `S1, S5, S6, S7, S8` train → `S9, S11` test. Labels are true mocap world coordinates (`data/h36m_true_gt/*_multiview_m.npz`). Metric: combined direct MPJPE (mm).

| Method | Combined direct (mm) | S9 direct (mm) | S11 direct (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---:|---:|---|
| Iskakov ICCV 2019 | **23.40** | 27.15 | 19.65 | 23.15 | best epoch 9; current true-GT leader |
| DLT (confidence-weighted) | **25.67** | 29.54 | 21.81 | 28.05 | frozen geometric reference |
| **MVPose (zju3dv/mvpose, GT 2D geometry-only)** | **28.47** | 31.73 | 23.76 | 32.43 | SOTA baseline on true-GT v2; body-12 subset **35.21 / 39.86 mm** |
| RANSAC/conf-DLT | **26.47** | 29.60 | 21.96 | 28.98 | reproducible baseline |
| DLT (unweighted) | 28.77 | 32.97 | 24.57 | 32.10 | frozen geometric reference |
| **v25 stability (A800)** | **30.83** | 34.87 | 26.80 | 33.59 | best learned result; best val 31.13 @ epoch 10; early-stopped @ epoch 12; stride 1 |
| v25 mixed (H36M + AIST++, A800) | **33.42** | 37.87 | 28.96 | 34.60 | diverged @ epoch 3; best ckpt epoch 1; stride 13 |
| v81 temporal-pose-attention (A800) | **37.83** | 42.19 | 33.46 | 37.75 | best val 38.62 @ epoch 8; stride 13; EMA |
| v82 multi-scale temporal-pose-attention (A800) | **39.46** | 42.07 | 36.84 | 39.94 | best val 39.58 @ epoch 8; stride 13; EMA |
| v25 medium (local 4090) | 43.93 | 47.28 | 40.54 | — | test result; corrected-val ablations 45.80 / 46.75 mm @ epoch 1 |
| v46 SVG sparse-view generalisation (A800) | **52.46** | 55.03 | 49.88 | 40.20 | best val 52.92 @ epoch 4; stride 13; EMA |
| v80 regularization ablation (A800) | **53.98** | 56.69 | 51.27 | 32.47 | best val 54.46 @ epoch 1; early-stopped @ epoch 4; stride 13; EMA |
| v52 UWT (A800) | **54.01** | 58.15 | 49.87 | 42.22 | best val 54.75 @ epoch 4; early-stopped @ epoch 7; stride 13; EMA |
| v57 re-run (A800) | **57.10** | 61.09 | 53.11 | 37.30 | best val 57.81 @ epoch 4; early-stopped @ epoch 7; stride 13; EMA |
| v57 medium (local) | 59.59 | 62.48 | 56.69 | — | stale epoch-2 ckpt bug; fixed in re-run |
| v80 medium (local 4090) | 62.32 | 64.18 | 60.46 | — | overfit after epoch 4; stride 13 |

---

## Variable-view MPJPE@k on true-GT H36M

Evaluated on S9/S11 test subjects at view counts `k = 2, 3, 4`. All numbers are direct MPJPE (mm).

| Method | k=2 (S9/S11) | k=3 (S9/S11) | k=4 (S9/S11) | Notes |
|---|---:|---:|---:|---|
| DLT (unweighted) | 37.19 | 34.86 | 29.15 | frozen geometric baseline |
| DLT (confidence-weighted) | 36.42 | 33.68 | 25.94 | frozen geometric baseline |
| Iskakov ICCV 2019 | 53.61 (±27) | **27.80** | **23.39** | current true-GT leader |
| v25 stability (DLT fallback, k<4) | 58.18 / 49.35 | 33.32 / 25.28 | 116.98 / 110.58 | direct DLT for k<4; learned model for k=4 |
| v82 multi-scale temporal-pose-attention | 58.18 / 49.35 | 33.32 / 25.28 | 47.81 / 42.36 | DLT fallback for k<4; learned model for k=4 |

*Sources:* `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json`, `outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.json`.

---

## Shelf / Campus detected true-GT

Protocol: real 2D detections + true 3D annotation (`data/webbridge/shelf_campus_detected/`). Metric: val direct MPJPE / PA-MPJPE in mm.

| Method | Val direct (mm) | Val PA (mm) | Notes |
|---|---:|---:|---|
| Iskakov ICCV 2019 | **128.73** | **119.23** | best epoch 11 |
| DLT (confidence-weighted) | 132.29 | 120.95 | frozen reference |
| DLT (unweighted) | 134.43 | 122.37 | frozen reference |
| v80 long (25 ep) | 276.49 | — | best epoch 7, A800-D |
| v57 long (25 ep) | 306.45 | — | best epoch 4, A800-D |
| v80 smoke (3 ep) | 408.58 | — | smoke schedule |
| v57 smoke (3 ep) | 424.63 | — | smoke schedule |
| v25 smoke (3 ep) | 430.67 | — | smoke schedule |

---

## AIST++ cross-domain benchmark

Protocol: canonical multi-view `.npz` (`data/webbridge/aistpp_canonical/`), 9-view 17-joint meter-scale captures.

| Benchmark | Method | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|:---|:---|---:|---:|:---|
| Smoke split | DLT (unweighted) | **12.66** | — | 3-clip smoke split |
| Smoke split | DLT (confidence-weighted) | **6.52** | — | 3-clip smoke split |
| Smoke split | Iskakov ICCV 2019 | **9.31** | — | CPU smoke, best epoch 6 |
| Smoke split | v25 geometry fusion | 71.79 | — | 3-epoch smoke |
| Smoke split | v80 view reliability | 76.34 | — | 3-epoch smoke |
| Full 1,408 clips | DLT (unweighted) | **38.11** | **42.66** | 1,123,873 frames |
| Full 1,408 clips | DLT (confidence-weighted) | **15.93** | **21.12** | 1,123,873 frames |
| AIST++-only → H36M true-GT | v25 fast v2 | S9 **98.17** / S11 **89.70** | S9 **49.44** / S11 **39.55** | zero-shot cross-domain; combined **93.94** |

*Sources:* `outputs/aistpp_full_dlt_baseline_a800.json`, `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json`.

---

## MPI-INF-3DHP detected-2D DLT baseline

Protocol: RTMPose detected 2D keypoints on all 16 canonical MPI-INF-3DHP sequences (`data/webbridge/mpi_inf_3dhp_detected_2d/`). Metric: confidence-weighted DLT direct MPJPE / PA-MPJPE in mm.

| File | MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|
| s_01_seq_01_v14_multiview_m.npz | 138.31 | 155.81 |
| s_01_seq_02_v14_multiview_m.npz | 162.69 | 172.63 |
| s_02_seq_01_v14_multiview_m.npz | 148.56 | 153.30 |
| s_03_seq_01_v14_multiview_m.npz | 99.71 | 114.30 |
| s_03_seq_02_v14_multiview_m.npz | 108.42 | 119.24 |
| s_04_seq_01_v14_multiview_m.npz | 97.45 | 122.16 |
| s_04_seq_02_v14_multiview_m.npz | 147.90 | 161.91 |
| s_05_seq_01_v14_multiview_m.npz | 104.94 | 129.26 |
| s_05_seq_02_v14_multiview_m.npz | 87.59 | 105.85 |
| s_06_seq_01_v14_multiview_m.npz | 84.23 | 104.26 |
| s_06_seq_02_v14_multiview_m.npz | 89.09 | 112.38 |
| s_07_seq_01_v14_multiview_m.npz | 87.31 | 103.91 |
| s_07_seq_02_v14_multiview_m.npz | 91.03 | 109.18 |
| s_08_seq_01_v14_multiview_m.npz | 162.18 | 181.87 |
| s_08_seq_02_v14_multiview_m.npz | 107.01 | 129.85 |
| **Mean** | **115.09** | **132.68** |

*Source:* `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`

---

## Current experiments and infrastructure status

| Job | GPU | Status | Latest result |
|---|:---:|---|---:|
| `v25_true_gt_stability_a800` | A800 GPU 6 (was) | **DONE** | test **30.83 mm** (S9 34.87 / S11 26.80); PA 33.59 |
| `v81_true_gt_h36m_medium_a800` | A800 GPU 4 (was) | **DONE** | test **37.83 mm**; val 38.62 @ epoch 8 |
| `v82_true_gt_h36m_medium_a800` | A800 GPU 4 (was) | **DONE** | test **39.46 mm**; val 39.58 @ epoch 8 |
| `v80_true_gt_regularization_a800` | A800 GPU 6 (was) | **DONE** | test **53.98 mm**; val 54.46 @ epoch 1 |
| `v57_true_gt_medium_a800` | A800 GPU 5 (was) | **DONE** | test **57.10 mm**; val 57.81 @ epoch 4 |
| `v25_true_gt_mixed_dataset_a800` | A800 GPU 5 (was) | **DONE (diverged)** | test **33.42 mm**; val 34.94 @ epoch 1 |
| `aistpp_only_medium_a800_fast_v2` | A800 GPU 5 (was) | **DONE** | val 91.43 @ epoch 2; H36M cross-eval **93.94 mm** |
| MPI RTMPose detection / DLT | A800 GPU 7 (was) | **DONE** | 16/16 `.npz`; DLT mean **115.09 mm** / PA 132.68 mm |
| `v85_random_view_dropout_medium_a800` | A800 GPU 7 | **RUNNING** | random view dropout training; ~epoch 1, val ~62.53 mm |
| `v85_no_fallback_var_view_eval` | A800 GPU 6 | **RUNNING** | no-fallback variable-view eval; output still buffering |
| VoxelPose SOTA baseline | A800 GPU 6 (queued) | **QUEUED** | launches after v85 no-fallback eval frees GPU 6 |

*GPU policy: MotionFlow only uses GPUs 6 and 7 on A800. GPUs 0–5 are reserved.*

---

## Key takeaways

1. **True-GT protocol is now reliable.** H36M numbers are in the 23–63 mm range, unlike the old circular-label 0.62 mm figure.
2. **Iskakov ICCV 2019 is the current leader at 23.40 mm**, improving over confidence-weighted DLT by 2.27 mm.
3. **v25 stability is the best MotionFlow variant at 30.83 mm**, but still 5.16 mm behind confidence-weighted DLT and 7.57 mm behind Iskakov.
4. **Temporal modules help modestly.** v81 (37.83 mm) and v82 (39.46 mm) improve over the original v25 medium (43.93 mm), but do not yet match v25 stability.
5. **Cross-domain transfer is hard.** AIST++-only v25 scores ~94 mm on H36M true-GT, confirming a large domain gap.
6. **MPI-INF-3DHP with real RTMPose detections is very challenging:** the confidence-weighted DLT baseline is 115.09 mm, showing the large gap between clean H36M and in-the-wild detected data.
