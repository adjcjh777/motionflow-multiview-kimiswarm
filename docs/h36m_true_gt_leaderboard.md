# H36M True-GT S9/S11 Test Leaderboard

> Protocol: train on **S1, S5, S6, S7, S8** → test on **S9, S11**.  
> Labels: `data/h36m_true_gt/*_multiview_m.npz` (true mocap world coordinates, non-circular).  
> Last updated: 2026-08-12 (v82 complete; AIST++ fast v2 → H36M cross-eval complete; MPI RTMPose 16/16 complete; v85 random-view dropout in progress).

| Rank | Method | S9 MPJPE (mm) | S11 MPJPE (mm) | Combined MPJPE (mm) ↓ | PA-MPJPE (mm) | Notes |
|:----:|--------|--------------:|---------------:|----------------------:|--------------:|-------|
| 1 | **Iskakov ICCV 2019** | 27.10 | 19.60 | **23.35** | — | Learnable triangulation; best epoch 4 |
| 2 | DLT (confidence-weighted) | 29.54 | 21.81 | **25.67** | 28.05 | Frozen reference; `data/h36m_true_gt/dlt_baseline_h36m.json` |
| 3 | RANSAC/conf-DLT (reproducible) | 29.60 | 21.96 | **26.47** | 28.98 | Confidence-weighted 3-view random-subset; `scripts/run_h36m_true_gt_ransac_baseline.py` |
| 4 | RANSAC/conf-DLT (historical) | 29.86 | 21.91 | **26.61** | 26.98 | Untracked one-off; `outputs/ransac_dlt_h36m_true_gt_focused.json` |
| 5 | DLT (unweighted) | 32.97 | 24.57 | **28.77** | 32.10 | Frozen reference; `scripts/run_h36m_true_gt_dlt_baseline.py --unweighted` |
| 6 | **v25 stability (A800)** | 34.87 | 26.80 | **30.83** | 33.59 | Best learned result; best val 31.13 @ Epoch 10; early-stopped @ Epoch 12; stride 1 |
| 7 | v25 mixed (H36M + AIST++, A800) | 37.87 | 28.96 | **33.42** | 34.60 | Diverged @ Epoch 3; best ckpt tested; stride 13 |
| 8 | v81 temporal-pose-attention (A800) | 42.19 | 33.46 | **37.83** | 37.75 | Best val 38.62 @ Epoch 8; stride 13; EMA |
| 9 | v82 multi-scale temporal-pose-attention (A800) | 42.07 | 36.84 | **39.46** | 39.94 | Best val 39.58 @ Epoch 8; stride 13; EMA |
| 10 | v25 medium (local 4090) | 47.28 | 40.54 | **43.91** | 39.53 | Original val 72.80 was inflated by missing `view_mask`; stride 1 |
| 11 | v46 SVG sparse-view generalization (A800) | 55.03 | 49.88 | **52.46** | 40.20 | Best val 52.92 @ Epoch 4; stride 13; EMA |
| 12 | v80 regularization ablation (A800) | 56.69 | 51.27 | **53.98** | 32.47 | Best val 54.46 @ Epoch 1; early-stopped @ Epoch 4; stride 13; EMA |
| 13 | v52 UWT (A800) | 58.15 | 49.87 | **54.01** | 42.22 | Best val 54.75 @ Epoch 4; early-stopped @ Epoch 7; stride 13; EMA |
| 14 | v57 re-run (A800) | 61.09 | 53.11 | **57.10** | 37.30 | Best val 57.81 @ Epoch 4; early-stopped @ Epoch 7; stride 13; EMA |
| 15 | v57 medium (local) | 62.48 | 56.69 | **59.59** | 47.32 | Stale epoch-2 checkpoint bug (fixed in re-run); stride 13 |
| 16 | v80 medium (local 4090) | 64.18 | 60.46 | **62.32** | 57.23 | Overfit after epoch 4; stride 13 |

## Cross-domain & external benchmarks

These results are not directly comparable to the H36M S9/S11 numbers above (different datasets / protocols), but they anchor the paper's cross-domain story.

| Benchmark | Method | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|:---|:---|---:|---:|:---|
| AIST++ smoke | DLT (confidence-weighted) | **6.52** | — | 3-clip smoke split |
| AIST++ smoke | Iskakov ICCV 2019 | **9.31** | — | CPU smoke, best epoch 6 |
| AIST++ full 1,408 clips | DLT (confidence-weighted) | **15.93** | **21.12** | 1,123,873 frames |
| AIST++ full → H36M true-GT | AIST++-only v25 fast v2 | S9 **98.17** / S11 **89.70** | S9 **49.44** / S11 **39.55** | Zero-shot cross-domain transfer; stride 1 |
| MPI-INF-3DHP detected 2D | DLT (confidence-weighted) | **115.09** | **132.68** | RTMPose detected-2D; 16 `.npz` files |

- AIST++ sources: `outputs/aistpp_full_dlt_baseline_a800.json`, `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json`.
- MPI source: `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`.

## How combined MPJPE is computed

- **Combined** is the simple average of S9 and S11 test MPJPE, matching the convention in `docs/results_true_gt_h36m.md`.
- The JSON `combined` fields from A800 test scripts also report this simple average.
- Some prior summaries also report a **weighted-by-frames** combined (e.g., v25 stability: 31.56 mm weighted vs. 30.83 mm simple average). The table above uses the simple average for consistency.

## Key sources

| Method | Source JSON / config |
|--------|----------------------|
| Iskakov ICCV 2019 | `outputs/iskakov_h36m_true_gt.config.json` |
| DLT (conf-weighted) | `data/h36m_true_gt/dlt_baseline_h36m.json` |
| DLT (unweighted) | `data/h36m_true_gt/dlt_baseline_h36m.json` (run with `--unweighted`) |
| RANSAC/conf-DLT (reproducible) | `outputs/ransac_dlt_h36m_true_gt_reproducible.json` |
| RANSAC/conf-DLT (historical) | `outputs/ransac_dlt_h36m_true_gt_focused.json` |
| v25 stability | A800 `outputs/eval_v25_true_gt_stability_h36m_test.json` |
| v25 mixed | A800 `outputs/eval_v25_true_gt_mixed_dataset_a800_h36m_test.json` |
| v81 | A800 `outputs/eval_v81_true_gt_h36m_test_a800.json` |
| v82 | A800 `outputs/eval_v82_true_gt_h36m_test_a800.json` |
| v25 medium | WSL `outputs/eval_v25_true_gt_h36m_test.json` |
| v46 | A800 `outputs/eval_v46_true_gt_h36m_test.json` |
| v80 reg | A800 `outputs/eval_v80_true_gt_h36m_test_a800.json` |
| v52 | A800 `outputs/eval_v52_true_gt_h36m_test_a800.json` |
| v57 re-run | A800 `outputs/eval_v57_true_gt_h36m_test_a800.json` |
| v57 local | WSL `outputs/eval_v57_true_gt_h36m_test_local_stride13.json` |
| v80 medium | WSL `outputs/eval_v80_true_gt_h36m_test_local_stride13.json` |

## Takeaways

1. **Iskakov remains the strongest overall method** at 23.35 mm, beating all geometric and learned baselines.
2. **v25 stability is the current best learned MotionFlow variant** at 30.83 mm, but still 7.5 mm behind Iskakov and 5.2 mm behind confidence-weighted DLT.
3. Temporal modules (v81, v82) improve over the original v25 (43.91 mm) but do not yet match v25 stability, possibly due to stride-13 vs. stride-1 evaluation and regularization differences.
4. Sparse-view / regularization variants (v46, v52, v57, v80) currently sit in the 52–62 mm range and need further tuning to close the gap.
5. **Cross-domain transfer is hard:** an AIST++-only v25 model scores ~94 mm on H36M true-GT, confirming a large domain gap.
6. **MPI-INF-3DHP with real RTMPose detections is very challenging:** the confidence-weighted DLT baseline is 115.09 mm, showing the large gap between clean H36M and in-the-wild detected data.
