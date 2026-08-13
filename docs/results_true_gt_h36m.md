# H36M True-GT Standard Protocol Leaderboard

> Standard protocol: **S1, S5, S6, S7, S8 train → S9, S11 test**  
> Labels: `data/h36m_true_gt_v2/*_multiview_m.npz` (true mocap world coordinates, non-circular).  
> Manifest: `configs/splits/h36m_true_gt_v2_standard.yaml`.  
> Last updated: **2026-08-13 ~03:32 UTC** (v86 no-count-embedding ablation done: early-stopped @ Epoch 6, best val **31.64 mm @ Epoch 3**; v25 true-GT v2 val 31.41 mm, test pending; v85 DLT-fallback running on GPU 6, ~8 min in).

## Label audit (non-circularity check)

| Subject | Direct MJE (mm) | Status |
|---:|---:|---|
| S1 train | 16.11 | non-circular |
| S5 train | 16.12 | non-circular |
| S6 train | 16.47 | non-circular |
| S7 train | 16.55 | non-circular |
| S8 train | 13.66 | non-circular |
| S9 test | 33.83 | non-circular |
| S11 test | 24.75 | non-circular |

- Source: `outputs/h36m_true_gt_v2/audit_all_h36m_true_gt_v2.log`
- All values are **>> 0 mm**, confirming the v2 labels are independent 3D mocap GT rather than DLT triangulations of the input 2D.

## Variable-view (sparse-view) robustness

v25 stability variable-view evaluation completed on A800 (GPU 4). Results are for the standard test subjects S9/S11, using `num_subsets_per_k=50` and `clip_len=13`.

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9 | 3482.62 | 1042.45 | 116.98 |
| S11 | 3376.04 | 1030.19 | 110.58 |

- k=4 is plausible (~110 mm) but much worse than the 4-view test result (30.83 mm), indicating the model overfits to the 4-view training rig.
- k=2/k=3 are catastrophic (~1000–3500 mm), matching the same failure pattern seen for v81/v82.
- **Root cause identified:** `HardenedVariableViewInferenceWrapper` (and `VariableViewInferenceWrapper`) zeroed out inactive-view observations but did **not** pass an explicit `view_mask` to `OmniMultiViewFusionV5.forward`. The model therefore treated inactive views as real zero-confidence observations, and downstream attention/triangulation layers without mask awareness produced garbage.
- **Fix applied:** `motionflow_mv/fusion/variable_view_inference.py` now builds a `view_mask` from the active-view mask and passes it to the model. A synthetic smoke test with a random `OmniMultiViewFusionV5` produced finite outputs for k=2/3/4 (no NaN/Inf). Full real-data re-evaluation on A800 completed (`scripts/run_v25_var_view_fix_a800.sh`) but k<4 errors remained catastrophic, confirming the root cause is the learned model rather than the wrapper.
- **DLT-fallback re-evaluation:** A second re-evaluation also added a `--var_view_dlt_fallback` mode to `HardenedVariableViewInferenceWrapper`: whenever the active view count is below the full 4-view rig, the evaluator directly triangulates the active views with confidence-weighted DLT instead of running the learned model. The v25 stability checkpoint is being evaluated with this fallback on A800; results will be inserted below once `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json` is available.

### Variable-view MPJPE@k with DLT fallback for k<4

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9 | 58.18 | 33.32 | 116.98 |
| S11 | 49.35 | 25.28 | 110.58 |

- Source: `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json`
- DLT fallback is triggered only when the active view count is below the full 4-view rig (k<4); k=4 still runs the learned v25 stability model, hence the same ~111-117 mm k=4 values seen without fallback.
- With DLT fallback, k=2 and k=3 become reasonable (S9 58.18/33.32 mm, S11 49.35/25.28 mm), confirming that the catastrophic sparse-view errors came from the learned model, not from the underlying 2D observations.

### v81 temporal-pose-attention variable-view MPJPE@k with DLT fallback for k<4

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9 | **58.18** | **33.32** | — |
| S11 | **49.35** | **25.28** | — |

- Source: `outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_k23.json` (only k=2,3 were evaluated; k=4 would use the learned v81 model and has not been run).
- k<4 numbers are essentially identical to v25/v82 because the DLT-fallback path is model-agnostic. The learned model itself is not evaluated here.

### v82 multi-scale temporal-pose-attention variable-view MPJPE@k with DLT fallback for k<4

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9 | **58.18** | **33.32** | **47.81** |
| S11 | **49.35** | **25.28** | **42.36** |

- Source: `outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.json`
- k<4 uses direct confidence-weighted DLT fallback; k=4 uses the learned v82 model. The v82 k=4 result (47.81/42.36 mm) is consistent with its full 4-view test result (~39.46 mm) and substantially better than the v25 stability k=4 (~117 mm), showing v82's learned full-view estimate is stronger.
- DLT-fallback k=2/k=3 numbers are comparable to the v25 stability DLT-fallback baseline, confirming the sparse-view 2D observations themselves are sound.

### v85 random-view-dropout variable-view MPJPE@k with DLT fallback for k<4

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9 | — | — | — |
| S11 | — | — | — |

- Source: `outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.json` (not generated; eval terminated).
- PID `2269984` disappeared after ~29 min without producing the expected JSON/CSV. The redirect log is empty and the nohup log only contains `Terminated`.
- Until the eval is rerun, the v25/v81/v82 DLT-fallback baseline remains the reference: S9 58.18/33.32/116.98 mm, S11 49.35/25.28/110.58 mm.
- The v85 no-fallback result (k=2 S9 2310.27 / S11 2308.80 mm; k=3 S9 1119.45 / S11 1118.18 mm; k=4 S9 83.52 / S11 77.07 mm) already demonstrates that random view dropout does **not** make the learned model reliable for k<4, so DLT fallback is still the practical choice for sparse views.

## True-GT v2 Leaderboard

| Method | S9 MPJPE (mm) | S11 MPJPE (mm) | Combined MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---:|---:|---|
| **Iskakov ICCV 2019** | 27.15 | 19.65 | **23.40** | **23.15** | Reproduced on true-GT v2 |
| DLT (confidence-weighted) | 29.54 | 21.81 | **25.67** | 28.05 | Frozen geometric baseline |
| **MVPose (zju3dv/mvpose, geometry-only)** | 29.19 | 21.54 | **26.06** | 28.32 | Native COCO17 skeleton; body-12 subset **31.13 / 34.45 mm** |
| **RANSAC/conf-DLT (reproducible)** | 29.60 | 21.96 | **26.47** | 28.98 | Confidence-weighted 3-view random subset |
| **v25 true-GT v2 medium (A800)** | — | — | **pending** | — | v2 protocol; best val **31.41 mm** @ Epoch 6; test pending |
| **v25 stability (A800)** | 34.87 | 26.80 | **30.83** | 33.59 | Best learned result so far; stride 1 |
| **v25 (mixed H36M+AIST++, A800)** | 37.87 | 28.96 | **33.42** | 34.60 | Mixed-dataset training; early-stopped Epoch 3 |
| **v81 temporal-pose-attention (A800)** | 42.19 | 33.46 | **37.83** | 37.75 | 8 epochs, stride 13 |
| **v82 multi-scale temporal-pose-attention (A800)** | 42.07 | 36.84 | **39.46** | 39.94 | 8 epochs, stride 13 |
| **v46 sparse-view generalization (A800)** | 55.03 | 49.88 | **52.46** | 40.20 | SVG + adaptive geometry fusion |
| **v80 regularization (A800)** | 56.69 | 51.27 | **53.98** | 32.47 | Regularization ablation; stride 13 |
| **v52 uncertainty-weighted triangulation (A800)** | 58.15 | 49.87 | **54.01** | 42.22 | UWT + sparse/cross-domain reliability |
| **v57 domain-conditional calibration (A800 re-run)** | 61.09 | 53.11 | **57.10** | 37.30 | DC-PSC re-run; stride 13 |
| **v57 domain-conditional calibration (local)** | 62.48 | 56.69 | **59.59** | — | Local run; stale-checkpoint bug fixed in re-run |
| **v80 view-reliability weighting (medium)** | 64.18 | 60.46 | **62.32** | — | Overfit after epoch 4 |
| **v37 self-critique v2 smoke** | — | — | **87.85** | — | 2-epoch smoke, incomplete |
| **v25 true-GT v2 medium** | — | — | **pending** | — | Training done; best val **31.41 mm** @ Epoch 6; test pending |
| **v86 no-count-embedding / sparse cross-domain v2** | — | — | **pending** | — | A800 GPU 6; early-stopped @ Epoch 6; best val **31.64 mm @ Epoch 3**; test pending |
| **v85 random view dropout (no-fallback, k=4 var-view)** | 83.52 | 77.07 | **80.30** | — | Variable-view k=4 only [^3]; full test pending |
| **v85 DLT-fallback** | — | — | **pending** | — | Variable-view eval running on GPU 6, ~8 min in; results pending |

- Most completed learned rows above were originally trained/evaluated on the pre-v2 true-GT labels and are being re-run on v2; the dedicated v2 re-runs are shown as **pending**. [^2]
- **v25 true-GT v2 medium** has finished training with best val **31.41 mm** @ Epoch 6, which is comparable to the v1 stability run (best val 31.13 mm / test 30.83 mm). Its test-set evaluation is **pending**.
- **v86 no-count-embedding ablation** has finished training with best val **31.64 mm @ Epoch 3** (early-stopped @ Epoch 6). Its test-set evaluation is **pending**.
- **Sparse-view remains unsolved**: v85 trained with random whole-view dropout still fails catastrophically for k<4 without geometric fallback (k=2 ~2310 mm, k=3 ~1119 mm); v85 DLT-fallback eval is **running** on GPU 6.
- The current best completed learned result on this protocol is **v25 stability at 30.83 mm** (v1 data), which still trails Iskakov (23.40 mm), confidence-weighted DLT (25.67 mm), and MVPose (26.06 mm).

## Current results

| Method | S9 direct (mm) | S11 direct (mm) | Combined direct (mm) | Combined PA-MPJPE (mm) | Notes |
|---|---:|---:|---:|---:|---|
| **Iskakov ICCV 2019** | **27.15** | **19.65** | **23.40** | **23.15** | best val epoch 9; run `iskakov_learnable_tri_h36m_true_gt_a800_gpu4` |
| DLT (confidence-weighted) | 29.54 | 21.81 | **25.67** | 28.05 | frozen reference; true-GT v2 |
| **MVPose (zju3dv/mvpose, GT 2D geometry-only)** | **29.19** | **21.54** | **26.06** | **28.32** | Native COCO17 skeleton; body12 subset **31.13 / 34.45 mm** |
| **RANSAC/conf-DLT (reproducible)** | **29.60** | **21.96** | **26.47** | **28.98** | confidence-weighted 3-view random-subset; true-GT v2 |
| DLT (unweighted) | 32.97 | 24.57 | 28.77 | 32.10 | frozen reference; `scripts/run_h36m_true_gt_dlt_baseline.py --unweighted` |
| **v25 stability (A800)** | **34.87** | **26.80** | **30.83** | **33.59** | **test** result (stride 1); best val **31.13 mm** @ Epoch 10; early-stopped @ Epoch 12 |
| **v25 (mixed H36M+AIST++, A800)** | **37.87** | **28.96** | **33.42** | **34.60** | early-stopped Epoch 3; best val **34.94 mm** @ Epoch 1 |
| **v81 (temporal-pose-attention, A800)** | **42.19** | **33.46** | **37.83** | **37.75** | **test** result (stride 13); best val **38.62 mm** @ Epoch 8; completed 8 epochs |
| **v82 (multi-scale temporal-pose-attention, A800)** | **42.07** | **36.84** | **39.46** | **39.94** | **test** result with EMA weights (stride 13); best val **39.58 mm** @ Epoch 8; completed 8 epochs |
| **v25 (medium, local 4090)** | **47.28** | **40.54** | **43.93** | — | **test** result; corrected-val ablations **45.80 / 46.75 mm** @ epoch 1 (was 72.80 mm) |
| **v46 (SVG, A800)** | **55.03** | **49.88** | **52.46** | **40.20** | **test** result with EMA weights (stride 13); val best **52.92 mm** @ epoch 4 |
| **v80 (regularization A800)** | **56.69** | **51.27** | **53.98** | **32.47** | **test** result with EMA weights (stride 13); val best **54.46 mm** @ epoch 1; early-stopped @ epoch 4 |
| **v52 (UWT, A800)** | **58.15** | **49.87** | **54.01** | **42.22** | **test** result with EMA weights (stride 13); val best **54.75 mm** @ epoch 4; early-stopped Epoch 7 |
| **v57 (medium, A800 re-run)** | **61.09** | **53.11** | **57.10** | **37.30** | **test** result with EMA weights (stride 13); val best **57.81 mm** @ epoch 4 |
| **v57 (medium, local)** | **62.48** | **56.69** | **59.59** | — | **test** result (stride 13); local run val best **75.16 mm** @ epoch 3 |
| **v80 (medium)** | **64.18** | **60.46** | **62.32** | — | **test** result (stride 13); val best **39.98 mm** @ epoch 4; overfit afterward |

- Iskakov outperforms the DLT, RANSAC, and all learned baselines on the true-GT protocol.
- v25 stability is currently the best learned result on this true-GT H36M protocol: **test MPJPE 30.83 mm** (S9 34.87 / S11 26.80, stride 1, PA-MPJPE 33.59 mm), followed by v25 mixed-dataset at **33.42 mm** (S9 37.87 / S11 28.96, stride 13, PA-MPJPE 34.60 mm) and v81 at **37.83 mm**.
- v82 multi-scale temporal-pose-attention finished training on GPU 4 with best val **39.58 mm** @ Epoch 8; **test MPJPE: 39.46 mm** (S9 42.07 / S11 36.84, stride 13, PA-MPJPE 39.94 mm).
- v25 stability (low LR, no `variable_view_permute`) finished training with best val **31.13 mm** @ epoch 10 and was early-stopped @ epoch 12; **test MPJPE: 30.83 mm** (S9 34.87 / S11 26.80, stride 1, PA-MPJPE 33.59 mm).
- v81 temporal-pose-attention finished training on GPU 4 with best val **38.62 mm** @ epoch 8; **test MPJPE: 37.83 mm** (S9 42.19 / S11 33.46, stride 13, PA-MPJPE 37.75 mm).
- v80 has been swept on A800 with several recipes; the best converged val is **39.70 mm** (v2, epoch 2, checkpoint on A800), while the local copy gives **42.60 mm** (v3). A new local medium run reached **39.98 mm** val (epoch 4). Verified **test MPJPE: 62.32 mm** (S9 64.18 / S11 60.46, stride 13). All v80 recipes overfit after the best epoch. The A800 regularisation ablation (`v80_true_gt_regularization_a800`) finished with best val **54.46 mm** @ epoch 1 and early-stopped @ epoch 4. Evaluating the saved checkpoint **with EMA weights** gives **test MPJPE: 53.98 mm** (S9 56.69 / S11 51.27, stride 13, PA-MPJPE 32.47 mm); source: `outputs/eval_v80_true_gt_h36m_test_a800.json`.
- v25 (single-dataset) test MPJPE is **43.93 mm** (S9 47.28 / S11 40.54, stride 1). The training log reports a best *validation* MPJPE of **72.80 mm** @ epoch 2, but this value is inflated because validation did not pass `view_mask`. Corrected-validation ablations on A800 reach **45.80 mm** (baseline fix) and **46.75 mm** (geometry regularization) in epoch 1, then diverge.
- v57 local run final val was **80.21 mm** with a true best of **75.16 mm** @ epoch 3, but the saved checkpoint corresponded to epoch 2 (81.47 mm) because the trainer previously monitored `loss` instead of `mpjpe`. Verified **test MPJPE: 59.59 mm** (S9 62.48 / S11 56.69, stride 13). The A800 re-run finished with best val **57.81 mm** @ epoch 4 and the checkpoint is saved correctly at `outputs/ablations/v57_true_gt_medium_a800.pth`. Evaluating the A800 re-run checkpoint **with EMA weights** gives **test MPJPE: 57.10 mm** (S9 61.09 / S11 53.11, stride 13, PA-MPJPE 37.30 mm); using the online (non-EMA) weights gives ~98 mm, so EMA must be applied at inference.

## AIST++ smoke (cross-dataset sanity)

AIST++ uses the same 17-joint skeleton as H36M and 9 calibrated views. The smoke split below trains on `gBR_sBM_cAll_d04_mBR0_ch01/ch02` and validates on `gBR_sBM_cAll_d04_mBR0_ch03` (see `configs/splits/aist_only_smoke.yaml`).

| Method | val MPJPE (mm) | Notes |
|---|---:|---|
| DLT (unweighted) | **12.66** | frozen reference |
| DLT (confidence-weighted) | **6.52** | frozen reference |
| Iskakov ICCV 2019 | **9.31** | best epoch 6, CPU smoke |
| v25 | **71.79** | 3-epoch smoke |
| v80 | **76.34** | 3-epoch smoke |

### Full AIST++ DLT baseline (all 1,408 canonical clips)

| Method | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| DLT (unweighted) | **38.11** | **42.66** | 1,408 clips, 1,123,873 frames |
| DLT (confidence-weighted) | **15.93** | **21.12** | 1,408 clips, 1,123,873 frames |

- Source: `outputs/aistpp_full_dlt_baseline_a800.json` (computed on A800, CPU-only; run with `experiments/run_aistpp_full_dlt_baseline.py`).

- The geometric baselines are very strong on AIST++: confidence-weighted DLT is already below 7 mm on the smoke split and ~16 mm on the full 1,408-clip set, and Iskakov reaches ~9 mm on the smoke split.
- v25/v80 smoke results are far behind the geometric baselines, suggesting the learned models have not yet adapted to AIST++'s camera rig / motion style. These are 3-epoch smoke runs only; full medium runs are needed before drawing firm conclusions.
- Numbers are comparable to the H36M true-GT scale, confirming AIST++ is a viable, non-circular cross-domain dataset.
- Source logs: `outputs/iskakov_aist_smoke.log`, `outputs/omniview_fusion_v25_aist_only_smoke.log`, `outputs/omniview_fusion_v80_aist_only_smoke.log`.
- An AIST++-only medium v25 training run has been launched on A800 GPU 5 now that all 1,408 canonical `.npz` files are present. Log: `outputs/ablations/aistpp_only_medium_a800_gpu5.log`.

A fast v2 run (`train_samples=64`, `batch_size=32`, `epochs=10`) was relaunched on GPU 5 after the AIST++ NaN/empty-sequence fix and finished/early-stopped at Epoch 4. Best val MPJPE **91.43 mm**. Log: `outputs/ablations/aistpp_only_medium_a800_fast_v2.log`; config: `outputs/ablations/aistpp_only_medium_a800_fast_v2.config.json`.

#### AIST++-only fast v2 → H36M true-GT S9/S11 cross-eval

The AIST++-only checkpoint was evaluated on H36M true-GT S9/S11 with the prepared script `scripts/eval_aistpp_only_on_h36m_test_a800.sh` (stride 1, batch 8, `clip_len=13`).

| Subject | MPJPE (mm) | PA-MPJPE (mm) | Frames |
|---|---:|---:|---:|
| S9 | **98.17** | **49.44** | 1,088,711 |
| S11 | **89.70** | **39.55** | 753,467 |
| **Combined (simple avg)** | **93.94** | **44.50** | — |

- Source: `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json`
- The model was trained only on AIST++ and has never seen H36M, so this is a zero-shot cross-domain transfer result. The error is much higher than models trained on H36M (e.g., v25 stability 30.83 mm), confirming a large domain gap between the two datasets, but it is finite and well above the catastrophic k<4 failure mode.
- Combined direct MPJPE: **(98.17 + 89.70) / 2 = 93.94 mm**.

### MPI-INF-3DHP DLT baseline (completed)

RTMPose detected-2D regeneration finished; DLT baseline computed with `scripts/run_mpi_dlt_baseline.py` on the 16 detected-2D `.npz` files.

| File | MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|
| s_01_seq_01_02_v14_multiview_m.npz            |       138.31 |          155.81 |
| s_01_seq_01_v14_multiview_m.npz               |       162.69 |          172.63 |
| s_01_seq_02_v14_multiview_m.npz               |       125.02 |          147.01 |
| s_02_seq_01_v14_multiview_m.npz               |       148.56 |          153.30 |
| s_03_seq_01_v14_multiview_m.npz               |        99.71 |          114.30 |
| s_03_seq_02_v14_multiview_m.npz               |       108.42 |          119.24 |
| s_04_seq_01_v14_multiview_m.npz               |        97.45 |          122.16 |
| s_04_seq_02_v14_multiview_m.npz               |       147.90 |          161.91 |
| s_05_seq_01_v14_multiview_m.npz               |       104.94 |          129.26 |
| s_05_seq_02_v14_multiview_m.npz               |        87.59 |          105.85 |
| s_06_seq_01_v14_multiview_m.npz               |        84.23 |          104.26 |
| s_06_seq_02_v14_multiview_m.npz               |        89.09 |          112.38 |
| s_07_seq_01_v14_multiview_m.npz               |        87.31 |          103.91 |
| s_07_seq_02_v14_multiview_m.npz               |        91.03 |          109.18 |
| s_08_seq_01_v14_multiview_m.npz               |       162.18 |          181.87 |
| s_08_seq_02_v14_multiview_m.npz               |       107.01 |          129.85 |
|---|---:|---:|
| **Mean** | **115.09** | **132.68** |

- Source: `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`


## Per-method details

### MVPose (zju3dv/mvpose geometry-only top-down triangulation)

MVPose was evaluated on the H36M true-GT S9/S11 split using the adapter in
`scripts/sota_baselines/mvpose_h36m_adapter.py`. The upstream `zju3dv/mvpose`
kernel was used for triangulation, with ground-truth 2D projections feeding the
geometry-only top-down pose kernel (`MultiEstimator._top_down_pose_kernel`).

```bash
python scripts/sota_baselines/mvpose_h36m_adapter.py \
    --input_pkl tmp/sota_baselines/mvpose_data_a800/h36m_true_gt_val.pkl \
    --output_dir tmp/sota_baselines/mvpose_predictions_full

python scripts/sota_baselines/eval_mvpose_predictions.py \
    --input_pkl tmp/sota_baselines/mvpose_data_a800/h36m_true_gt_val.pkl \
    --pred_dir tmp/sota_baselines/mvpose_predictions_full \
    --out_json outputs/sota_baselines/mvpose_h36m_true_gt_metrics.json
```

| Subject | All-17 MPJPE (mm) | All-17 PA-MPJPE (mm) | Body-12 MPJPE (mm) | Body-12 PA-MPJPE (mm) | Frames |
|---|---:|---:|---:|---:|---:|
| S9 | **29.19** | **31.90** | **35.33** | **39.10** | 83,759 |
| S11 | **21.54** | **23.15** | **25.06** | **27.73** | 57,971 |
| **Combined** | **26.06** | **28.32** | **31.13** | **34.45** | 141,730 |

- Source: `outputs/sota_baselines/mvpose_h36m_true_gt_metrics.json`
- The body-12 subset (COCO17 joints 5-16, i.e. shoulders/elbows/wrists/hips/knees/ankles) is
  reported separately because the five COCO17 facial keypoints are approximated from the
  single H36M Head joint and can distort PA-MPJPE.
- MVPose slightly beats the confidence-weighted DLT baseline on S11 and is within 0.4 mm
  of DLT on the combined metric (26.06 mm vs. 25.67 mm).

### DLT / RANSAC baselines (true-GT v2)

Computed by `scripts/run_h36m_true_gt_dlt_baseline.py` and `scripts/run_h36m_true_gt_ransac_baseline.py` over the corrected true-GT v2 files in `data/h36m_true_gt_v2/`. The v2 labels were regenerated from the official mocap release with aligned cameras and 2D projections; direct MJE is ~16 mm, confirming they are not circular.

| Method | S9 direct (mm) | S11 direct (mm) | Combined simple (mm) | Combined weighted (mm) | PA-MPJPE (mm) | Source |
|---|---:|---:|---:|---:|---:|---|
| DLT (confidence-weighted) | 29.54 | 21.81 | **25.67** | 26.38 | 28.05 | `outputs/h36m_true_gt_v2/dlt_baseline_h36m_true_gt_v2.json` |
| DLT (unweighted) | 32.97 | 24.57 | **28.77** | 29.53 | 32.10 | `outputs/h36m_true_gt_v2/dlt_baseline_h36m_true_gt_v2.json` |
| RANSAC/conf-DLT (reproducible) | 29.60 | 21.96 | **26.47** | 26.47 | 28.98 | `outputs/h36m_true_gt_v2/ransac_dlt_h36m_true_gt_v2.json` |

```bash
# v2 confidence-weighted DLT (GPU)
python scripts/run_h36m_true_gt_dlt_baseline.py \
    --config configs/splits/h36m_true_gt_v2_standard.yaml \
    --device cuda --unweighted \
    --output outputs/h36m_true_gt_v2/dlt_baseline_h36m_true_gt_v2.json

# v2 reproducible RANSAC-DLT
python scripts/run_h36m_true_gt_ransac_baseline.py \
    --config configs/splits/h36m_true_gt_v2_standard.yaml \
    --output outputs/h36m_true_gt_v2/ransac_dlt_h36m_true_gt_v2.json
```

### DLT / RANSAC baselines (historical true-GT v1)

These numbers were computed on the older `data/h36m_true_gt/*_multiview_m.npz` files before the v2 camera-alignment fix. They are retained for reference only.

```bash
# Confidence-weighted DLT (fast; uses CPU by default, add --device cuda for GPU)
python scripts/run_h36m_true_gt_dlt_baseline.py

# Also compute unweighted DLT (slower)
python scripts/run_h36m_true_gt_dlt_baseline.py --unweighted --device cuda

# Reproducible confidence-weighted 3-view random-subset RANSAC-DLT
python scripts/run_h36m_true_gt_ransac_baseline.py
```

The reproducible RANSAC-DLT result is **26.47 mm** (S9 29.60 mm, S11 21.96 mm, PA-MPJPE 28.98 mm) using the confidence-weighted 3-view random-subset variant. The historical 26.61 mm result in `outputs/ransac_dlt_h36m_true_gt_focused.json` is retained only as a reference; it was produced by an untracked one-off script and is now superseded by the reproducible script.

Reproducible RANSAC-DLT variants on the same true-GT test data:

| Variant | S9 direct | S11 direct | Combined direct | PA-MPJPE | Source |
|---|---:|---:|---:|---:|---|
| Confidence-weighted 3-view random-subset RANSAC-DLT | 29.60 mm | 21.96 mm | **26.47 mm** | **28.98 mm** | `scripts/run_h36m_true_gt_ransac_baseline.py` → `outputs/ransac_dlt_h36m_true_gt_reproducible.json` |
| Confidence-weighted 3-view all-subsets RANSAC-DLT | 27.84 mm | 20.77 mm | **24.94 mm** | **26.92 mm** | `scripts/run_h36m_true_gt_ransac_baseline.py --all_unique` → `outputs/ransac_dlt_h36m_true_gt_reproducible_all_unique.json` |
| Unweighted 3-view random-subset RANSAC-DLT | 31.63 mm | 23.62 mm | **~28.36 mm** | — | `experiments/baselines.baseline_ransac_dlt` (`tmp/verify_ransac_baseline_h36m_exact.py`) |

`experiments/baselines.py` has been updated so that `baseline_ransac_dlt` no longer falls back to plain DLT when `V=4`; this lets RANSAC actually sample 3-view subsets on the 4-view H36M test data, but the resulting number is ~28.36 mm because it uses unweighted subset triangulation. The confidence-weighted variant in the new script closes the gap to the historical 26.61 mm result.

| Reference | S9 direct | S11 direct | Combined direct | PA-MPJPE | Source |
|---|---:|---:|---:|---:|---|
| Unweighted DLT | 32.97 mm | 24.57 mm | 28.77 mm | 32.10 mm | `data/h36m_true_gt/dlt_baseline_h36m.json` |
| Confidence-weighted DLT | 29.54 mm | 21.81 mm | **25.67 mm** | 28.05 mm | `data/h36m_true_gt/dlt_baseline_h36m.json` |
| RANSAC/conf-DLT (reproducible) | 29.60 mm | 21.96 mm | **26.47 mm** | 28.98 mm | `outputs/ransac_dlt_h36m_true_gt_reproducible.json` |
| RANSAC/conf-DLT (historical) | 29.86 mm | 21.91 mm | 26.61 mm | — | `outputs/ransac_dlt_h36m_true_gt_focused.json` (untracked one-off) |

### Iskakov learnable triangulation

Best run:

```bash
python experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m --epochs 10 --batch_size 8 --lr 1e-3 --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --log_path outputs/iskakov_h36m_true_gt.log \
    --ckpt_path outputs/iskakov_h36m_true_gt.pth
```

| Epoch | Combined direct | S9 direct | S11 direct |
|---|---:|---:|---:|
| 1 | 25.31 mm | 29.11 mm | 21.50 mm |
| 2 | 23.60 mm | 27.28 mm | 19.91 mm |
| 3 | 23.49 mm | 27.23 mm | 19.76 mm |
| 4 | 23.45 mm | 27.19 mm | 19.72 mm |
| 5 | 23.44 mm | 27.18 mm | 19.70 mm |
| 6 | 23.41 mm | 27.16 mm | 19.67 mm |
| 7 | 23.42 mm | 27.17 mm | 19.67 mm |
| 8 | 23.40 mm | 27.15 mm | 19.65 mm |
| 9 (best) | **23.40** | **27.15** | **19.65** |
| 10 | 23.41 mm | 27.16 mm | 19.65 mm |

- Early-stopped by patience, best epoch = 9.
- Gain over confidence-weighted DLT: **+2.46 mm** combined direct (25.87 vs. 23.40 mm; run-computed frozen reference).
- Gain over unweighted DLT: **+5.79 mm** combined direct (29.19 vs. 23.40 mm; run-computed frozen reference).
- A confirmation run on A800 GPU 6 with the same recipe produced **23.40 mm** (best epoch 9; log `outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_gpu6.log`). A local run with larger batches (batch 64, 8,192 samples/epoch) produced **23.38 mm** (best epoch 7); see `docs/results_iskakov_h36m_true_gt.md`.

### v80 (view-reliability weighting)

Detailed sweep in `docs/results_v80_h36m_true_gt.md`. Local evidence:

| Recipe | Best val MPJPE (mm) | Best epoch | Log / checkpoint |
|---|---:|---:|---|
| v1 (long, no reg) | 65.28 | 2 | `outputs/a800_h36m_reg/omniview_fusion_v80_h36m_true_gt_long.log` |
| v2 | **39.70** | 2 | A800 only (`..._reg_epoch2best.pth`) |
| v3 (reg) | **42.60** | 2 | `outputs/a800_h36m_reg/omniview_fusion_v80_h36m_true_gt_reg.{log,pth}` |
| v4 (reg) | 45.31 | 2 | `outputs/a800_h36m_reg/v4.{log,pth}` |
| medium | **39.98** | 4 | `outputs/omniview_fusion_v80_h36m_true_gt_medium.{log,pth}` |
| smoke | 98.12 | 2 | `outputs/omniview_fusion_v80_h36m_true_gt_smoke.{log,pth}` |

- The medium recipe improves the local best to **39.98 mm** at epoch 4, but still overfits afterward (epoch 8: 133.71 mm).
- **Test MPJPE: 62.32 mm** (combined direct, simple average of S9/S11; S9 64.18 mm, S11 60.46 mm, stride 13). There is a large train/val-to-test gap: test is ~22 mm worse than the best validation, suggesting the model overfits to the val distribution.
  - Source: `outputs/eval_v80_true_gt_h36m_test_local_stride13.json` (A800, GPU 4).
- Best local result: **39.98 mm** (medium, epoch 4). Best known result: **39.70 mm** (v2, A800 checkpoint).
- v80 still lags Iskakov (~23.40 mm) and even confidence-weighted DLT (~25.67 mm).

### v46 (sparse-view generalization, A800)

```bash
python3 scripts/eval_v46_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v46_true_gt_h36m_a800.pth \
    --out_json outputs/eval_v46_true_gt_h36m_test.json
```

- Log: `outputs/eval_v46_true_gt_h36m_test.log`
- Checkpoint: `outputs/ablations/v46_true_gt_h36m_a800.pth`

| Metric | Value |
|---|---:|
| S9 MPJPE | **55.03 mm** |
| S11 MPJPE | **49.88 mm** |
| Combined MPJPE | **52.46 mm** |
| Combined PA-MPJPE | **40.20 mm** |

- Trained on the corrected true-GT H36M standard protocol with the v46 sparse-view generalization head (random view dropout + per-view reliability head) and v45 adaptive geometry fusion.
- **Test MPJPE: 52.46 mm** (S9 55.03 mm / S11 49.88 mm, stride 13, PA-MPJPE 40.20 mm) when evaluated with the saved EMA shadow weights. Source: `outputs/eval_v46_true_gt_h36m_test.json`.
- Best validation MPJPE was **52.92 mm** @ epoch 4; early-stopped @ epoch 7. v46 sits between v25 (43.93 mm) and the v80 regularization run (53.98 mm) on this protocol.

### v52 (uncertainty-weighted triangulation, A800)

```bash
bash scripts/run_eval_v52_true_gt_h36m_test_a800.sh
```

- Log: `outputs/eval_v52_true_gt_h36m_test_a800.log`
- Checkpoint: `outputs/ablations/v52_true_gt_h36m_a800.pth`

| Metric | Value |
|---|---:|
| S9 MPJPE | **58.15 mm** |
| S11 MPJPE | **49.87 mm** |
| Combined MPJPE | **54.01 mm** |
| Combined PA-MPJPE | **42.22 mm** |

- Trained on the corrected true-GT H36M standard protocol with v25 geometry fusion + v45 adaptive geometry fusion + v46 sparse-view generalisation + v50 self-evolution feedback head + v51 cross-domain sparse-view reliability + v52 uncertainty-weighted triangulation. v57 DC-PSC was removed so the effect of the v52 learnable triangulation module could be measured in isolation.
- **Test MPJPE: 54.01 mm** (S9 58.15 mm / S11 49.87 mm, stride 13, PA-MPJPE 42.22 mm) when evaluated with the saved EMA shadow weights. Source: `outputs/eval_v52_true_gt_h36m_test_a800.json`.
- Best validation MPJPE was **54.75 mm** @ epoch 4; early-stopped @ epoch 7. v52 is slightly behind v46 (52.46 mm) and the v80 regularisation run (53.98 mm) on this protocol, but all three are within a narrow band.

### v82 (multi-scale temporal-pose-attention, A800)

```bash
bash scripts/run_eval_v82_true_gt_h36m_test_a800.sh
```

- Log: `outputs/eval_v82_true_gt_h36m_test_a800.log`
- Checkpoint: `outputs/ablations/v82_true_gt_h36m_medium_a800.pth`

| Metric | Value |
|---|---:|
| S9 MPJPE | **42.07 mm** |
| S11 MPJPE | **36.84 mm** |
| Combined MPJPE | **39.46 mm** |
| Combined PA-MPJPE | **39.94 mm** |

- Trained on the corrected true-GT H36M standard protocol with v25 geometry fusion + deformable cross-view attention v18 + multi-scale temporal-pose-attention v82 (temporal windows=9, residual gate init=-6.0, temporal dropout=0.1) on top of triangulated 3-D poses.
- **Test MPJPE: 39.46 mm** (S9 42.07 mm / S11 36.84 mm, stride 13, PA-MPJPE 39.94 mm) when evaluated with the saved EMA shadow weights. Source: `outputs/eval_v82_true_gt_h36m_test_a800.json`.
- Per-epoch validation MPJPE:

| Epoch | val MPJPE (mm) |
|---:|---:|
| 1 | 87.13 |
| 2 | 71.86 |
| 3 | 58.39 |
| 4 | 50.64 |
| 5 | 45.96 |
| 6 | 42.12 |
| 7 | 39.92 |
| 8 (best) | **39.58** |

- Best validation MPJPE **39.58 mm** @ epoch 8; the run completed all 8 epochs and the checkpoint was saved at `outputs/ablations/v82_true_gt_h36m_medium_a800.pth`. v82 is slightly behind v81 (37.83 mm) by **~1.6 mm** combined test MPJPE, but shows monotonic improvement across all 8 epochs without overfitting.

#### v82 A800 variable-view MPJPE@k (S9 / S11)

Evaluated the saved A800 checkpoint under the variable-view protocol (`--min_views 2 --max_views 4`, `--num_subsets_per_k 50`, `clip_len 13`).

| dataset | k=2 | k=3 | k=4 |
|---|---:|---:|---:|
| S9 | **>>1,000** | **>>1,000** | **47.81** |
| S11 | **>>1,000** | **>>1,000** | **42.36** |

- Source CSV: `outputs/variable_view_v82_true_gt_medium_a800.csv`
- Source JSON: `outputs/variable_view_v82_true_gt_medium_a800.json`
- With all 4 views (k=4) the model reports **~45 mm** on the subset benchmark, comparable to the full 4-view test result of **39.46 mm**. k=2 and k=3 remain catastrophically high (thousands of mm), mirroring the v81 variable-view failure mode.

#### v85 random-view-dropout A800 variable-view MPJPE@k (S9 / S11)

v85 was the first model trained natively with random whole-view dropout (`p=0.3`, `min_views=2`) and active-view-count embedding. The no-fallback variable-view eval ran on GPU 6.

| dataset | k=2 | k=3 | k=4 |
|---|---:|---:|---:|
| S9 | **2310.27** | **1119.45** | **83.52** |
| S11 | **2308.80** | **1118.18** | **77.07** |

- Source JSON: `outputs/variable_view_v85_random_view_dropout_medium_a800_k{2,3,4}.json`
- Source CSV: `outputs/variable_view_v85_random_view_dropout_medium_a800_k{2,3,4}.csv`
- **k<4 remains catastrophic** (~1100-2300 mm), even though v85 was trained with random view dropout. The learned model alone still cannot generalise to sparse views.
- **k=4 is reasonable** (S9 83.52 / S11 77.07 mm) but worse than v82 k=4 (47.81 / 42.36 mm), suggesting the dropout regularisation hurt full-view accuracy.
- The DLT-fallback evaluation of v85 has not yet run; once available it should be compared to the v25/v81/v82 DLT-fallback numbers (S9 k=2/3/4 = 58.18/33.32/116.98 mm; S11 = 49.35/25.28/110.58 mm).

##### v85 DLT-fallback variable-view MPJPE@k (S9 / S11) — pending

| dataset | k=2 | k=3 | k=4 |
|---|---:|---:|---:|
| S9 | **TODO** | **TODO** | **TODO** |
| S11 | **TODO** | **TODO** | **TODO** |

- **Not yet run.** Expected source: `outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.{json,csv}`.
- This will reveal whether the v85 model's full-view k=4 estimate improves when combined with geometric DLT fallback for k<4, and whether its k<4 numbers match the model-agnostic v25/v81/v82 DLT-fallback baseline.

### v25 (multiview geometry fusion)

```bash
bash scripts/run_v25_h36m_true_gt_medium_local_4090.sh
```

- Log: `outputs/omniview_fusion_v25_h36m_true_gt_medium.log`
- Checkpoint: `outputs/omniview_fusion_v25_h36m_true_gt_medium.pth`

| Epoch | val MPJPE (mm) |
|---:|---:|
| 1 | 83.19 |
| 2 (best) | **72.80** |
| 3 | 80.14 |
| 4 | 94.27 |
| 5 | 113.48 |
| 6 | 139.21 |
| 7 | 174.90 |
| 8 (final) | 207.62 |

- **Test MPJPE: 43.93 mm** (combined direct, simple average of S9/S11; S9 47.28 mm, S11 40.54 mm, stride 1).
  - Source: `outputs/eval_v25_true_gt_h36m_test.json`.
- The training log above reports validation MPJPE. Best *validation* MPJPE was **72.80 mm** @ epoch 2, but this is **inflated**: validation did not pass `view_mask`, so the model was evaluated without view masking. After fixing the validation pass-through, the corrected **test** result is **43.93 mm**.
- Training completed 8 epochs before early-stopping patience was exhausted; the run began to diverge after epoch 2.
- Gap to baselines on combined direct (test):
  - **Iskakov**: +20.53 mm (43.93 vs. 23.40 mm).
  - **DLT (confidence-weighted)**: +18.26 mm (43.93 vs. 25.67 mm).
  - **DLT (unweighted)**: +15.16 mm (43.93 vs. 28.77 mm).
- v25 does not currently beat the geometric or learnable-triangulation baselines on this true-GT protocol.

#### v25 true-GT divergence ablations

Two corrected-validation runs were started on A800 after the `view_mask` fix. These use the true-GT labels and pass `view_mask` through validation, so the val MPJPE is directly comparable to the test MPJPE above.

| Run | GPU | Best val MPJPE (mm) | Best epoch | Final val MPJPE (mm) | Final epoch | Early stopped | Log / checkpoint |
|---|---:|---:|---:|---:|---:|---|---|
| `v25_true_gt_baseline_fix` | 4 | **45.80** | 1 | **323.35** | 4 | yes | `outputs/ablations/v25_true_gt_baseline_fix.{log,pth}` |
| `v25_true_gt_geometry_regularization_a800` | 6 | **46.75** | 1 | **281.22** | 4 | yes | `outputs/ablations/v25_true_gt_geometry_regularization_a800.{log,pth}` |

- Both runs already beat the original local val log of **72.80 mm** in their first epoch, confirming the missing `view_mask` was the dominant source of the inflated validation error.
- Both diverged after epoch 1, with final validation MPJPE rising to **323.35 mm** and **281.22 mm**. Early stopping fired at epoch 4 in each case.
- Neither the hyperparameter-only fix nor the added bone / joint-limit / temporal-bone losses prevent the v25 architecture from overfitting on the small true-GT H36M training set. A mixed-dataset or structural intervention is needed.

### v57 (domain-conditional physical-space calibration)

```bash
bash scripts/run_v57_h36m_true_gt_medium.sh
```

- Log: `outputs/omniview_fusion_v57_h36m_true_gt_medium.log`
- Checkpoint: `outputs/omniview_fusion_v57_h36m_true_gt_medium.pth`

| Epoch | val MPJPE (mm) |
|---:|---:|
| 1 | 98.11 |
| 2 | 81.47 |
| 3 (best) | 75.16 |
| 4 | 76.60 |
| 5 | 80.21 |

- Training completed 5 epochs before early-stopping patience was exhausted.
- Best **observed** val MPJPE: epoch 3, **combined direct MPJPE = 75.16 mm**.
- **Final reported val MPJPE: 80.21 mm** (early-stopped @ epoch 5).
- **Test MPJPE: 59.59 mm** (combined direct, simple average of S9/S11; S9 62.48 mm, S11 56.69 mm, stride 13). Note: test is substantially better than the final val (80.21 mm) because the saved checkpoint is epoch 3 (75.16 mm val) and the test distribution differs from val.
  - Source: `outputs/eval_v57_true_gt_h36m_test_local_stride13.json` (A800, GPU 6).
- **Checkpoint mismatch:** the trainer saved the checkpoint from **epoch 2 (81.47 mm)** because the `early_stopping_min_delta` threshold treated the epoch-3 `val_loss` improvement (0.000384) as insignificant, even though `val_MPJPE` dropped to 75.16 mm. The trainer has been fixed to monitor `mpjpe` for best-checkpoint selection.

#### v57 true-GT re-run (A800 GPU 5)

A fresh v57 medium run was launched with the trainer best-checkpoint bug fixed (it now monitors `mpjpe` instead of `loss`).

| Epoch | val MPJPE (mm) | Notes |
|---:|---:|---|
| 4 | **57.81** | already beats the previous lost best of 75.16 mm |
| 5 | **60.72** | still rising but remains below the old best |
| 7 | **—** | early-stopped; best epoch = 4 |

- The previous v57 run finished with a final val MPJPE of **80.21 mm** and a true best of **75.16 mm** @ epoch 3, but the saved checkpoint was stale (epoch 2, **81.47 mm**) because the trainer monitored `loss` instead of `mpjpe`.
- The re-run finished with best val **57.81 mm** @ epoch 4 and checkpoint saved correctly at `outputs/ablations/v57_true_gt_medium_a800.pth`.
- **Test MPJPE: 57.10 mm** (S9 61.09 / S11 53.11, stride 13, PA-MPJPE 37.30 mm) when evaluated with the saved EMA shadow weights. Source: `outputs/eval_v57_true_gt_h36m_test_a800.json`.

#### v57 A800 variable-view MPJPE@k (S9 / S11)

Evaluated the saved A800 re-run checkpoint under the variable-view protocol (`--min_views 2 --max_views 4`, `--num_subsets_per_k 50`, `clip_len 13`).

| dataset | k=2 | k=3 | k=4 |
|---|---:|---:|---:|
| S9 | **182.58** (std 43.07) | **148.35** (std 4.07) | **143.02** |
| S11 | **174.22** (std 40.36) | **142.24** (std 3.81) | **137.39** |

- Source CSV: `outputs/variable_view_v57_true_gt_medium_a800.csv`
- Source JSON: `outputs/variable_view_v57_true_gt_medium_a800.json`
- With all 4 views (k=4) the model is still far worse than the 4-view test MPJPE of 57.10 mm reported above, because the variable-view benchmark drops into the model subsets of the 4 H36M cameras without retraining and uses a clip-level evaluation that differs from the full-sequence test script.

### v25 true-GT stability (A800 GPU 6)

```bash
bash scripts/run_v25_true_gt_stability_a800.sh
```

- Log: `outputs/ablations/v25_true_gt_stability_a800.log`
- Checkpoint: `outputs/ablations/v25_true_gt_stability_a800.pth`

| Epoch | val MPJPE (mm) | Notes |
|---:|---:|---|
| 10 | **31.13** | best val |
| 12 | — | early-stopped (no val improvement for 3 epochs) |

- Training finished and was early-stopped @ Epoch 12; best val **31.13 mm** @ Epoch 10.
- **Test MPJPE: 30.83 mm** (S9 34.87 / S11 26.80, stride 1, PA-MPJPE 33.59 mm).
- Source test log: `outputs/eval_v25_true_gt_stability_h36m_test.log` / `.json`.

### v81 temporal-pose-attention (A800 GPU 4)

```bash
bash scripts/run_v81_true_gt_h36m_medium_a800.sh
```

- Log: `outputs/ablations/v81_true_gt_h36m_medium_a800.log`
- Checkpoint: `outputs/ablations/v81_true_gt_h36m_medium_a800.pth`
- Test eval script: `scripts/run_eval_v81_true_gt_h36m_test_a800.sh`

| Epoch | val MPJPE (mm) | Notes |
|---:|---:|---|
| 1 | 84.52 | — |
| 2 | 73.36 | — |
| 3 | 70.78 | — |
| 4 | 72.94 | — |
| 5 | 66.94 | — |
| 6 | 55.22 | — |
| 7 | 44.75 | — |
| 8 | **38.62** | best val |

- v81 adds a per-joint temporal pose-attention module to the v25 architecture. Training completed all 8 epochs without triggering early stopping (best val improved at epoch 8).
- **Test MPJPE: 37.83 mm** (S9 42.19 mm / S11 33.46 mm, stride 13, PA-MPJPE 37.75 mm) when evaluating the saved EMA shadow weights. Source: `outputs/eval_v81_true_gt_h36m_test_a800.json`.
- Best validation MPJPE was **38.62 mm** @ Epoch 8.

#### v81 A800 variable-view MPJPE@k (S9 / S11)

Evaluated the saved A800 checkpoint under the variable-view protocol (`--min_views 2 --max_views 4`, `--num_subsets_per_k 50`, `clip_len 13`).

| dataset | k=2 | k=3 | k=4 |
|---|---:|---:|---:|
| S9 | **4230.29** (std 2359.57) | **1356.67** (std 425.22) | **54.53** |
| S11 | **4258.15** (std 2427.20) | **1374.38** (std 453.25) | **47.41** |

- Source CSV: `outputs/variable_view_v81_true_gt_medium_a800.csv`
- Source JSON: `outputs/variable_view_v81_true_gt_medium_a800.json`
- Variable-view k=4 combined MPJPE is **~50.97 mm**, which is substantially worse than the full 4-view test result of **37.83 mm**, indicating the model has not generalised to the subset evaluation protocol. k=2 and k=3 are catastrophically high (>>1,000 mm), similar to the v57 variable-view failure mode.

### In-flight A800 ablations / baselines (2026-08-12)

Follow-up runs were launched on A800 after the corrected-validation `view_mask` fix and the v57 re-run. They are intended to isolate the cause of the v25 Epoch-1 divergence, probe whether regularization or mixed-dataset training improves generalisation, and validate cross-dataset (AIST++ / MPI-INF-3DHP) baselines.

| Run / baseline | GPU | Config / script | Status | Latest val MPJPE | Notes |
|---|---|---|---:|---:|---|
| `v80_true_gt_regularization_a800` | 4 | `scripts/run_v80_ablation_true_gt_regularization_a800.sh` | **completed** | **54.46 mm** @ Epoch 1 | Best val **54.46 mm** @ epoch 1, early-stopped @ epoch 4. Test MPJPE **53.98 mm** (S9 56.69 / S11 51.27, stride 13, PA-MPJPE 32.47 mm). Log: `outputs/ablations/v80_true_gt_regularization_a800.log`. |
| `v25_true_gt_mixed_dataset_a800` | 5 | `configs/splits/h36m_true_gt_aist_mixed_train_val_a800.yaml` | **completed** | **584.25 mm** @ Epoch 3 | Trained v25 on H36M + AIST++ mixed loader. Best val **34.94 mm** @ Epoch 1, then diverged. Test MPJPE **33.42 mm** (S9 37.87 / S11 28.96, stride 13, PA-MPJPE 34.60 mm). Log: `outputs/ablations/v25_true_gt_mixed_dataset_a800.log`. |
| `v25_true_gt_stability_a800` | 6 | `configs/splits/h36m_true_gt_standard.yaml` | **completed** | **31.13 mm** @ Epoch 10 | Low LR (`1e-4`), 4-epoch warmup, no `variable_view_permute`. Early-stopped @ Epoch 12. Test MPJPE **30.83 mm** (S9 34.87 / S11 26.80, stride 1, PA-MPJPE 33.59 mm). Log: `outputs/ablations/v25_true_gt_stability_a800.log`; test log: `outputs/eval_v25_true_gt_stability_h36m_test.log`. |
| `v81_true_gt_h36m_medium_a800` | 4 | `scripts/run_v81_true_gt_h36m_medium_a800.sh` | **completed** | **38.62 mm** @ Epoch 8 | v25 + per-joint temporal pose attention. Completed 8 epochs (no early stop). Test MPJPE **37.83 mm** (S9 42.19 / S11 33.46, stride 13, PA-MPJPE 37.75 mm). Log: `outputs/ablations/v81_true_gt_h36m_medium_a800.log`; test JSON: `outputs/eval_v81_true_gt_h36m_test_a800.json`. |
| `v82_true_gt_h36m_medium_a800` | 4 | `scripts/run_v82_true_gt_h36m_medium_a800.sh` | **completed** | **39.58 mm** @ Epoch 8 | v81 + multi-scale temporal-pose-attention. Completed 8 epochs. Test MPJPE **39.46 mm** (S9 42.07 / S11 36.84, stride 13, PA-MPJPE 39.94 mm). Log: `outputs/ablations/v82_true_gt_h36m_medium_a800.log`; test JSON: `outputs/eval_v82_true_gt_h36m_test_a800.json`. |
| `aistpp_only_medium_a800_fast_v2` | 5 | `configs/splits/aist_only_smoke.yaml` / `outputs/ablations/aistpp_only_medium_a800_fast_v2.*` | **completed** | **91.43 mm** @ Epoch 2 | Early-stopped @ Epoch 4. Cross-eval on H36M true-GT S9/S11: **93.94 mm** (S9 98.17 / S11 89.70, stride 1, PA-MPJPE 44.50 mm). Log: `outputs/ablations/aistpp_only_medium_a800_fast_v2.log`; test JSON: `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json`. |
| `v85_random_view_dropout_medium_a800` | 7 | `scripts/run_v85_random_view_dropout_medium_a800.sh` | **completed** | ~37.86 mm | Random view dropout (`--use_random_view_dropout_v85 --v85_dropout_prob 0.3 --v85_min_views 2 --v85_use_count_embedding`). Training finished; no-fallback variable-view eval completed on GPU 6. Log: `outputs/ablations/v85_random_view_dropout_medium_a800.log`. |
| `v81_var_view_dlt_fallback` | 6/7 | manifest eval | **in progress** | — | Per-dataset DLT-fallback eval for S9/S11; outputs: `outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback.*`. |
| `v82_var_view_dlt_fallback` | 6 | manifest eval | **in progress** | — | DLT-fallback eval on `tmp/h36m_true_gt_val_manifest.txt`; outputs: `outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.*`. |
| `v25_true_gt_stability_dlt_fallback` | 6 | manifest eval | **in progress** | — | DLT-fallback re-evaluation of v25 stability checkpoint. Output: `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.*`. |
| `eval_variable_views_v25_true_gt_stability_a800` | 4 | `scripts/run_eval_variable_views_v25_true_gt_stability_a800.sh` | **completed** | — | Variable-view evaluation of the v25 stability checkpoint on GPU 4. Outputs: `outputs/variable_view_v25_true_gt_stability_a800.csv/.json`. |
| MPI DLT baseline | 7 | `scripts/run_mpi_dlt_baseline.py` | **completed** | **115.09 mm** / PA-MPJPE **132.68 mm** | RTMPose detected-2D regeneration produced 16/16 `.npz` files; confidence-weighted DLT baseline computed on all 16 files. Source: `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`. |

- **v80 regularization** finished with best val **54.46 mm** @ epoch 1 and early-stopped @ epoch 4. Test MPJPE is **53.98 mm** (S9 56.69 / S11 51.27, stride 13, PA-MPJPE 32.47 mm). This is a clear improvement over the previous v80 medium test result of **62.32 mm**, but still well behind the v80 best val of **39.70 mm** (A800 v2) / **39.98 mm** (local medium), indicating a train/val-to-test gap.
- **v25 mixed-dataset** diverged after Epoch 1. Best val **34.94 mm** @ Epoch 1, then rose to **81.35 mm** @ Epoch 2 and **584.25 mm** @ Epoch 3, so the run was early-stopped and the best checkpoint was tested. Test MPJPE is **33.42 mm** (S9 37.87 / S11 28.96, stride 13, PA-MPJPE 34.60 mm).
- **v25 stability** finished training with best val **31.13 mm** @ Epoch 10 and was early-stopped @ Epoch 12. **Test MPJPE: 30.83 mm** (S9 34.87 / S11 26.80, stride 1, PA-MPJPE 33.59 mm). A variable-view evaluation of this checkpoint is currently running on GPU 4.
- **v81 temporal-pose-attention** completed training on GPU 4. Best val **38.62 mm** @ Epoch 8, **test MPJPE: 37.83 mm** (S9 42.19 / S11 33.46, stride 13, PA-MPJPE 37.75 mm).
- **v82 multi-scale temporal-pose-attention** completed training on GPU 4. Best val **39.58 mm** @ Epoch 8, **test MPJPE: 39.46 mm** (S9 42.07 / S11 36.84, stride 13, PA-MPJPE 39.94 mm). The variable-view evaluation on GPU 6 is complete: k=4 reports **~45 mm** (S9 47.81 / S11 42.36), while k=2/k=3 are catastrophically high (>>1,000 mm). Log: `outputs/ablations/v82_true_gt_h36m_medium_a800.log`; test JSON: `outputs/eval_v82_true_gt_h36m_test_a800.json`; var-view CSV: `outputs/variable_view_v82_true_gt_medium_a800.csv`.
- **AIST++-only fast v2** (`aistpp_only_medium_a800_fast_v2`) finished on GPU 5 and was early-stopped @ Epoch 4 with best val **91.43 mm** @ Epoch 2. Cross-eval on H36M true-GT S9/S11 gives **93.94 mm** (S9 98.17 / S11 89.70, stride 1, PA-MPJPE 44.50 mm). Log: `outputs/ablations/aistpp_only_medium_a800_fast_v2.log`; test JSON: `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json`.
- **MPI-INF-3DHP DLT baseline** is completed. All 16 detected-2D `.npz` files have been generated; the confidence-weighted DLT mean is **115.09 mm** / PA-MPJPE **132.68 mm**. Source: `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`.
- A800 disk remains **99 % full**. AIST++ canonical `.npz` files are now present on A800 (1,408 clips).

## How to reproduce

```bash
# Iskakov baseline
python experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m --epochs 10 --batch_size 8 --lr 1e-3 --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --log_path outputs/iskakov_h36m_true_gt.log \
    --ckpt_path outputs/iskakov_h36m_true_gt.pth

# v25 medium
bash scripts/run_v25_h36m_true_gt_medium_local_4090.sh

# v57 medium
bash scripts/run_v57_h36m_true_gt_medium.sh

# v80 medium
bash scripts/run_v80_h36m_true_gt_medium.sh

# v80 smoke
bash scripts/run_v80_h36m_true_gt_smoke_local_4090.sh
```

## Related docs

- `docs/results_iskakov_h36m_true_gt.md` — full Iskakov baseline report including MPJPE@k curves.
- `docs/results_v80_h36m_true_gt.md` — v80 recipe sweep and interpretation.

## Takeaways

1. **True-GT protocol is now reliable**: numbers are in the expected 15–30 mm range, unlike the old circular-label 0.62 mm.
2. **Iskakov is a strong baseline**: it beats DLT, RANSAC, and all current learned MotionFlow variants on this protocol.
3. **MotionFlow variants need re-tuning**: v80 reaches 39.98 mm at epoch 4 but then overfits (133.71 mm by epoch 8); v25 reaches **43.93 mm test** and corrected-validation ablations are at **~46.5 mm** epoch 1 (the old 72.80 mm val was inflated by a missing `view_mask`); v57 reaches a true best of 75.16 mm @ epoch 3 but its final reported val is **80.21 mm** and the saved checkpoint corresponds to epoch 2 (81.47 mm). None yet beats the geometric / learnable-triangulation baselines.
4. **Sparse-view k<4 is still unsolved**: v85, trained with random whole-view dropout, still fails catastrophically at k=2/k=3 (>>1000 mm without DLT fallback). A stronger count-conditioning mechanism or a dedicated sparse-view head is likely needed.
5. **The project now has a real leaderboard**: Iskakov / DLT / RANSAC / v80 / v25 / v57 / v85 on a non-circular H36M standard protocol.

---

## Footnotes

[^1]: **True-GT v2 protocol**: train on subjects **S1, S5, S6, S7, S8**; test on subjects **S9** and **S11**. Labels are regenerated from the official mocap release with aligned cameras and 2D projections (`data/h36m_true_gt_v2/*_multiview_m.npz`, manifest `configs/splits/h36m_true_gt_v2_standard.yaml`). Direct MJE ≈ 13–34 mm, confirming the labels are independent 3D mocap GT rather than circular DLT triangulations of the input 2D.

[^2]: The completed learned rows above reflect the best available true-GT numbers from pre-v2 or early v2 labels. Dedicated v2 re-runs (v25, v46, v52, v57, v80, v81, v82, v85, v86) are queued on A800 and will be backfilled once results are available.

[^3]: The v85 k=4 value (S9 83.52 / S11 77.07 mm) is a **variable-view k=4 subset result** taken from the no-fallback evaluation, not the standard full 4-view test. A full 4-view test and the DLT-fallback variant are pending.
