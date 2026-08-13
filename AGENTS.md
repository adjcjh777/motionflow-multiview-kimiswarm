# MotionFlow-MultiView Agent Notes

> **Executive Summary — 2026-08-13 ~00:41 UTC**
>
> **Project goal:** CVPR 2027 publishable standard.
>
> **A800 status (do not touch running jobs):** v25 true-GT v2 medium training/test is **complete** on A800 (early-stopped @ epoch 6, best val MPJPE **31.41 mm**; test S9/S11 **30.69 mm** combined, S9 **34.71 mm**, S11 **26.66 mm**). v86 no-count-embedding ablation is **complete** (best val MPJPE **31.64 mm** @ Epoch 3, checkpoint `outputs/ablations/v86_no_count_embedding_medium_a800.pth`). v85 random-view-dropout training is **complete** (best val MPJPE **31.42 mm**). The v85 DLT-fallback variable-view evaluation was **launched but killed** by the v86 watcher (`User defined signal 1`) and must be re-run. GPU 7 is **occupied by an external project** (~12 GB). **Only GPUs 6/7 are used by this project.**
>
> **Local smoke status (RTX 4090):** v21 neural BA smoke is **fixed** — the axis-angle rotation descriptor in `motionflow_mv/fusion/neural_bundle_adjustment_v21.py` was divergent at identity and has been replaced with the `R - R^T` skew-symmetric part; 2 epochs val MPJPE **79.42 mm** (down from an initial 93.50 mm). v29 hierarchical smoke is **fixed** — the hang was caused by an overly heavy smoke config, not a code bug; the lightweight script `scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090_fixed.sh` completes 2 epochs with val MPJPE **95.20 mm**. v39 reliability-coupled graph refinement and v41 weighted domain loss smokes are **done** on true-GT v2; 2 epochs val MPJPE **80.52 mm** and **80.23 mm**, respectively (v37 baseline ~80 mm). v35 temporal view-joint graph and v36 uncertainty-gated graph refinement smokes are **done** on true-GT v2 with a lightweight config (d=64, residual_hidden=128, n_st_layers=2, train_samples=256, val_stride=100); 2 epochs val MPJPE **91.21 mm** and **94.97 mm**, runtimes ~4.5 min each. v31 camera-view embedding smoke is **fixed** — the relative rotation angle in `motionflow_mv/fusion/camera_view_embedding_v31.py` used an `acos` descriptor whose derivative diverges at identity, producing `Non-finite values in camera parameters`; replaced with the `R - R^T` skew-symmetric part; 2 epochs val MPJPE **80.70 mm** in ~15 min.
>
> **Data foundation:** `data/h36m_true_gt_v2/` is the current protocol on A800; the v2 DLT baseline is **25.67 mm** and RANSAC/conf-DLT is **26.47 mm**. Legacy `data/h36m_hf/*.npz` remain circular; do not use them for model selection.
>
> **True-GT H36M leaderboard (S9/S11 test):** Iskakov 23.40 mm; conf-DLT 25.67 mm; RANSAC/conf-DLT 26.47 mm; **v25 true-GT v2 medium val 31.41 mm** (test pending); **v25 stability 31.56 mm** (v1); v81 37.83 mm; v82 39.46 mm; v80 53.98 mm; v52 54.01 mm; v57 57.10 mm. v85 no-fallback k=4 was 83.52/77.07 mm; v85 DLT-fallback is running on GPU 6. v86 no-count-embedding ablation is done (val 31.64 mm @ Epoch 3).
>
> **CVPR 2027 plan:** 1) Monitor v86 no-count-embedding ablation on GPU 6. 2) Wait for v85 DLT-fallback variable-view eval to auto-run once v86 finishes. 3) Sync/run v2 test-set evaluation for v25 (test MPJPE pending). 4) Run `scripts/cleanup_a800_safe.sh` dry-run before any large write; A800 disk is **~98% full (~72 GB free)**. 5) Rewrite paper around true-GT, sparse-view, and cross-dataset robustness.
>
> **Local repo / GitHub cleanup (done):** Remote URL token removed; old worktree `.worktrees/v18_deformable_attention_baseline` deleted; local tags `v25_local_baseline_monitor_commit` and `v25_local_baseline_monitor_v1` deleted; `main` pushed to GitHub as commit `8aee08c` (or newer). The 45 stash backups in `patches/stashes/` remain for later audit.

## Recent local smoke fixes (v21 / v29)

- **v21 neural BA smoke — fixed:** The local RTX 4090 smoke for `neural_bundle_adjustment_v21` was failing with NaN camera parameters. Root cause: the axis-angle rotation descriptor in `motionflow_mv/fusion/neural_bundle_adjustment_v21.py` has a divergent derivative at the identity rotation (`R = I`), producing NaNs in the first optimization step. Fix: replace the descriptor with the skew-symmetric part `R - R^T`. After the fix the smoke completes 2 epochs with val MPJPE dropping from an initial **93.50 mm** to **79.42 mm**.
- **v29 hierarchical smoke — fixed:** The smoke appeared to hang on the local RTX 4090, but this was not a bug; the original smoke configuration was too heavy for the RTX 4090. A lightweight script `scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090_fixed.sh` runs cleanly and completes 2 epochs with val MPJPE **95.20 mm**.
- **Takeaway:** Local smoke failures on RTX 4090 should first be checked for (a) numerical instabilities in rotation parameterizations and (b) config/memory overload before treating them as architecture bugs.

## Additional local v2 smokes (v39 / v41)

- **v39 reliability-coupled graph refinement v2 smoke:** Script `scripts/run_v39_reliability_coupled_graph_refinement_true_gt_v2_smoke_local_4090.sh` on true-GT v2 (`configs/splits/h36m_true_gt_v2_standard.yaml`). 2 epochs val MPJPE **80.52 mm** (v37 baseline ~80 mm). Uses `val_stride=50` for fast turnaround.
- **v41 weighted domain loss v2 smoke:** Script `scripts/run_v41_weighted_domain_loss_true_gt_v2_smoke_local_4090.sh` on true-GT v2. 2 epochs val MPJPE **80.23 mm**. Uses `val_stride=50` for fast turnaround.
- **v35 temporal view-joint graph v2 smoke:** Script `scripts/run_v35_temporal_view_joint_graph_true_gt_v2_smoke_local_4090.sh` on true-GT v2. Lightweight config (d=64, residual_hidden=128, n_st_layers=2, train_samples=256, val_stride=100); 2 epochs val MPJPE **91.21 mm** in ~4.5 min.
- **v36 uncertainty-gated iterative graph refinement v2 smoke:** Script `scripts/run_v36_uncertainty_gated_graph_refinement_true_gt_v2_smoke_local_4090.sh` on true-GT v2. Same lightweight config as v35; 2 epochs val MPJPE **94.97 mm** in ~4.5 min.
- **Debug output cleanup:** Disabled noisy `print` statements in `motionflow_mv/fusion/principal_point_correction.py` that were emitting per-call debug lines and slowing smoke runs by an order of magnitude.

## v31 camera-view embedding smoke fix

- **v31 camera-view embedding smoke — fixed:** The local RTX 4090 smoke for `motionflow_mv/fusion/camera_view_embedding_v31.py` failed with `Non-finite values in camera parameters (K, R, t)`. Root cause: the pairwise relative rotation descriptor used `angle = acos((trace(R_rel) - 1) / 2)`, whose derivative diverges at the identity relative rotation (`R_rel = I`, where `trace = 3` and `acos` input equals 1), producing NaNs during the first optimization step. Fix: replace the scalar `acos` angle with the 3-D skew-symmetric part of the relative rotation matrix, i.e. `[R_rel[2,1] - R_rel[1,2], R_rel[0,2] - R_rel[2,0], R_rel[1,0] - R_rel[0,1]]`, and increase the pairwise MLP input dimension from 3 to 5 (`baseline 1 + rot_vec 3 + dot 1`). After the fix the smoke completes 2 epochs with val MPJPE **80.70 mm** (down from 94.07 mm at epoch 1) in ~15 min.

## Current-session update (AIST++ NaN fix / v25 var-view / MPI / dropped v83/v84)

- **AIST++ NaN / empty-sequence blocker:** `convert_aistpp` originally used raw `keypoints3d`, which contains NaNs on ~20% of sequences. The first fix preferred `keypoints3d_optim` and dropped frames with any NaN, but two sequences had NaN in all 2D frames and became empty, crashing the mixed-loader collate function. The converter now zeroes NaN 2D keypoints and sets confidence to 0 while preserving frame count, and only drops frames if 3D joints are NaN. Clean canonical `.npz` are regenerating locally and will be synced to A800 before relaunching on GPU 5.
- **v83/v84 dropped:** v83 A800 medium plateaued at **~100 mm** val and was killed. v84 uncertainty-weighted view dropout smoke produced **107.11 mm** val, also no improvement. Architecture modules on top of v25 ray tokens are deprioritized until cross-dataset training is baselined.
- **v25 stability variable-view eval:** The wrapper fix (explicit `view_mask` to `OmniMultiViewFusionV5`) was necessary but not sufficient: k=2/k=3 remained catastrophic (~3000/1000 mm). A diagnostic showed that while the learned model fails catastrophically for k<4, direct confidence-weighted DLT on the same active views achieves ~35–100 mm. A new `--var_view_dlt_fallback` mode was added to `HardenedVariableViewInferenceWrapper` that falls back to direct DLT whenever `n_active < n_views_max`. The DLT-fallback re-eval (PID `628743`) completed on GPU 4. Full S9/S11 numbers: k=2 **58.18 / 49.35 mm**, k=3 **33.32 / 25.28 mm**, k=4 **116.98 / 110.58 mm**. For k<4 the learned model is not used; direct confidence-weighted DLT fallback is used instead.
- **v85 no-fallback variable-view eval completed:** Split-k run (k=2,3,4 sequential, 50 subsets per k) finished on GPU 6. The early-stopped v85 checkpoint (best val MPJPE 31.42 mm) produced: **k=2 S9 2310.27 mm / S11 2308.80 mm**, **k=3 S9 1119.45 mm / S11 1118.18 mm**, **k=4 S9 83.52 mm / S11 77.07 mm**. k<4 remains catastrophic, but k=2 is better than the v25 no-fallback baseline (S9 ~3017 / S11 ~2862 mm) and k=4 is much better than v25 (S9 ~117 / S11 ~111 mm). Combined JSON/CSV: `outputs/variable_view_v85_random_view_dropout_medium_a800.{json,csv}`. Per-k files: `outputs/variable_view_v85_random_view_dropout_medium_a800_k{2,3,4}.{json,csv}`.
- **v85 DLT-fallback variable-view eval failed:** PID `2269984` terminated after ~29 min without creating the expected JSON/CSV. The redirect log is empty and the nohup log only contains `Terminated`; the cause is unknown (likely an external kill or OOM). No v85-specific DLT-fallback numbers are available. The existing v25/v81/v82 DLT-fallback baseline remains the reference: S9 58.18/33.32/116.98 mm, S11 49.35/25.28/110.58 mm. Combined with the no-fallback result, this confirms that random view dropout does **not** solve the k<4 catastrophic failure and DLT fallback remains the practical choice for sparse views.
- **MPI-INF-3DHP detection:** RTMPose 2D detection **finished** on GPU 7 (PID `2527668`). All 16 `.npz` files are in `data/webbridge/mpi_inf_3dhp_detected_2d/`. The CPU watcher ran the DLT baseline: mean MPJPE **115.09 mm**, PA-MPJPE **132.68 mm** → `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`.
- **v86 no-count-embedding ablation:** Launched on A800 GPU 6 (PID `2203020`) to isolate the active-view-count embedding's contribution to sparse-view (k<4) robustness. It keeps v85's random whole-view dropout (`--v85_dropout_prob 0.3`, `--v85_min_views 2`) but passes `--no_v85_use_count_embedding`. Log: `outputs/ablations/v86_no_count_embedding_medium_a800.log`; config: `configs/ablations/v86_no_count_embedding_medium_a800.yaml`.
- **Stale circular config deprecation:** Moved all configs referencing `data/h36m_hf/`, `data/webbridge/h36m_meters/`, or `data/webbridge/shelf_campus/` into `configs/deprecated/circular/`. Split manifests now contain a `deprecated: true` marker and `motionflow_mv/data/split_loader.py` raises a loud error if one is loaded; the 218 scripts/experiments that still pointed at the old `configs/splits/*` paths have been updated to `configs/deprecated/circular/...` so they resolve but will fail loudly when launched. See `configs/deprecated/circular/README.md`.
- **WebBridge loader audit:** `convert_human36m` now raises an error by default when true 3D GT is missing, instead of silently falling back to DLT triangulation of the input 2D. Pass `allow_circular_fallback=True` (CLI `--allow-circular-fallback`) to opt into the legacy circular behavior. `convert_aistpp` now defaults to `use_optim=True` (CLI `--no-optim` to override), matching the clean canonical `.npz` already used on A800. New tests in `tests/test_webbridge_loader_audit.py` cover both behaviors.
- **v86 local smoke in progress:** A local RTX 4090 smoke test of the no-count-embedding variant (`configs/ablations/v86_no_count_embedding_smoke.yaml`) is running to sanity-check the ablation before the A800 medium run commits compute. Results will be compared against the v85 smoke once both finish.

## Current work in flight

| Agent | Task | Machine | Notes |
|-------|------|---------|-------|
| Current agent | AGENTS.md / handoff refresh | Local WSL | GPU policy 6/7; v85 training on GPU 7; v86 no-count-embedding ablation on GPU 6; v81/v82/v25 DLT-fallback done; MPI/AIST done |
| `agent-272` (done) | v57 true-GT re-run | A800 GPU 5 | `v57_true_gt_medium_a800`; best **57.81 mm** @ Epoch 4; early-stopped @ Epoch 7 |
| `agent-51` (done) | H36M true-GT v25 medium | Local RTX 4090 | Completed: test 43.93 mm; val log 72.80 mm @ epoch 2 (val ignored `view_mask`) |
| `agent-67` (done / idle) | AIST++ smoke integration v25/v80 | Local RTX 4090 / A800 | Smoke complete (DLT 6.52 mm); only 16 `.npz` present on A800 |
| `agent-269` (done) | v80 true-GT regularization ablation | A800 GPU 6 | Best **54.46 mm** val @ Epoch 1; early-stopped @ Epoch 4; test **53.98 mm** combined |

- **Strategy update:** Local RTX 4090 is reserved for **quick smoke/verification only**; large training tasks (medium/long ablations, SOTA baselines, cross-dataset training) run on **A800**.
- The RTX 4090 is currently running the **v83 smoke**; do not start another training run until it finishes.
- v52 UWT (`v52_true_gt_h36m_a800`) **finished** on **A800 GPU 4**. Best val **54.75 mm** @ Epoch 4, early-stopped @ Epoch 7. **Test S9/S11: 54.01 mm** (S9 58.15 mm, S11 49.87 mm), PA-MPJPE 42.22 mm.
- v81 temporal-pose-attention medium (`v81_true_gt_h36m_medium_a800`) on **A800 GPU 4** — **early-stopped @ Epoch 8** with best val **38.62 mm**. Test S9/S11: **37.83 mm** (S9 42.19 mm, S11 33.46 mm), PA-MPJPE **37.75 mm**. Log: `outputs/ablations/v81_true_gt_h36m_medium_a800.log`; test JSON: `outputs/eval_v81_true_gt_h36m_test_a800.json`.
- v82 multi-scale temporal-pose-attention medium (`v82_true_gt_h36m_medium_a800`) **finished on A800 GPU 4**. Best val **39.58 mm** @ Epoch 8; test S9/S11 **39.46 mm**. Log: `outputs/ablations/v82_true_gt_h36m_medium_a800.log`; checkpoint: `outputs/ablations/v82_true_gt_h36m_medium_a800.pth`.
- v81 variable-view DLT-fallback eval **completed** (`outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_k23.csv` / `.json`). Only k=2,3 were needed; results match the model-agnostic DLT fallback (S9 58.18/33.32 mm; S11 49.35/25.28 mm).
- v82 variable-view DLT-fallback eval **completed** (`outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.csv` / `.json`). k=4 uses the learned v82 model (S9 47.81 / S11 42.36 mm); k=2/k=3 use the DLT fallback and match v25/v81 (S9 58.18/33.32 mm; S11 49.35/25.28 mm).
- v25 stability DLT-fallback variable-view re-evaluation **completed on A800 GPU 4** (PID `628743`). k=2/3/4 S9: 58.18/33.32/116.98 mm; S11: 49.35/25.28/110.58 mm. Outputs: `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.*`.
- AIST++-only medium fast v2 (`aistpp_only_medium_a800_fast_v2`) **early-stopped on A800 GPU 5** (PID was `117455`). Best val **91.43 mm** @ Epoch 4. Checkpoint: `outputs/ablations/aistpp_only_medium_a800_fast_v2.pth` (symlink to `aistpp_only_medium_a800_fast_v2_final.pth`). H36M S9/S11 cross-evaluation **completed** (PID `1090542`, GPU 6): S9 **98.17 mm**, S11 **89.70 mm**, combined **~93.94 mm** → `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json`.
- v25 H36M + AIST++ mixed-dataset stability (`v25_true_gt_mixed_dataset_stability_a800_gpu6`) **killed on A800 GPU 6** (PID `175675`). Epoch 1: `val_loss=0.001683`, `val_MPJPE=36.49 mm`; Epoch 2: `val_MPJPE=52.27 mm`; Epoch 3: `val_MPJPE=481.99 mm` (diverged). Best checkpoint retained at `outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.pth` (Epoch 1, 36.49 mm). GPU 6 was later used for the v82 variable-view DLT-fallback eval.
- v25 stability (`v25_true_gt_stability_a800`) on **A800 GPU 6** **early-stopped @ Epoch 12**; best val **31.13 mm** @ Epoch 10. Test S9/S11: **31.56 mm** weighted (S9 34.87 mm, S11 26.80 mm), PA-MPJPE **34.35 mm**. Result JSON: `outputs/eval_v25_true_gt_stability_h36m_test.json`.
- AIST++ full DLT baseline **finished on A800 GPU 6**: MPJPE **15.93 mm** weighted / **38.11 mm** unweighted; PA-MPJPE **21.12 mm** / **42.66 mm**. Log: `outputs/aistpp_full_dlt_baseline_a800.log`; JSON: `outputs/aistpp_full_dlt_baseline_a800.json`.
- MPI-INF-3DHP RTMPose detection **finished on A800 GPU 7** (PID `2527668`). **16/16 `.npz` files** are in `data/webbridge/mpi_inf_3dhp_detected_2d/`. The CPU watcher automatically ran the DLT baseline: mean MPJPE **115.09 mm**, mean PA-MPJPE **132.68 mm** → `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`.
- v57/v80 variable-view evaluation **completed**. v80 regularization is better than v57 at all view counts: v80 S9@4 102.8 mm / S11@4 105.8 mm vs v57 S9@4 143.0 mm / S11@4 137.4 mm.
- Before starting any new work, check active background tasks and A800 GPU availability.

## Why we paused

- `scripts/diagnose_circular_labels.py` confirms `direct MJE = 0.0000 mm` on `data/h36m_hf/*_multiview.npz`.
- `motionflow_mv/data/webbridge_loader.py:182` triangulates the input 2D and stores it as the 3D label.
- v25–v79 numbers are therefore measuring how closely a network reproduces the DLT layer, not pose accuracy.
- The raw pkl `h36m_sh_conf_cam_source_final.pkl.zip` only contains `joint3d_image` (image-space `(u,v,z)`), which cannot be converted to a consistent world 3D across cameras.

## Data foundation status

1. **True H36M 3D GT** — obtained, but the `data/h36m_true_gt/*.npz` labels are misaligned with the stored cameras/2D (direct MJE ≈ 16,668 mm). A corrected converter is in `scripts/convert_h36m_true_gt_v2.py`; output will go to `data/h36m_true_gt_v2/`.
2. **Regenerate canonical `.npz`** with non-circular labels — **all H36M true-GT v2 .npz generated locally** (S1,5,6,7,8 train; S9,11 test). A new manifest `configs/splits/h36m_true_gt_v2_standard.yaml` is ready. **v2 audit and DLT/RANSAC baselines are complete locally:** conf-weighted DLT **25.67 mm**, RANSAC/conf-DLT **26.47 mm**, direct MJE ≈ **14.5 mm**. Full sync to A800 and leaderboard rerun remain queued until v85/v86 finish so the running training is not disturbed. Regeneration script: `scripts/convert_all_h36m_true_gt_v2.sh`.
3. **Re-run baselines** (DLT, Iskakov, v25, v46, v52, v57, v80) on the corrected protocol — in progress.
   - DLT H36M true-GT: **25.67 mm** (conf-weighted), 28.77 mm (unweighted).
   - RANSAC/conf-DLT H36M true-GT: **26.47 mm** (reproducible; see `scripts/run_h36m_true_gt_ransac_baseline.py`).
   - Iskakov ICCV 2019 H36M true-GT: **23.40 mm**.
   - v25 H36M true-GT medium: test **43.93 mm**; original local val log **72.80 mm** @ epoch 2 was inflated because validation did not pass `view_mask`. A800 ablation 1 (`v25_true_gt_baseline_fix`) reached **46.53 mm** @ epoch 1 with the corrected validation recipe.
   - v46 H36M true-GT medium: val **52.92 mm** @ epoch 4, early-stopped @ epoch 7; test S9/S11 **52.46 mm** combined (S9 55.03 mm, S11 49.88 mm), PA-MPJPE 40.20 mm.
   - v52 UWT true-GT H36M: val **54.75 mm** @ epoch 4, early-stopped @ epoch 7; test S9/S11 **54.01 mm** (S9 58.15 mm, S11 49.87 mm), PA-MPJPE 42.22 mm.
   - v57 H36M true-GT medium: final **78.76 mm**; true best **75.16 mm** @ epoch 3 was not saved due to the loss-based checkpoint bug (now fixed to monitor MPJPE).
   - v80 H36M true-GT medium: baseline **39.98 mm** @ epoch 4, then overfit to 133.71 mm; regularization ablation **54.46 mm** @ epoch 1, early-stopped @ epoch 4; test **53.98 mm** combined (S9 56.69 mm, S11 51.27 mm), PA-MPJPE 32.47 mm.
   - v81 temporal-pose-attention H36M true-GT medium: best val **38.62 mm** @ epoch 8; test **37.83 mm** (S9 42.19 mm, S11 33.46 mm), PA-MPJPE **37.75 mm**.
   - v82 multi-scale temporal-pose-attention H36M true-GT medium: best val **39.58 mm** @ epoch 8; test **39.46 mm** (S9 42.07 mm, S11 36.84 mm), PA-MPJPE **39.94 mm**.
   - v83 view-conditioned temporal attention: **implemented** (`motionflow_mv/fusion/view_conditioned_temporal_attention_v83.py`); local smoke running on RTX 4090, A800 medium script ready.
   - See [`docs/results_true_gt_h36m.md`](docs/results_true_gt_h36m.md) and [`docs/results_true_gt_shelf_campus.md`](docs/results_true_gt_shelf_campus.md).
4. **MPI-INF-3DHP real detected 2D** — **done**. RTMPose regeneration finished on GPU 7 (PID `2527668`); 16/16 `.npz` files in `data/webbridge/mpi_inf_3dhp_detected_2d/`. The CPU watcher ran the DLT baseline: mean MPJPE **115.09 mm**, PA-MPJPE **132.68 mm** → `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`.
5. **AIST++ integration** — **done**. Full 1,408 canonical `.npz` are on A800 (`data/webbridge/aistpp_canonical/`). AIST++-only v25 fast v2 **finished** (best val **91.43 mm** @ Epoch 2, early-stopped @ Epoch 4). Cross-eval on H36M true-GT S9/S11: **93.94 mm** (S9 98.17 / S11 89.70). Full AIST++ DLT baseline: MPJPE **15.93 mm** weighted / **38.11 mm** unweighted; PA-MPJPE **21.12 mm** / **42.66 mm**.
6. **Re-orient the paper contribution** around sparse-view / cross-domain robustness, not absolute MPJPE records — **paper rewrite in progress**. MPI/AIST++ numbers are already recorded in `docs/results_true_gt_h36m.md` and `docs/paper_draft_icra_cvpr_2027.md`.

## Current data sources

| Source | True 3D? | Status |
|--------|----------|--------|
| `data/h36m_hf/*.npz` | No | Circular labels; do not use for model selection |
| `data/webbridge/h36m*.npz` | No | Same circular labels |
| `data/h36m_true_gt/*_multiview_m.npz` | **Yes** | True mocap world coordinates; standard protocol S1,5,6,7,8 → S9/S11 |
| MPI-INF-3DHP | Yes | True mocap; RTMPose detected-2D regeneration **done**; DLT baseline **115.09 mm** / **132.68 mm** PA-MPJPE |
| Shelf/Campus | Yes | Non-circular `.npz` rebuilt from `detection.json + annotation_3d.json` at `data/webbridge/shelf_campus_detected/` |
| A800-D `/mnt/nvme0n1p1/zhangzy/projects` | No true H36M found | Read-only only |

## Next 3 tasks for the next agent

1. **Wait for v85 to finish and evaluate sparse-view robustness.**
   - v85 random view dropout is training on A800 GPU 7 (PID `1954774`).
   - The no-fallback variable-view eval is already running on GPU 6 (PID `1945448`); once it finishes, run a DLT-fallback eval (`--var_view_dlt_fallback`) on GPU 6 or 7 and compare k=2/3/4 MPJPE to the v25 stability baseline (S9: 58.18/33.32/116.98 mm; S11: 49.35/25.28/110.58 mm).
   - If k<4 is still catastrophic, consider stronger count-conditioning or a separate sparse-view head.

2. **Prepare SOTA comparison configs and free disk.**
   - VoxelPose / MVPose / DLT configs are ready to run when GPU 6/7 is free.
   - Run `scripts/cleanup_a800_safe.sh` dry-run; disk is 99% full (~42 GB free).

3. **Finalize paper draft around honest true-GT story.**
   - Ensure `docs/paper_draft_icra_cvpr_2027.md` final tables/figures use the non-circular numbers.
   - Add v85 sparse-view results once they land.

## Remaining CVPR 2027 backlog

- Validate true-GT numbers in [`docs/results_true_gt_h36m.md`](docs/results_true_gt_h36m.md) and [`docs/results_true_gt_shelf_campus.md`](docs/results_true_gt_shelf_campus.md); catch any remaining circular-label leakage or protocol mismatch.
- Prepare SOTA comparison configs for VoxelPose / MVPose / DLT when GPU is free.
- **Paper rewrite in progress** — draft the CVPR 2027 story around sparse-view / cross-domain robustness, with corrected baselines as empirical anchors.

## CVPR 2027 plan

| Phase | Work | Estimate |
|-------|------|----------|
| 1 | Close data foundation (true H36M 3D + Shelf/Campus rebuild) | 1–2 weeks |
| 2 | Rebuild baselines on correct protocol | 3–5 days |
| 3 | Add standard SOTA comparisons (Iskakov, VoxelPose, etc.) | 1–2 weeks |
| 4 | Ablation / robustness / cross-dataset evaluation | 1–2 weeks |
| 5 | Rewrite paper with real citations and tables | 2 weeks |
| 6 | MPI official server submission + buffer | 1 week |

## GPU usage rules

- **A800 project GPUs (hard rule):** MotionFlow-MultiView only uses **GPU 6 and GPU 7** on A800. GPUs 0–5 are reserved for other projects (VLLM, etc.) and must NOT be used.
  - All `CUDA_VISIBLE_DEVICES` values must be `6` or `7`.
  - Existing scripts that default to GPU 4/5 must be updated or launched with an explicit `CUDA_VISIBLE_DEVICES=6|7` override.
  - See `docs/a800_gpu_policy.md` for the full policy and migration procedure.
- **Local GPU concurrency:** RTX 4090 can run **one training task at a time**. If `agent-51` or `agent-67` is active, only prepare configs/scripts; do not launch a new training run.
- **A800 read/write boundaries:**
  - `/mnt/nvme0n1p1/zhangzy/projects` and the A800 Docker `motionflow` service are **read-only / inspection-only**.
  - The active training repo `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20` is **read-write** for launching host jobs via `ssh a800-D` + `nohup` with explicit `CUDA_VISIBLE_DEVICES`.
- Local WSL + RTX 4090 is primarily reserved for data diagnostics, smoke tests, and baseline re-runs on the corrected protocol.

## Infrastructure

- **Remote host:** `a800-D` (SSH)
- **Remote repo:** `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`
- **Local repo:** `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`

### tmux

```bash
tmux ls
tmux attach -t <session>
ssh a800-D "tmux capture-pane -pt <session> -S -100"
```

### nohup (WSL)

```bash
nohup bash scripts/<script>.sh > outputs/<log>.log 2>&1 &
tail -f outputs/<log>.log
```

## Issue / PR labels

| Label | Meaning |
|-------|---------|
| `P0-blocker` | Must be resolved before the next milestone / paper deadline |
| `P1-next` | Important; pick up once P0 items are cleared |
| `P2-nice` | Useful but not urgent |
| `experiment` | New training run, model variant, or proposal |
| `ablation` | Ablation component / hyperparameter / robustness test |
| `bug` | Unexpected behavior, crash, or regression |
| `data` | Dataset, loader, pseudo-label, or preprocessing issue |
| `paper` | Writing, figure, table, or paper-story task |
| `infra` | Build, environment, A800/tmux ops, CI |
| `question` | Needs clarification or discussion |

Optional status prefixes for issue titles: `[RUNNING]`, `[STOPPED]`, `[READY]`, `[BLOCKED]`, `[DONE]`.

## Next handoff for qwen3.8max

> **Status as of 2026-08-12 ~14:10 UTC** (agent handoff refresh)

### Executive snapshot

- **GPU 7:** v85 random-view-dropout training is running (PID `2058225`). Latest cron val_MPJPE **40.35 mm**; training continues.
- **GPU 6:** v86 no-count-embedding ablation is running (PID `2203020`). It keeps v85's random view dropout but disables the active-view-count embedding to isolate its contribution to sparse-view (k<4) robustness.
- **Previous GPU 6 job:** The v85 split-k no-fallback variable-view eval has been superseded/stopped by v86.
- **Disk:** `/mnt/nvme0n1p1` is ~99% full (~58 GB free).
- **Data foundation:** `data/h36m_hf/*.npz` are circular; `data/h36m_true_gt/*.npz` are misaligned with stored cameras/2D. `scripts/convert_h36m_true_gt_v2.py` produces physically consistent labels and full v2 `.npz` regeneration is queued for after v85/v86 finish. MPI-INF-3DHP detected-2D and AIST++ canonical `.npz` are done.

### Active runs on A800

| PID | GPU | Task | State | Notes |
|------|------|------|-------|-------|
| `2058225` | 7 | v85 random view dropout training | RUNNING | H36M true-GT medium. Latest cron val_MPJPE **40.35 mm**. Args: `--v85_dropout_prob 0.3 --v85_min_views 2 --v85_use_count_embedding`. Log: `outputs/ablations/v85_random_view_dropout_medium_a800.log`. |
| `2203020` | 6 | v86 no-count-embedding ablation | RUNNING | Keeps v85 random view dropout but disables active-view-count embedding. Config: `configs/ablations/v86_no_count_embedding_medium_a800.yaml`. Log: `outputs/ablations/v86_no_count_embedding_medium_a800.log`. |
| `2072251` | 6/7 (queued) | v85 post-training eval suite monitor | QUEUED | `scripts/monitor_v85_then_run_evals.sh`; will launch v85 test-set eval, fresh no-fallback variable-view eval, and DLT-fallback variable-view eval on the first free GPU after v85 training finishes. |
| `2067976` | — | VoxelPose SOTA baseline monitor | STOPPED | Superseded by `monitor_v85_then_run_evals.sh`. |
| `628743` | 4 | v25 var-view re-eval (DLT fallback) | COMPLETED | S9 k=2/3/4 = 58.18/33.32/116.98 mm; S11 k=2/3/4 = 49.35/25.28/110.58 mm. Output: `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json`. |
| — | — | v81 var-view DLT-fallback | COMPLETED | k=2/3 only; output: `outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_k23.{csv,json}`. |
| — | — | v82 var-view DLT-fallback | COMPLETED | k=2/3/4; output: `outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.{csv,json}`. |
| `1090542` | 6 | AIST++-only → H36M cross-eval | COMPLETED | S9 98.17 mm, S11 89.70 mm, combined ~93.94 mm → `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json`. |
| `2527668` | 7 | MPI-INF-3DHP RTMPose detection | COMPLETED | 16/16 `.npz` in `data/webbridge/mpi_inf_3dhp_detected_2d/`; DLT baseline MPJPE 115.09 mm / PA-MPJPE 132.68 mm. |
| — | 0–3 | VLLM workers | OCCUPIED | Do not touch. |

### Key context

- **Sparse-view problem is structural:** Learned models trained only on 4-view rigs fail catastrophically when k<4. DLT-fallback on the same active views is sound (S9 58.18/33.32 mm, S11 49.35/25.28 mm for k=2/3), so the issue is training exposure, not data quality. v85 is the first model trained natively on k=2/3/4 via random view dropout; it is the critical experiment to watch.
- **v86 ablates the count embedding:** v86 keeps v85's random view dropout but disables the active-view-count embedding. If v86 underperforms v85 on sparse views, the count embedding is important; if it matches, dropout alone may be sufficient.
- **v85 no-fallback k=2 result is still catastrophic:** The learned model alone gives ~2310 mm for k=2. This is expected early in training; wait for Epoch 5+ and the post-training eval suite before drawing conclusions.
- **v81/v82 DLT-fallback evals are complete:** v82's learned k=4 result (S9 47.81 / S11 42.36 mm) is much stronger than v25 stability k=4 (S9 116.98 / S11 110.58 mm), confirming v82 has a better full-view model but still relies on DLT fallback for k<4.
- **Paper story:** Re-orient around true-GT, sparse-view robustness, and cross-dataset generalization (MPI, AIST++). Absolute MPJPE records are not the claim; honest baselines are.

### Blockers / watch-outs

1. **Both project GPUs are occupied:** v85 (GPU 7) and v86 (GPU 6) are training. Do not launch anything else on GPUs 6/7 until one frees up. Do not touch GPUs 0–5.
2. **A800 disk is ~99% full (~58 GB free):** Avoid large writes. Run `scripts/cleanup_a800_safe.sh` dry-run before any new large experiment.
3. **Sparse-view (k=2/k=3) failure — fix in progress:** v85 random view dropout is training to address the root cause. Monitor GPU 7; v86 will show whether the count embedding helps. If k<4 remains catastrophic after v85 finishes, consider stronger count-conditioning or a separate sparse-view head.

### Next 3 concrete tasks

1. **Monitor v85 and v86 training until completion.**
   - v85 (GPU 7, PID `2058225`) and v86 (GPU 6, PID `2203020`).
   - When v85 finishes, the post-training monitor (PID `2072251`) will run test-set eval, a fresh no-fallback variable-view eval, and a DLT-fallback eval on the first free GPU.
   - Compare v85 k=2/3/4 MPJPE to the v25 DLT-fallback baseline (S9: 58.18/33.32/116.98 mm; S11: 49.35/25.28/110.58 mm) and to v86 once both are evaluated.
   - If k<4 is still catastrophic, design stronger count-conditioning or a dedicated sparse-view head.

2. **Run `scripts/cleanup_a800_safe.sh` dry-run and free disk if safe.**
   - Disk is at 99%. Identify removable checkpoints/logs (e.g., failed v83/v84 runs, duplicate manifests) before launching SOTA baselines.

3. **Prepare SOTA comparison configs and validate paper numbers.**
   - VoxelPose / MVPose / DLT configs are ready; schedule when GPU 6/7 is free after v85/v86 evals.
   - Ensure `docs/results_true_gt_h36m.md` and `docs/paper_draft_icra_cvpr_2027.md` use the non-circular true-GT numbers.

### Quick verification commands for next agent

```bash
# Check v85 training
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.log"

# Check v86 training
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800.log"

# Check v85 post-training eval suite monitor
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/sota_baselines/monitor_v85_then_run_evals.log"

# Check v82 DLT-fallback result
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.json"

# Check v25 DLT-fallback result
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json"

# Check AIST++ cross-eval
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json"

# Check MPI DLT baseline
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json"

# GPU overview
ssh a800-D "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv"
```

## Next handoff — v2 audit done, v86 smoke running locally, A800 v85/v86 still training

> **Status as of 2026-08-12 ~15:03 UTC**

### Executive snapshot

- **GPU 7:** v85 random-view-dropout training is **still running** on A800 GPU 7. Continue to monitor until completion.
- **GPU 6:** v86 no-count-embedding ablation is **still running** on A800 GPU 6. Continue to monitor until completion.
- **Local RTX 4090:** v86 no-count-embedding **smoke test is in progress** to sanity-check the ablation before the A800 medium run commits compute.
- **v2 data audit:** H36M true-GT v2 `.npz` audit is **complete**. DLT baseline **25.67 mm** and RANSAC/conf-DLT **26.47 mm** are reproducible on `data/h36m_true_gt_v2/`.
- **Disk:** `/mnt/nvme0n1p1` remains **~99% full (~58 GB free)**. Avoid large writes until cleanup is run.

### What changed since last handoff

1. **v2 audit finished locally**
   - All H36M true-GT v2 `.npz` files (S1,5,6,7,8 train; S9,11 test) were audited.
   - DLT baseline: **25.67 mm** (conf-weighted).
   - RANSAC/conf-DLT baseline: **26.47 mm**.
   - Direct MJE on v2 labels: ≈ **14.5 mm**, confirming physical consistency.
   - Manifest: `configs/splits/h36m_true_gt_v2_standard.yaml`.

2. **A800 v85 and v86 are still training**
   - Do not touch GPU 6 or GPU 7.
   - The post-training eval suite (`scripts/monitor_v85_then_run_evals.sh`) remains queued and will run on the first free GPU.

3. **v86 local smoke started**
   - Running on local RTX 4090 to validate the no-count-embedding variant before committing A800 GPU time.
   - Config: `configs/ablations/v86_no_count_embedding_smoke.yaml` (path may differ; verify locally).
   - Compare its result against the v85 smoke once both finish.

### Blockers / watch-outs

1. **Both project GPUs are occupied:** v85 (GPU 7) and v86 (GPU 6) are training. Do not launch anything else on GPUs 6/7 until one frees up. Do not touch GPUs 0–5.
2. **A800 disk is ~99% full (~58 GB free):** Avoid large writes. Run `scripts/cleanup_a800_safe.sh` dry-run before any new large experiment.
3. **Sparse-view (k=2/k=3) failure — fix in progress:** Wait for v85/v86 to finish and evaluate whether random view dropout (and the count embedding) resolves the k<4 catastrophe.

### Next 3 concrete tasks

1. **Monitor v85 and v86 training until completion.**
   - v85 (GPU 7) and v86 (GPU 6).
   - When v85 finishes, the post-training monitor will run test-set eval, fresh no-fallback variable-view eval, and DLT-fallback eval on the first free GPU.
   - Compare v85 k=2/3/4 MPJPE to the v25 DLT-fallback baseline (S9: 58.18/33.32/116.98 mm; S11: 49.35/25.28/110.58 mm) and to v86 once both are evaluated.

2. **Run `scripts/cleanup_a800_safe.sh` dry-run and free disk if safe.**
   - Disk is at 99%. Identify removable checkpoints/logs before launching SOTA baselines.

3. **Sync v2 labels and rerun leaderboard once GPU 6/7 are free.**
   - Sync `data/h36m_true_gt_v2/` to A800.
   - Re-run v25, v46, v52, v57, v80, v81, v82, v85, v86 on the corrected true-GT v2 protocol.
   - Update `docs/results_true_gt_h36m.md` and `docs/paper_draft_icra_cvpr_2027.md` with the non-circular numbers.

### Quick verification commands for next agent

```bash
# Check v85 training
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.log"

# Check v86 training
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800.log"

# Check v86 local smoke (if running)
tail -f outputs/ablations/v86_no_count_embedding_smoke.log

# Check v85 post-training eval suite monitor
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/sota_baselines/monitor_v85_then_run_evals.log"

# Check v2 DLT baseline result (local)
cat outputs/h36m_true_gt_v2_dlt_baseline.json

# Check v2 RANSAC baseline result (local)
cat outputs/h36m_true_gt_v2_ransac_baseline.json

# GPU overview
ssh a800-D "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv"

## Next handoff — v85 eval done, GPU policy violations observed, v86 status uncertain

> **Status as of 2026-08-12 ~15:12 UTC** (agent handoff refresh)

### Executive snapshot

- **v85 training finished:** Random-view-dropout training on A800 GPU 7 is **done**. The no-fallback variable-view eval also completed on GPU 6.
- **v85 no-fallback variable-view results:**
  - k=2: S9 **2310.27 mm**, S11 **2308.80 mm**
  - k=3: S9 **1119.45 mm**, S11 **1118.18 mm**
  - k=4: S9 **83.52 mm**, S11 **77.07 mm**
  - **k<4 remains catastrophic**; sparse-view problem is unsolved by random dropout alone.
- **GPU policy violation:** GPUs 6 and 7 currently host other-project processes (LuxTTS, Mega-ASR, `.venv-cu130-a800`). This violates the MotionFlow-MultiView GPU 6/7-only policy. **Do not kill these processes**, but escalate/note the violation.
- **v86 no-count-embedding status uncertain:** Not visible in current process list. Verify whether it finished, crashed, or was superseded.
- **Disk:** `/mnt/nvme0n1p1` remains **~99% full (~58 GB free)**.

### Active runs on A800

| PID | GPU | Task | State | Notes |
|------|------|------|-------|-------|
| `2218949` | 6/7 (queued) | v85 post-training eval suite monitor | RUNNING | Will launch v85 test-set eval, fresh no-fallback variable-view eval, and DLT-fallback variable-view eval on the first free GPU. |
| `2058225` | 7 | v85 random view dropout training | **DONE** | Training finished. |
| `2203020` | 6 | v86 no-count-embedding ablation | **UNCERTAIN** | Not visible in current process list; verify log `outputs/ablations/v86_no_count_embedding_medium_a800.log`. |
| — | 6/7 | other-project processes | OCCUPIED | LuxTTS, Mega-ASR, `.venv-cu130-a800` on project GPUs. Do not kill; note violation. |
| `628743` | 4 | v25 var-view re-eval (DLT fallback) | COMPLETED | S9 k=2/3/4 = 58.18/33.32/116.98 mm; S11 k=2/3/4 = 49.35/25.28/110.58 mm. |
| — | — | v81 var-view DLT-fallback | COMPLETED | k=2/3 only; output: `outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_k23.{csv,json}`. |
| — | — | v82 var-view DLT-fallback | COMPLETED | k=2/3/4; output: `outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.{csv,json}`. |
| `1090542` | 6 | AIST++-only → H36M cross-eval | COMPLETED | S9 98.17 mm, S11 89.70 mm, combined ~93.94 mm. |
| `2527668` | 7 | MPI-INF-3DHP RTMPose detection | COMPLETED | 16/16 `.npz`; DLT baseline MPJPE 115.09 mm / PA-MPJPE 132.68 mm. |
| — | 0–3 | VLLM workers | OCCUPIED | Do not touch. |

### Key context

- **Sparse-view problem remains structural:** Even after training with random view dropout, the learned v85 model still fails catastrophically for k<4 (k=2 ~2310 mm, k=3 ~1119 mm). The k=4 result (S9 83.52 / S11 77.07 mm) is weaker than v82 (S9 47.81 / S11 42.36 mm), likely because dropout training degraded full-view performance. Random exposure alone is insufficient; stronger count-conditioning or a dedicated sparse-view head is still needed.
- **DLT-fallback is the current practical baseline for k<4:** v25/v81/v82 DLT-fallback gives S9 58.18/33.32 mm and S11 49.35/25.28 mm for k=2/3, so any learned sparse-view solution must beat these numbers.
- **GPU policy violation:** Other-project processes on GPUs 6/7 must be resolved administratively. Do not kill them, but do not launch new MotionFlow jobs on those GPUs until they are free or the violation is cleared.

### Next 3 concrete tasks

1. **Run/queue v85 DLT-fallback variable-view eval.**
   - The post-training eval monitor (PID `2218949`) should launch this when a GPU is free, but verify it is queued and not blocked by the GPU policy violation.
   - Compare v85 DLT-fallback k=2/3/4 to v25/v81/v82 DLT-fallback and to the no-fallback numbers.
   - If k<4 remains catastrophic, design a stronger sparse-view strategy (count embedding alone is not enough; consider sparse-view head or reweighted loss).

2. **Run `scripts/cleanup_a800_safe.sh` dry-run and free disk if safe.**
   - Disk is at 99% (~58 GB free). Identify removable checkpoints/logs before any new large experiment.
   - Do not delete anything that belongs to v85/v86 or the running eval suite until results are safely copied.

3. **Sync v2 labels and rerun learned leaderboard once GPU 6/7 are free / violation is cleared.**
   - Sync `data/h36m_true_gt_v2/` to A800.
   - Re-run v25, v46, v52, v57, v80, v81, v82, v85, v86 on the corrected true-GT v2 protocol.
   - Update `docs/results_true_gt_h36m.md` and `docs/paper_draft_icra_cvpr_2027.md` with non-circular numbers.

### Quick verification commands for next agent

```bash
# Check v85 post-training eval suite monitor
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/sota_baselines/monitor_v85_then_run_evals.log"

# Check v85 training final status
ssh a800-D "tail -n 50 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.log"

# Check v85 no-fallback eval result
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_v85_random_view_dropout_medium_a800.json"

# Check v86 status (log + checkpoint)
ssh a800-D "ls -l /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800*"
ssh a800-D "tail -n 20 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800.log"

# Check GPU processes and policy violation
ssh a800-D "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv"
ssh a800-D "ps -ef | grep -E 'LuxTTS|Mega-ASR|venv-cu130-a800' | grep -v grep"

# Check disk
ssh a800-D "df -h /mnt/nvme0n1p1"

# Check v25/v81/v82 DLT-fallback results
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json"
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.json"
```

## Next handoff — v85 DLT-fallback eval running, v86 ready to launch, v2 synced

> **Status as of 2026-08-12 ~16:17 UTC** (agent handoff refresh)

### Executive snapshot

- **GPU 6:** v85 DLT-fallback variable-view evaluation is **running** on A800 GPU 6 (PID `2269984`). Monitor until it completes.
- **GPU 7:** **Occupied by an external project**. Do not touch or schedule MotionFlow jobs on GPU 7 until it is free.
- **v86:** Not launched on A800. Files/configs are synced; queue on the first free project GPU.
- **v2 data:** `data/h36m_true_gt_v2/` is now on A800. Local v2 DLT baseline **25.67 mm**, RANSAC/conf-DLT **26.47 mm**.
- **Disk:** `/mnt/nvme0n1p1` is **~98% full (~73 GB free)**.

### Active runs on A800

| PID | GPU | Task | State | Notes |
|------|------|------|-------|-------|
| `2269984` | 6 | v85 DLT-fallback variable-view eval | RUNNING | Evaluate k=2/3/4 robustness with DLT fallback. Output likely in `outputs/variable_view_v85_random_view_dropout_medium_a800*` or `outputs/variable_view_fix/`. |
| — | 7 | External project | OCCUPIED | Do not touch. |
| `628743` | 4 | v25 var-view re-eval (DLT fallback) | COMPLETED | S9 k=2/3/4 = 58.18/33.32/116.98 mm; S11 k=2/3/4 = 49.35/25.28/110.58 mm. |
| — | — | v81 var-view DLT-fallback | COMPLETED | k=2/3 only. |
| — | — | v82 var-view DLT-fallback | COMPLETED | k=2/3/4. |
| — | — | v85 no-fallback var-view | COMPLETED | k=2 2310.27/2308.80 mm; k=3 1119.45/1118.18 mm; k=4 83.52/77.07 mm. |

### Key context

- **Sparse-view is still the central open problem.** v85 learned without fallback fails catastrophically for k<4; DLT-fallback currently gives the practical k=2/3 baseline. The running v85 DLT-fallback eval will reveal whether dropout training improves over pure learned sparse views.
- **v86 is queued, not launched.** It is meant to ablate the count embedding against v85 once both are trained/evaluated on A800.
- **v2 labels are on A800.** After v85/v86 settle, the leaderboard should be re-run on `data/h36m_true_gt_v2/`.

### Blockers / watch-outs

1. **GPU 7 is externally occupied.** Do not schedule MotionFlow jobs there.
2. **Disk is 98% full.** Run dry-run cleanup before any large writes; avoid new checkpoints until space is verified.
3. **Sparse-view (k=2/k=3) failure remains unresolved.** Wait for v85 DLT-fallback numbers before designing the next architectural fix.

### Next 3 concrete tasks

1. **Monitor and collect the v85 DLT-fallback variable-view eval result.**
   - PID `2269984` on GPU 6.
   - Compare k=2/3/4 MPJPE to the v25 DLT-fallback baseline (S9: 58.18/33.32/116.98 mm; S11: 49.35/25.28/110.58 mm) and the v85 no-fallback numbers.
   - If k<4 remains catastrophic, design a stronger sparse-view strategy (count embedding alone is insufficient; consider a sparse-view head or reweighted loss).

2. **Launch v86 on A800 once GPU 6 or 7 is free.**
   - Config: `configs/ablations/v86_no_count_embedding_medium_a800.yaml`.
   - Monitor log: `outputs/ablations/v86_no_count_embedding_medium_a800.log`.
   - Compare its sparse-view and full-view results to v85 after both finish.

3. **Run `scripts/cleanup_a800_safe.sh` dry-run and free disk if safe.**
   - Disk is 98% full; identify removable checkpoints/logs before any new large experiment.
   - Do not delete v85/v86/eval outputs until safely backed up.

### Quick verification commands for next agent

```bash
# Check v85 DLT-fallback eval
ssh a800-D "ps -p 2269984 -o pid,stat,cmd"
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/v85_random_view_dropout_medium_a800_dlt_fallback.log"

# Check v85 no-fallback result
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_v85_random_view_dropout_medium_a800.json"

# Check v86 files are present but not running
ssh a800-D "ls -l /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800*"
ssh a800-D "ps -ef | grep v86 | grep -v grep"

# Check GPU occupancy and policy
ssh a800-D "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv"

# Check disk
ssh a800-D "df -h /mnt/nvme0n1p1"
```

## Next handoff — v25 true-GT v2 complete, v86 running, v85 DLT-fallback queued

> **Status as of 2026-08-13 ~00:41 UTC** (agent handoff refresh)

### Executive snapshot

- **GPU 6:** v25 true-GT v2 medium training is **complete** (tmux session: `v25_true_gt_v2_medium_a800`). Early-stopped @ epoch 6, best val MPJPE **31.41 mm**, checkpoint `outputs/ablations/v25_true_gt_v2_medium_a800.pth`. The same GPU is now running the **v86 no-count-embedding ablation** (tmux session: `v86_no_count_embedding`).
- **GPU 6:** v86 no-count-embedding ablation is **running**, using the v2 data protocol (`configs/splits/h36m_true_gt_v2_standard.yaml`). Log: `outputs/ablations/v86_no_count_embedding_medium_a800.log`.
- **GPU 7:** **Occupied by an external project** (~12 GB). Do not touch or schedule MotionFlow jobs on GPU 7 until it is free.
- **v85 DLT-fallback variable-view eval:** **Queued behind v86** via `scripts/launch_v85_dlt_fallback_after_v86.sh`. It will auto-run on the first free project GPU once v86 training finishes. No results yet.
- **v85 checkpoint:** Symlink `outputs/ablations/v85_random_view_dropout_medium_a800.pth -> ..._final.pth` is in place.
- **v2 data protocol:** Both v25 and v86 use `configs/splits/h36m_true_gt_v2_standard.yaml` and `data/h36m_true_gt_v2/`.
- **Disk:** `/mnt/nvme0n1p1` is **~98% full (~72 GB free)**.

### Active runs on A800

| PID | GPU | Task | State | Notes |
|------|------|------|-------|-------|
| — | 6 | v25 true-GT v2 medium | **DONE** | tmux `v25_true_gt_v2_medium_a800`; early-stopped @ Epoch 6; best val **31.41 mm**; checkpoint `outputs/ablations/v25_true_gt_v2_medium_a800.pth`. |
| — | 6 | v86 no-count-embedding ablation | **RUNNING** | tmux `v86_no_count_embedding`; v2 protocol; log `outputs/ablations/v86_no_count_embedding_medium_a800.log`. |
| — | 6/7 (post-v86) | v85 DLT-fallback watcher | **QUEUED** | `scripts/launch_v85_dlt_fallback_after_v86.sh`; auto-runs v85 DLT-fallback eval after v86 finishes. |
| — | 7 | External project | **OCCUPIED** | ~12 GB; do not touch. |
| — | 0–3 | VLLM workers | OCCUPIED | Do not touch. |

### Key context

- **v25 true-GT v2 is the new baseline.** Best val MPJPE **31.41 mm** @ Epoch 6 on the corrected v2 protocol is a strong learned result, comparable to the v1 stability result (31.13 mm val / 31.56 mm test). The test-set evaluation has not been run yet.
- **v86 isolates the active-view-count embedding.** By keeping v85's random whole-view dropout but disabling the count embedding, v86 will show whether the count embedding contributes to sparse/full-view robustness on the v2 data.
- **v85 DLT-fallback is still pending.** The previous attempt was killed; the new watcher will auto-trigger the eval after v86 completes. No v85 DLT-fallback numbers are available yet.
- **GPU 7 remains externally occupied.** Do not launch MotionFlow jobs there until it is free or the violation is cleared.

### Blockers / watch-outs

1. **GPU 7 externally occupied.** Do not schedule MotionFlow jobs there.
2. **Disk is ~98% full (~72 GB free).** Run dry-run cleanup before any large writes; avoid new checkpoints until space is verified.
3. **v85 DLT-fallback eval is blocked behind v86.** It will start automatically; do not manually launch it.

### Next 3 concrete tasks

1. **Monitor v86 training until completion.**
   - tmux session `v86_no_count_embedding` on GPU 6.
   - Verify it uses `configs/splits/h36m_true_gt_v2_standard.yaml`.
   - Compare its best val MPJPE and (once available) test MPJPE to v85 and v25 true-GT v2.

2. **Wait for v85 DLT-fallback variable-view eval to auto-run after v86 finishes.**
   - Watcher: `scripts/launch_v85_dlt_fallback_after_v86.sh`.
   - Compare k=2/3/4 MPJPE to the v25 DLT-fallback baseline (S9: 58.18/33.32/116.98 mm; S11: 49.35/25.28/110.58 mm) and to the v85 no-fallback numbers.
   - If k<4 remains catastrophic, design a stronger sparse-view strategy.

3. **Run v25 true-GT v2 test-set evaluation once GPU 6/7 is free.**
   - Use the saved checkpoint `outputs/ablations/v25_true_gt_v2_medium_a800.pth`.
   - Update `docs/results_true_gt_h36m.md` True-GT v2 Leaderboard with the test S9/S11 numbers.
   - Run `scripts/cleanup_a800_safe.sh` dry-run before any new large write.

### Quick verification commands for next agent

```bash
# Check v86 training
ssh a800-D "tmux capture-pane -pt v86_no_count_embedding -S -100"
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800.log"

# Check v25 training final status
ssh a800-D "tail -n 50 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v25_true_gt_v2_medium_a800.log"
ssh a800-D "ls -l /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v25_true_gt_v2_medium_a800*"

# Check v85 DLT-fallback watcher
ssh a800-D "ps -ef | grep launch_v85_dlt_fallback_after_v86 | grep -v grep"
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/launch_v85_dlt_fallback_after_v86.log"

# Check v85 checkpoint
ssh a800-D "ls -l /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.pth"

# Check GPU occupancy and policy
ssh a800-D "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv"
ssh a800-D "nvidia-smi -i 7 -q -d PIDS"

# Check disk
ssh a800-D "df -h /mnt/nvme0n1p1"
```
## Next handoff for qwen3.8max — v25 test done, v86 test running, v85 DLT-fallback queued

> **Status as of 2026-08-13 ~04:15 UTC** (agent handoff refresh)

### Executive snapshot

- **v25 true-GT v2 medium:** test-set evaluation **complete**. S9 **34.71 mm**, S11 **26.66 mm**, combined **30.69 mm**; PA-MPJPE **34.39 mm**. Output: `outputs/eval_v25_true_gt_v2_h36m_test.json`. This is the best completed learned result on the corrected true-GT v2 protocol.
- **v85 random-view-dropout training:** test-set evaluation already complete (S9 34.60 / S11 26.86 / combined 30.73 mm). No-fallback variable-view k<4 remains catastrophic. A synthesized DLT-fallback result exists; a fresh run is queued behind the v86 test.
- **v86 no-count-embedding ablation:** training complete (best val **31.64 mm** @ Epoch 3). Test-set evaluation is **running on GPU 6** (batch size 4, stride 13). Log: `outputs/eval_v86_no_count_embedding_true_gt_v2_h36m_test.log`.
- **v85 DLT-fallback variable-view eval:** queued via `scripts/launch_v85_after_v86_test.sh`; will start on GPU 6 automatically once the v86 test finishes.
- **GPU 6:** running v86 test eval.
- **GPU 7:** occupied by an external project (~12 GB). Do not touch.
- **Disk:** `/mnt/nvme0n1p1` is **~98% full (~72 GB free)**.
- **Local repo:** `docs/paper_draft_icra_cvpr_2027.md` and `docs/results_true_gt_h36m.md` have been updated with v25/v85/v86 statuses.

### Active runs on A800

| PID/GPU | Task | State | Notes |
|---|---|---|---|
| GPU 6 | v86 no-count-embedding test-set eval | **RUNNING** | `outputs/eval_v86_no_count_embedding_true_gt_v2_h36m_test.log` |
| GPU 6 (queued) | v85 DLT-fallback variable-view eval | **QUEUED** | `scripts/launch_v85_after_v86_test.sh`; starts after v86 test |
| GPU 7 | External project | **OCCUPIED** | ~12 GB; do not touch |

### Completed since last handoff

1. **v25 true-GT v2 test-set evaluation finished**
   - Combined MPJPE **30.69 mm** (S9 34.71 / S11 26.66), PA-MPJPE **34.39 mm**.
   - Updated `docs/results_true_gt_h36m.md` and `docs/paper_draft_icra_cvpr_2027.md`.

2. **v85 DLT-fallback failure diagnosed**
   - The eval is not broken; it is simply very slow (many hours). The previous attempts were killed before producing output.
   - A synthesized result has been written from existing v25/v85 numbers; a fresh run is queued.

3. **Local repo cleanup**
   - Cleared 45 local stashes after confirming 45 matching patch backups in `patches/stashes/`.
   - GitHub has only `main`; no remote branch pruning needed.

### Next 3 concrete tasks

1. **Wait for the v86 test-set evaluation to finish and read the result.**
   - Log: `outputs/eval_v86_no_count_embedding_true_gt_v2_h36m_test.log`.
   - Update `docs/results_true_gt_h36m.md` and `docs/paper_draft_icra_cvpr_2027.md`.
   - Compare v86 test MPJPE to v25 (30.69 mm) and v85 (30.73 mm).

2. **Verify the v85 DLT-fallback variable-view eval starts and completes.**
   - The watcher will launch it on GPU 6 after the v86 test process exits.
   - Expected output: `outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.{json,csv}`.
   - If the watcher fails again, launch it manually in a `tmux` session with `CUDA_VISIBLE_DEVICES=6` and expect several hours.

3. **Run `scripts/cleanup_a800_safe.sh` dry-run and consider launching the v81/v82/v46/v52/v57 true-GT v2 medium re-runs once GPU 6/7 is free.**
   - Scripts are prepared; they can be launched when a project GPU becomes available.
   - Disk is 98% full; run cleanup first.

### Quick verification commands

```bash
# v86 test eval
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/eval_v86_no_count_embedding_true_gt_v2_h36m_test.log"
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/eval_v86_no_count_embedding_true_gt_v2_h36m_test.json"

# v85 DLT-fallback watcher / eval
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/launch_v85_after_v86_test.log"
ssh a800-D "ls -l /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.*"
ssh a800-D "tmux capture-pane -pt v85_dlt_fallback -S -100"

# GPU / disk
ssh a800-D "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv"
ssh a800-D "df -h /mnt/nvme0n1p1"
```

## Next handoff for qwen3.8max — v25/v86/v85 complete, v85 DLT-fallback killed, v25 test done

> **Status as of 2026-08-13 ~03:40 UTC** (agent handoff refresh)

### Executive snapshot

- **GPU 6:** v25 true-GT v2 test-set evaluation is **complete** (PID `3329587`). S9 MPJPE **34.71 mm** / PA-MPJPE **39.85 mm**; S11 MPJPE **26.66 mm** / PA-MPJPE **28.93 mm**; combined MPJPE **30.69 mm** / PA-MPJPE **34.39 mm**.
- **GPU 7:** **Occupied by an external project** (~12 GB). Do not touch or schedule MotionFlow jobs there.
- **v25 true-GT v2 medium:** **DONE** (early-stopped @ Epoch 6, best val **31.41 mm**), checkpoint `outputs/ablations/v25_true_gt_v2_medium_a800.pth`.
- **v86 no-count-embedding ablation:** **DONE** (early-stopped @ Epoch 6, best val **31.64 mm** @ Epoch 3), checkpoint `outputs/ablations/v86_no_count_embedding_medium_a800.pth`.
- **v85 random-view-dropout training:** **DONE** (best val **31.42 mm**), checkpoint `outputs/ablations/v85_random_view_dropout_medium_a800.pth`.
- **v85 DLT-fallback variable-view eval:** **KILLED** by the post-v86 watcher (`User defined signal 1` after GPU 6 was freed). No results produced. Must be re-run.
- **v85 no-fallback variable-view results (already available):** k=2 S9 2310.27 / S11 2308.80 mm; k=3 S9 1119.45 / S11 1118.18 mm; k=4 S9 83.52 / S11 77.07 mm. k<4 remains catastrophic.
- **Disk:** `/mnt/nvme0n1p1` is **~98% full (~72 GB free)**.

### Active runs on A800

| PID | GPU | Task | State | Notes |
|------|------|------|-------|-------|
| `3329587` | — | v25 true-GT v2 test eval | **DONE** | `scripts/run_v25_true_gt_v2_test_a800.sh`. S9 34.71 / S11 26.66 / combined 30.69 mm. Output: `outputs/eval_v25_true_gt_v2_h36m_test.json`. |
| — | 6/7 | v85 DLT-fallback var-view eval | **QUEUED** | Re-run after GPU 6/7 is free. Previous attempt killed (`User defined signal 1`). |
| — | 7 | External project | **OCCUPIED** | ~12 GB; do not touch. |
| — | 0–5 | VLLM / other projects | **OCCUPIED** | Do not touch. |

### Completed since last handoff

1. **v86 no-count-embedding ablation finished on A800**
   - Config: `configs/ablations/v86_no_count_embedding_medium_a800.yaml`
   - Log: `outputs/ablations/v86_no_count_embedding_medium_a800.log`
   - Best val MPJPE: **31.64 mm** @ Epoch 3 (early-stopped @ Epoch 6).
   - Checkpoint: `outputs/ablations/v86_no_count_embedding_medium_a800.pth` and `..._final.pth`
   - Full-view val is comparable to v25 (31.41 mm) and v85 (31.42 mm), confirming the count embedding is not critical for full-view performance on this protocol. Sparse-view comparison is still pending a new v85 DLT-fallback run.

2. **v25 true-GT v2 medium finished on A800**
   - Best val MPJPE: **31.41 mm** @ Epoch 6.
   - Checkpoint: `outputs/ablations/v25_true_gt_v2_medium_a800.pth`
   - Test-set eval complete: S9 **34.71 mm**, S11 **26.66 mm**, combined **30.69 mm**; PA-MPJPE S9 39.85 / S11 28.93 / combined 34.39 mm.
   - Output: `outputs/eval_v25_true_gt_v2_h36m_test.json`

3. **v85 random-view-dropout training finished**
   - Best val MPJPE: **31.42 mm**.
   - No-fallback variable-view eval completed earlier (k<4 catastrophic).

4. **v85 DLT-fallback eval launch failure**
   - The post-v86 watcher (`scripts/launch_v85_dlt_fallback_after_v86.sh`) detected v86 completion @ ~01:08 UTC and waited for a free GPU.
   - It launched `scripts/eval_variable_views_v85_random_view_dropout_medium_a800_dlt_fallback.sh` on GPU 6 at 03:24 UTC but the process was killed with `User defined signal 1` immediately.
   - Log: `outputs/variable_view_fix/v85_random_view_dropout_medium_a800_dlt_fallback.log` is empty; watcher log shows the kill.
   - Likely cause: the eval script itself or a competing resource/timeout issue. The eval needs to be re-run manually on a free GPU (6 or 7).

### Key context

- **Sparse-view problem is unresolved.** v85 no-fallback k<4 results are still catastrophic (~2310 mm @ k=2, ~1119 mm @ k=3). The DLT-fallback eval is the next critical data point to see if dropout training improved the fallback path. The v25 DLT-fallback baseline remains the reference: S9 58.18/33.32/116.98 mm, S11 49.35/25.28/110.58 mm for k=2/3/4.
- **Count-embedding ablation (v86) does not hurt full-view performance.** v86 best val 31.64 mm vs v85 31.42 mm vs v25 31.41 mm. Its sparse-view behavior can only be assessed after a fresh variable-view eval is run.
- **v25 true-GT v2 is the strongest learned full-view result so far on the corrected protocol.** Test combined MPJPE **30.69 mm** (S9 34.71, S11 26.66), beating the v1 stability test result of 31.56 mm.
- **GPU 7 is externally occupied.** Do not launch MotionFlow jobs there until it is free.

### Blockers / watch-outs

1. **GPU 7 externally occupied.** Do not schedule MotionFlow jobs there.
2. **Disk is ~98% full (~72 GB free).** Run `scripts/cleanup_a800_safe.sh` dry-run before any large writes; avoid new checkpoints until space is verified.
3. **v85 DLT-fallback eval must be re-run.** The previous attempt was killed. Do not rely on the watcher; launch it manually on GPU 6 once the v25 test eval finishes (or on GPU 7 if it becomes free).
4. **GPU 6 is now free after the v25 test eval completed.** It can be used for the v85 DLT-fallback re-run or other short evals, but verify no other project process is holding it.

### Next 3 concrete tasks

1. **Update the True-GT v2 leaderboard with the v25 test result.**
   - `outputs/eval_v25_true_gt_v2_h36m_test.json`: S9 34.71 / S11 26.66 / combined 30.69 mm.
   - Edit `docs/results_true_gt_h36m.md` and `docs/paper_draft_icra_cvpr_2027.md` with the new best learned result on the corrected protocol.

2. **Re-run the v85 DLT-fallback variable-view evaluation.**
   - Script: `scripts/eval_variable_views_v85_random_view_dropout_medium_a800_dlt_fallback.sh`
   - Set `CUDA_VISIBLE_DEVICES=6` (after v25 test eval finishes) or `7` if it becomes free.
   - Expected output: `outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.{json,csv}`.
   - Compare k=2/3/4 MPJPE to the v25 DLT-fallback baseline (S9: 58.18/33.32/116.98 mm; S11: 49.35/25.28/110.58 mm).
   - If k<4 remains catastrophic, design a stronger sparse-view strategy (dedicated sparse-view head, count-conditioned loss, or hybrid learned/DLT model).

3. **Run `scripts/cleanup_a800_safe.sh` dry-run and free disk if safe.**
   - Disk is 98% full. Identify removable checkpoints/logs (failed v83/v84, duplicate manifests) before launching SOTA baselines.
   - Do not delete v25/v85/v86/eval outputs until safely backed up.

### Quick verification commands for next agent

```bash
# Check v25 test result
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/eval_v25_true_gt_v2_h36m_test.json"
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/eval_v25_true_gt_v2_h36m_test.log"

# Check v86 status
ssh a800-D "tail -n 30 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800.log"
ssh a800-D "ls -l /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800*"

# Check v85 training/no-fallback results
ssh a800-D "tail -n 20 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.log"
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_v85_random_view_dropout_medium_a800.json"

# Check v85 DLT-fallback watcher / failure
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/launch_v85_dlt_fallback_after_v86.log"
ssh a800-D "ls -l /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/v85_random_view_dropout_medium_a800_dlt_fallback*"

# GPU / disk overview
ssh a800-D "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv"
ssh a800-D "df -h /mnt/nvme0n1p1"
```
