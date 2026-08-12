# MotionFlow-MultiView Agent Notes

> **Executive Summary — 2026-08-12 ~12:56 UTC**
>
> **A800 status (do not touch running jobs):** v85 random-view-dropout training is **still running** on GPU 7; v86 no-count-embedding ablation is **still running** on GPU 6. **Only GPUs 6/7 are used by this project.** A800 disk is **~99% full (~58 GB free)**.
>
> **Data foundation — v2 audit complete:** The legacy `data/h36m_hf/*.npz` are circular (direct MJE ≈ 0 mm). The `data/h36m_true_gt/*.npz` files used by v85 and earlier runs are misaligned with their stored cameras/2D (direct MJE ≈ 16,668 mm). A corrected converter `scripts/convert_h36m_true_gt_v2.py` now produces physically consistent true-mocap labels. Full H36M true-GT v2 `.npz` audit is **complete**: DLT baseline **25.67 mm** and RANSAC/conf-DLT baseline **26.47 mm** are reproducible on `data/h36m_true_gt_v2/`. MPI-INF-3DHP detected-2D (16/16 `.npz`) and DLT baseline (MPJPE 115.09 mm) done. AIST++ canonical `.npz` and H36M cross-eval (93.94 mm) done.
>
> **True-GT H36M leaderboard (S9/S11 test):** Iskakov 23.40 mm; conf-DLT 25.67 mm; RANSAC/conf-DLT 26.47 mm; **v25 stability 31.56 mm** (best learned); v81 37.83 mm; v82 39.46 mm; v80 53.98 mm; v52 54.01 mm; v57 57.10 mm.
>
> **CVPR 2027 plan:** 1) Let v85 finish; evaluate sparse-view k=2/3/4 robustness against the v25 DLT-fallback baseline (S9 58.18/33.32/116.98 mm; S11 49.35/25.28/110.58 mm). 2) Run VoxelPose/MVPose SOTA comparisons once GPU 6/7 free. 3) Run `scripts/cleanup_a800_safe.sh` dry-run before any large write. 4) Rewrite paper around true-GT, sparse-view, and cross-dataset robustness.

## Current-session update (AIST++ NaN fix / v25 var-view / MPI / dropped v83/v84)

- **AIST++ NaN / empty-sequence blocker:** `convert_aistpp` originally used raw `keypoints3d`, which contains NaNs on ~20% of sequences. The first fix preferred `keypoints3d_optim` and dropped frames with any NaN, but two sequences had NaN in all 2D frames and became empty, crashing the mixed-loader collate function. The converter now zeroes NaN 2D keypoints and sets confidence to 0 while preserving frame count, and only drops frames if 3D joints are NaN. Clean canonical `.npz` are regenerating locally and will be synced to A800 before relaunching on GPU 5.
- **v83/v84 dropped:** v83 A800 medium plateaued at **~100 mm** val and was killed. v84 uncertainty-weighted view dropout smoke produced **107.11 mm** val, also no improvement. Architecture modules on top of v25 ray tokens are deprioritized until cross-dataset training is baselined.
- **v25 stability variable-view eval:** The wrapper fix (explicit `view_mask` to `OmniMultiViewFusionV5`) was necessary but not sufficient: k=2/k=3 remained catastrophic (~3000/1000 mm). A diagnostic showed that while the learned model fails catastrophically for k<4, direct confidence-weighted DLT on the same active views achieves ~35–100 mm. A new `--var_view_dlt_fallback` mode was added to `HardenedVariableViewInferenceWrapper` that falls back to direct DLT whenever `n_active < n_views_max`. The DLT-fallback re-eval (PID `628743`) completed on GPU 4. Full S9/S11 numbers: k=2 **58.18 / 49.35 mm**, k=3 **33.32 / 25.28 mm**, k=4 **116.98 / 110.58 mm**. For k<4 the learned model is not used; direct confidence-weighted DLT fallback is used instead.
- **v85 no-fallback variable-view eval completed:** Split-k run (k=2,3,4 sequential, 50 subsets per k) finished on GPU 6. The early-stopped v85 checkpoint (best val MPJPE 31.42 mm) produced: **k=2 S9 2310.27 mm / S11 2308.80 mm**, **k=3 S9 1119.45 mm / S11 1118.18 mm**, **k=4 S9 83.52 mm / S11 77.07 mm**. k<4 remains catastrophic, but k=2 is better than the v25 no-fallback baseline (S9 ~3017 / S11 ~2862 mm) and k=4 is much better than v25 (S9 ~117 / S11 ~111 mm). Combined JSON/CSV: `outputs/variable_view_v85_random_view_dropout_medium_a800.{json,csv}`. Per-k files: `outputs/variable_view_v85_random_view_dropout_medium_a800_k{2,3,4}.{json,csv}`.
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
```
