# MotionFlow-MultiView Agent Notes

> **Status: AIST++ CANONICAL .NPZ NaN BLOCKER DIAGNOSED AND FIX APPLIED; v25 STABILITY VAR-VIEW EVAL RUNNING ON GPU 4, MPI RTMPose 5/16 ON GPU 7, v81 DONE (TEST 37.83 MM), v25 STABILITY TEST 31.56 MM, AIST++ FULL DLT FINISHED — LOCAL RTX 4090 FOR QUICK SMOKE ONLY — 2026-08-12 (agent-6 handoff refresh)**
>
> H36M circular-label problem is **confirmed and repaired**: `scripts/diagnose_circular_labels.py` reports `direct MJE = 0.0000 mm` on `data/h36m_hf/*.npz`, and `motionflow_mv/data/webbridge_loader.py` now sources true GT 3D instead of triangulating from input 2D. Trainer best-checkpoint saving was fixed to monitor `mpjpe` instead of `loss`, eliminating the v57 epoch-2 vs epoch-3 mismatch.
>
> **Latest fixes / launches:** v25 true-GT stability **finished**: test S9/S11 **31.56 mm** weighted (best true-GT learned result so far). v81 temporal-pose-attention medium on **A800 GPU 4**: **early-stopped @ Epoch 8**, test **37.83 mm**. **v82 multi-scale temporal-pose-attention medium** on **A800 GPU 4**: **finished 8 epochs**, best val **39.58 mm** @ Epoch 8, test **39.46 mm** (slightly worse than v81). v81 variable-view evaluation **completed**: k=4 **50.97 mm**, k=2/k=3 catastrophically high. **AIST++-only fast v2 was reporting `val_loss=nan, val_MPJPE=nan` at Epoch 1** because `convert_aistpp` used raw `keypoints3d`, which contains NaNs on ~20% of sequences. The converter now prefers `keypoints3d_optim` and drops any residual NaN frames. Clean canonical `.npz` are being regenerated locally and will be synced to A800. v25 true-GT mixed-dataset **diverged @ Epoch 3** but best checkpoint tests at **33.42 mm** average. v52 UWT finished (test **54.01 mm**). AIST++ full DLT baseline **finished**: MPJPE **15.93 mm** weighted / **38.11 mm** unweighted. MPI-INF-3DHP RTMPose detection is running on **GPU 7** (PID `2527668`), **5/16 .npz files** done. v57/v80/v81 variable-view evaluation **completed**. v83/v84 architecture variants **dropped** after plateauing at ~100 mm.
>
> **Current true-GT H36M leaderboard (test S9/S11):**
> - Iskakov ICCV 2019: **23.35 mm**
> - Conf-weighted DLT: **25.67 mm**
> - RANSAC/conf-DLT: **26.61 mm**
> - v81 temporal-pose-attention: **37.83 mm** @ test (S9 42.19 mm, S11 33.46 mm), PA-MPJPE 37.75 mm; best val 38.62 mm @ epoch 8
> - v82 multi-scale temporal-pose-attention: **39.46 mm** @ test (S9 42.07 mm, S11 36.84 mm), PA-MPJPE 39.94 mm; best val 39.58 mm @ epoch 8
> - v80: **39.98 mm** (epoch 4, then overfit)
> - v25: **43.93 mm** on test (val log 72.80 mm was inflated due to missing `view_mask` during validation)
> - v25 mixed (H36M + AIST++): **33.42 mm** average / **34.23 mm** weighted on test (S9 37.87 mm, S11 28.96 mm) — diverged @ Epoch 3
> - v25 stability: **31.56 mm** weighted / **30.83 mm** average on test (S9 34.87 mm, S11 26.80 mm), PA-MPJPE **34.35 mm**; best val **31.13 mm** @ Epoch 10
> - v57 re-run: **57.81 mm** @ epoch 4 (early-stopped @ epoch 7); test S9/S11 **57.10 mm**; checkpoint saved correctly at `outputs/ablations/v57_true_gt_medium_a800.pth`
> - v52 UWT: **54.01 mm** @ test (S9 58.15 mm, S11 49.87 mm); PA-MPJPE 42.22 mm; val best **54.75 mm** @ epoch 4
> - v80 regularization: **53.98 mm** @ test (S9 56.69 mm, S11 51.27 mm), PA-MPJPE 32.47 mm; val best **54.46 mm** @ epoch 1 (early-stopped @ epoch 4)
> - v46: **52.46 mm** combined (S9 55.03 mm, S11 49.88 mm), PA-MPJPE 40.20 mm
>
> **v25 true-GT stability on A800 (GPU 6):** Completed / early-stopped. Best val MPJPE **31.13 mm** @ Epoch 10 (early-stopped @ Epoch 12). Test S9/S11: **31.56 mm** weighted (S9 34.87 mm, S11 26.80 mm), PA-MPJPE **34.35 mm**; average **30.83 mm**. Result JSON: `outputs/eval_v25_true_gt_stability_h36m_test.json`.
>
> **AIST++-only medium on A800 (GPU 5):** Fast run v3 (`train_samples=64`, `batch_size=32`, `epochs=10`) **relaunched without `--use_full_precision_dlt`**; avoids the cusolver `eigh` error and is training. Log: `outputs/ablations/aistpp_only_medium_a800_fast_v2.log`; checkpoint: `outputs/ablations/aistpp_only_medium_a800_fast_v2.pth`.
>
> **v81 temporal-pose-attention medium on A800 (GPU 4):** Early-stopped at Epoch 8 with best val **38.62 mm**. Test S9/S11: **37.83 mm** (S9 42.19 mm, S11 33.46 mm), PA-MPJPE **37.75 mm**. Log: `outputs/ablations/v81_true_gt_h36m_medium_a800.log`; checkpoint: `outputs/ablations/v81_true_gt_h36m_medium_a800.pth`; test JSON: `outputs/eval_v81_true_gt_h36m_test_a800.json`.
>
> **v57 re-run on A800:** Completed on **GPU 5** (`v57_true_gt_medium_a800`, log `outputs/ablations/v57_true_gt_medium_a800.log`). Best val MPJPE **57.81 mm** @ epoch 4; early-stopped @ epoch 7. Checkpoint: `outputs/ablations/v57_true_gt_medium_a800.pth` (55 MB).
>
> **v80 true-GT regularization on A800:** Completed on **GPU 6** (`v80_true_gt_regularization_a800`). Best val MPJPE **54.46 mm** @ epoch 1; early-stopped @ epoch 4. Checkpoint/log: `outputs/ablations/v80_true_gt_regularization_a800.pth` / `.log`.
>
> **GPU reality check:** GPU 4 is free after v82 training finished. GPU 5 hosts the AIST++-only fast v2 training run. GPU 6 runs the v82 variable-view evaluation. GPU 7 runs MPI RTMPose detection (PID `2527668`, **4/16 .npz** complete) and a CPU DLT-baseline watcher. GPUs 0-3 run VLLM workers.
>
> **Disk space warning:** A800 `/mnt/nvme0n1p1` is **99% full** (~46 GB free out of 3.5 TB). Project outputs are small, but system-wide space is tight; avoid large checkpoint dumps or extracted video frames until cleanup is done.
>
> **Post-8th-swarm / current-session state:** v82 multi-scale temporal-pose-attention medium **finished** on GPU 4 (best val **39.58 mm** @ Epoch 8, test **39.46 mm**). v83 view-conditioned temporal attention is **implemented** and its local smoke is **running on RTX 4090**. AIST++-only fast v2 is **running on A800 GPU 5** after the previous cusolver crash, currently past step 450 with loss decreasing. v82 variable-view evaluation is **running on A800 GPU 6**. MPI RTMPose detection has produced **4/16 .npz files** on GPU 7; the waiting DLT baseline already reports **150.50 mm** mean MPJPE on those files. v81 test **37.83 mm** and v25 stability test **31.56 mm** remain the best true-GT learned results. AIST++ full DLT baseline is done.

## Current-session update (AIST++ NaN fix / v25 var-view / MPI / dropped v83/v84)

- **AIST++ NaN blocker:** `convert_aistpp` in `motionflow_mv/data/webbridge_loader.py` was using raw `keypoints3d`, which contains NaNs on ~20% of sequences. The fix prefers `keypoints3d_optim` (verified clean across all 1,408 sequences) and drops any residual NaN frames before saving. Clean canonical `.npz` are being regenerated locally and will be rsynced to A800; the old A800 run has been stopped and will be relaunched on GPU 5.
- **v83/v84 dropped:** v83 A800 medium plateaued at **~100 mm** val and was killed. v84 uncertainty-weighted view dropout smoke produced **107.11 mm** val, also no improvement. Architecture modules on top of v25 ray tokens are deprioritized until cross-dataset training is baselined.
- **v25 stability variable-view eval:** Running on A800 GPU 4 (PID `4184808`). Outputs will land in `outputs/variable_view_v25_true_gt_stability_a800.csv/.json`.
- **MPI-INF-3DHP detection:** RTMPose 2D detection is running on GPU 7 (PID `2527668`). Five of 16 `.npz` files are written to `data/webbridge/mpi_inf_3dhp_detected_2d/`.

## Current work in flight

| Agent | Task | Machine | Notes |
|-------|------|---------|-------|
| `agent-6` (running) | AGENTS.md / handoff refresh | Local WSL | Updating status; v83 smoke running, AIST++ v2 running, v82 eval running, MPI 4/16; no code changes |
| `agent-272` (done) | v57 true-GT re-run | A800 GPU 5 | `v57_true_gt_medium_a800`; best **57.81 mm** @ Epoch 4; early-stopped @ Epoch 7 |
| `agent-51` (done) | H36M true-GT v25 medium | Local RTX 4090 | Completed: test 43.93 mm; val log 72.80 mm @ epoch 2 (val ignored `view_mask`) |
| `agent-67` (done / idle) | AIST++ smoke integration v25/v80 | Local RTX 4090 / A800 | Smoke complete (DLT 6.52 mm); only 16 `.npz` present on A800 |
| `agent-269` (done) | v80 true-GT regularization ablation | A800 GPU 6 | Best **54.46 mm** val @ Epoch 1; early-stopped @ Epoch 4; test **53.98 mm** combined |

- **Strategy update:** Local RTX 4090 is reserved for **quick smoke/verification only**; large training tasks (medium/long ablations, SOTA baselines, cross-dataset training) run on **A800**.
- The RTX 4090 is currently running the **v83 smoke**; do not start another training run until it finishes.
- v52 UWT (`v52_true_gt_h36m_a800`) **finished** on **A800 GPU 4**. Best val **54.75 mm** @ Epoch 4, early-stopped @ Epoch 7. **Test S9/S11: 54.01 mm** (S9 58.15 mm, S11 49.87 mm), PA-MPJPE 42.22 mm.
- v81 temporal-pose-attention medium (`v81_true_gt_h36m_medium_a800`) on **A800 GPU 4** — **early-stopped @ Epoch 8** with best val **38.62 mm**. Test S9/S11: **37.83 mm** (S9 42.19 mm, S11 33.46 mm), PA-MPJPE **37.75 mm**. Log: `outputs/ablations/v81_true_gt_h36m_medium_a800.log`; test JSON: `outputs/eval_v81_true_gt_h36m_test_a800.json`.
- v82 multi-scale temporal-pose-attention medium (`v82_true_gt_h36m_medium_a800`) **finished on A800 GPU 4**. Best val **39.58 mm** @ Epoch 8; test S9/S11 **39.46 mm**. Log: `outputs/ablations/v82_true_gt_h36m_medium_a800.log`; checkpoint: `outputs/ablations/v82_true_gt_h36m_medium_a800.pth`.
- v81 variable-view evaluation (`eval_variable_views_v81_true_gt_medium_a800`) **completed on A800 GPU 6**. Outputs: `outputs/variable_view_v81_true_gt_medium_a800.csv` / `.json`.
- v82 variable-view evaluation (`eval_variable_views_v82_true_gt_medium_a800`) **running on A800 GPU 6**. Outputs: `outputs/variable_view_v82_true_gt_medium_a800.csv` / `.json`.
- AIST++-only medium fast v2 (`aistpp_only_medium_a800_fast_v2`) **stopped after NaN validation** (`val_loss=nan, val_MPJPE=nan` at Epoch 1). Root cause: raw `keypoints3d` contains NaNs; fix applied in `motionflow_mv/data/webbridge_loader.py`. Clean canonical `.npz` are regenerating locally and will be synced to A800 before relaunching on GPU 5.
- v25 mixed-dataset (`v25_true_gt_mixed_dataset_a800`) on **A800 GPU 5** has **diverged** (Epoch 3 val **584.25 mm**), but its best checkpoint tests at **33.42 mm** avg on H36M S9/S11. Consider killing or debugging before re-running.
- v25 stability (`v25_true_gt_stability_a800`) on **A800 GPU 6** **early-stopped @ Epoch 12**; best val **31.13 mm** @ Epoch 10. Test S9/S11: **31.56 mm** weighted (S9 34.87 mm, S11 26.80 mm), PA-MPJPE **34.35 mm**. Result JSON: `outputs/eval_v25_true_gt_stability_h36m_test.json`.
- AIST++ full DLT baseline **finished on A800 GPU 6**: MPJPE **15.93 mm** weighted / **38.11 mm** unweighted; PA-MPJPE **21.12 mm** / **42.66 mm**. Log: `outputs/aistpp_full_dlt_baseline_a800.log`; JSON: `outputs/aistpp_full_dlt_baseline_a800.json`.
- MPI-INF-3DHP RTMPose detection running on **A800 GPU 7** (PID `2527668`, CUDA provider active). Log: `outputs/mpi_rtmpose_detected_2d/generate_20260811_191500.log`. Files land in `data/webbridge/mpi_inf_3dhp_detected_2d/`; **4/16 `.npz` files complete**.
- v57/v80 variable-view evaluation **completed**. v80 regularization is better than v57 at all view counts: v80 S9@4 102.8 mm / S11@4 105.8 mm vs v57 S9@4 143.0 mm / S11@4 137.4 mm.
- Before starting any new work, check active background tasks and A800 GPU availability.

## Why we paused

- `scripts/diagnose_circular_labels.py` confirms `direct MJE = 0.0000 mm` on `data/h36m_hf/*_multiview.npz`.
- `motionflow_mv/data/webbridge_loader.py:182` triangulates the input 2D and stores it as the 3D label.
- v25–v79 numbers are therefore measuring how closely a network reproduces the DLT layer, not pose accuracy.
- The raw pkl `h36m_sh_conf_cam_source_final.pkl.zip` only contains `joint3d_image` (image-space `(u,v,z)`), which cannot be converted to a consistent world 3D across cameras.

## Data foundation status

1. **True H36M 3D GT** — obtained; true-GT `.npz` are in `data/h36m_true_gt/`.
2. **Regenerate canonical `.npz`** with non-circular labels — done; H36M and Shelf/Campus datasets have been rebuilt.
3. **Re-run baselines** (DLT, Iskakov, v25, v46, v52, v57, v80) on the corrected protocol — in progress.
   - DLT H36M true-GT: **25.67 mm** (conf-weighted), 28.77 mm (unweighted).
   - RANSAC/conf-DLT H36M true-GT: **26.61 mm**.
   - Iskakov ICCV 2019 H36M true-GT: **23.35 mm**.
   - v25 H36M true-GT medium: test **43.93 mm**; original local val log **72.80 mm** @ epoch 2 was inflated because validation did not pass `view_mask`. A800 ablation 1 (`v25_true_gt_baseline_fix`) reached **46.53 mm** @ epoch 1 with the corrected validation recipe.
   - v46 H36M true-GT medium: val **52.92 mm** @ epoch 4, early-stopped @ epoch 7; test S9/S11 **52.46 mm** combined (S9 55.03 mm, S11 49.88 mm), PA-MPJPE 40.20 mm.
   - v52 UWT true-GT H36M: val **54.75 mm** @ epoch 4, early-stopped @ epoch 7; test S9/S11 **54.01 mm** (S9 58.15 mm, S11 49.87 mm), PA-MPJPE 42.22 mm.
   - v57 H36M true-GT medium: final **78.76 mm**; true best **75.16 mm** @ epoch 3 was not saved due to the loss-based checkpoint bug (now fixed to monitor MPJPE).
   - v80 H36M true-GT medium: baseline **39.98 mm** @ epoch 4, then overfit to 133.71 mm; regularization ablation **54.46 mm** @ epoch 1, early-stopped @ epoch 4; test **53.98 mm** combined (S9 56.69 mm, S11 51.27 mm), PA-MPJPE 32.47 mm.
   - v81 temporal-pose-attention H36M true-GT medium: best val **38.62 mm** @ epoch 8; test **37.83 mm** (S9 42.19 mm, S11 33.46 mm), PA-MPJPE **37.75 mm**.
   - v82 multi-scale temporal-pose-attention H36M true-GT medium: best val **39.58 mm** @ epoch 8; test **39.46 mm** (S9 42.07 mm, S11 36.84 mm), PA-MPJPE **39.94 mm**.
   - v83 view-conditioned temporal attention: **implemented** (`motionflow_mv/fusion/view_conditioned_temporal_attention_v83.py`); local smoke running on RTX 4090, A800 medium script ready.
   - See [`docs/results_true_gt_h36m.md`](docs/results_true_gt_h36m.md) and [`docs/results_true_gt_shelf_campus.md`](docs/results_true_gt_shelf_campus.md).
4. **MPI-INF-3DHP real detected 2D** — RTMPose regeneration running on GPU 7 (CUDA provider active, PID `2527668`). Output will land in `data/webbridge/mpi_inf_3dhp_detected_2d/`. Once files are present, validate the DLT baseline.
5. **AIST++ integration** — full 1,408 canonical `.npz` are now on A800 (`data/webbridge/aistpp_canonical/`). AIST++-only v25 medium fast v2 **relaunched and is training on GPU 5** after the earlier cusolver `eigh` crash; full AIST++ DLT baseline **finished**: MPJPE **15.93 mm** weighted / **38.11 mm** unweighted; PA-MPJPE **21.12 mm** / **42.66 mm**. Smoke DLT was conf-weighted **6.52 mm**, unweighted **12.66 mm**.
6. **Re-orient the paper contribution** around sparse-view / cross-domain robustness, not absolute MPJPE records — **paper rewrite in progress**.

## Current data sources

| Source | True 3D? | Status |
|--------|----------|--------|
| `data/h36m_hf/*.npz` | No | Circular labels; do not use for model selection |
| `data/webbridge/h36m*.npz` | No | Same circular labels |
| `data/h36m_true_gt/*_multiview_m.npz` | **Yes** | True mocap world coordinates; standard protocol S1,5,6,7,8 → S9/S11 |
| MPI-INF-3DHP | Yes | Labels are real mocap; RTMPose detected-2D regeneration in progress (GPU); previous MediaPipe run was ~326–400 mm due to joint-mapping/detection issues |
| Shelf/Campus | Yes | Non-circular `.npz` rebuilt from `detection.json + annotation_3d.json` at `data/webbridge/shelf_campus_detected/` |
| A800-D `/mnt/nvme0n1p1/zhangzy/projects` | No true H36M found | Read-only only |

## Next 3 tasks for the next agent

1. **Sync clean AIST++ data to A800 and restart AIST++-only training.**
   - Wait for the local WSL regeneration of `data/webbridge/aistpp_canonical/*.npz` to finish, then verify no NaN remains.
   - `rsync` the clean canonical directory to A800 `data/webbridge/aistpp_canonical/`.
   - Relaunch `scripts/run_aistpp_only_medium_a800_gpu5_fast_v2.sh` on A800 GPU 5 and confirm the first validation reports finite `val_loss` and `val_MPJPE`.

2. **Monitor the v25 stability variable-view eval and MPI RTMPose detection.**
   - Tail `outputs/variable_view_v25_true_gt_stability_a800_nohup.log` and record the final `MPJPE@k` curves.
   - Continue polling MPI RTMPose until 16/16 `.npz` files are ready; the CPU watcher will run the DLT baseline automatically.

3. **Prepare cross-dataset mixed training once AIST++-only converges.**
   - Evaluate the AIST++-only checkpoint on H36M S9/S11 true-GT test.
   - If loss/MPJPE are reasonable, launch a **H36M + AIST++ mixed-dataset** medium run using `configs/splits/h36m_true_gt_aist_mixed_train_val_a800.yaml`. Target: improve over v25 stability's **31.56 mm** test weighted.

3. **Finish MPI-INF-3DHP detected-2D validation.**
   - RTMPose detection is ongoing on GPU 7 (4/16 `.npz` files done). A CPU watcher is already running the DLT baseline on completed files (interim: **150.50 mm** MPJPE / **164.22 mm** PA-MPJPE).
   - Once all 16 files are ready, record the full DLT baseline in `docs/results_true_gt_h36m.md` and add v81/v82/v25-stability variable-view curves when their evals finish.

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
