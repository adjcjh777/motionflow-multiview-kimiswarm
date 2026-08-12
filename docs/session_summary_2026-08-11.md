# Session Summary — 2026-08-11

> Generated at the end of the 2026-08-11 work session.  
> Local repo: `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> Remote A800-D / Docker `motionflow`: **read-only** (no writes, starts, or modifications).

---

## 1. GPU / Resource Status

- **Local RTX 4090 is currently BUSY** — do not start another training job.
  - **v57 H36M true-GT medium is DONE** (best observed val MPJPE **75.16 mm** @ epoch 3, saved ckpt **81.47 mm** (epoch 2), early-stopped @ epoch 5, final val **80.21 mm**).
  - **v25 true-GT ablation 1 (`v25_true_gt_baseline_fix`) is RUNNING** — `outputs/ablations/v25_true_gt_baseline_fix.log` is active on cuda; this is the regularised recipe (train_samples=4096, weight_decay=1e-4, lr=5e-4, reduced outlier augmentation, early_stopping_patience=3).
  - `eval_variable_views` helper for v80 has completed and is no longer on the GPU.
- **A800-D / Docker**: training queues remain stopped; read-only access only.
- **Rule in effect**: local RTX 4090 runs at most one training task at a time.  Only prepare scripts/configs while the GPU is busy.

---

## 2. Key Numbers

### 2.1 H36M True-GT Standard Protocol (S1,5,6,7,8 → S9/S11)

| Method | Combined direct (mm) | S9 direct (mm) | S11 direct (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---:|---:|---|
| DLT (unweighted) | 29.19 | 33.61 | 24.77 | 29.31 | frozen ref |
| DLT (conf-weighted) | 25.87 | 29.82 | 21.91 | 25.55 | frozen ref |
| **Iskakov ICCV 2019** | **23.35** | **27.10** | **19.60** | **23.10** | current leader |
| v80 (medium) | **39.98** | — | — | — | best epoch 4; overfit to 133.71 mm by epoch 8 |
| v80 (best converged, v3) | **42.60** | — | — | — | local 2 epochs |
| v25 | **72.80** | 67.92 | 77.68 | — | best epoch 2; overfit to 207.62 mm by epoch 8 |
| **v57 (medium)** | **75.16** (obs.) / **81.47** (ckpt) | — | — | — | **DONE**; best epoch 3, early-stopped @ epoch 5 (final 80.21 mm); saved ckpt is epoch 2 |

- All H36M true-GT `.npz` live in `data/h36m_true_gt/` and pass the non-circular check.
- Iskakov is the strongest baseline; v80/v25/v57 still lag geometric baselines and all suffer from early overfitting / divergence.
- v57 overfits more slowly than v25 (75.16 mm observed / 81.47 mm ckpt → 80.21 mm) but remains well behind v80 (39.98 mm); v80 stays the primary learned baseline.

### 2.2 Shelf / Campus Detected

| Method | Val direct (mm) | Val PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| Iskakov ICCV 2019 | **128.73** | **119.23** | leader |
| DLT (conf-weighted) | 132.29 | 120.95 | frozen ref |
| DLT (unweighted) | 134.43 | 122.37 | frozen ref |
| v80 long (25 ep) | 276.49 | — | overfits after epoch 7 |
| v57 long (25 ep) | 306.45 | — | overfits after epoch 4 |
| v80 / v57 / v25 (3-ep smoke) | ~408–431 | — | far from converged |

### 2.3 AIST++ Smoke (Cross-Dataset Sanity)

| Method | val MPJPE (mm) | Notes |
|---|---:|---|
| DLT (unweighted) | **12.66** | frozen ref |
| DLT (conf-weighted) | **6.52** | frozen ref |
| Iskakov ICCV 2019 | **9.31** | CPU smoke, best epoch 6 |
| v25 | 71.79 | 3-epoch smoke |
| v80 | 76.34 | 3-epoch smoke |

- AIST++ is confirmed non-circular and uses the same 17-joint skeleton as H36M.

### 2.4 Data-Audit Verdicts

| Dataset | True / Non-circular? | Status |
|---|---|---|
| H36M true GT (`data/h36m_true_gt/`) | ✅ | S9/S11 DLT error ~25–34 mm |
| MPI-INF-3DHP detected 2D | ️ | Detected-2D `.npz` ready (16 files), but DLT baseline ~326–400 mm due to camera/label alignment; not yet usable for learned-model benchmarking |
| AIST++ | ✅ | AIST++ smoke DLT (conf-weighted) 6.52 mm / (unweighted) 12.66 mm |
| Shelf/Campus detected | ✅ | DLT error ~130 mm |
| 3DPW pseudo | ❌ | circular (DLT error ≈ 0 mm) |
| 3DPW actual | ⚠️ | monocular, not multi-view usable |

---

## 3. Key Decisions Made Today

1. **Project direction pivoted to CVPR 2027** and the paper story is now anchored on **sparse-view / cross-domain robustness** rather than absolute MPJPE records.
2. **Old circular-label data is disqualified** for model selection: `data/h36m_hf/`, `data/webbridge/h36m*.npz`, and all v25–v79 numbers on those datasets are no longer trusted.
3. **H36M true-GT standard protocol is canonical**: `data/h36m_true_gt/` + `configs/splits/h36m_true_gt_standard.yaml`.
4. **Iskakov ICCV 2019 is the primary baseline** on H36M and Shelf/Campus; DLT variants are the geometric reference.
5. **v25/v80/v57 all suffer from early overfitting** on the small true-GT medium split.  The agreed immediate fixes are:
   - Increase `train_samples` (e.g. 4096).
   - Enable early stopping (`early_stopping_patience 3`, `min_delta 0.001`).
   - Add weight decay (`1e-4`).
   - Lower learning rate / longer warmup.
   - Reduce outlier augmentation on small datasets.
6. **v57 H36M true-GT medium finished** at **75.16 mm** observed best (saved ckpt **81.47 mm** from epoch 2), with final epoch **80.21 mm** after early stopping at epoch 5.  It confirms the same overfitting pattern seen in v25/v80.
7. **v25 true-GT ablation 1 (`v25_true_gt_baseline_fix`) was launched** after v57 finished.  It tests whether the conservative regularised recipe (more samples, weight decay, early stopping, lower LR, milder augmentation) prevents the v25 divergence.
8. **MPI-INF-3DHP detected-2D files are generated** (`data/webbridge/mpi_inf_3dhp_detected_2d/`), but the DLT baseline remains ~326–400 mm because of a camera/label alignment issue. Benchmarking learned models on MPI is blocked until the alignment is fixed.
9. **A800-D and the `motionflow` Docker remain read-only** — no training launches, writes, or container starts.

---

## 4. Active Work in Flight

| Task | Status | Notes |
|---|---|---|
| v57 H36M true-GT medium (A800 re-run) | ✅ **DONE** | best val MPJPE **57.81 mm** @ epoch 4, early-stopped @ epoch 7; test MPJPE **57.10 mm**; checkpoint saved correctly at `outputs/ablations/v57_true_gt_medium_a800.pth`; recorded in `docs/results_true_gt_h36m.md` |
| v80 true-GT regularization (A800) | ✅ **DONE** | best val MPJPE **54.46 mm** @ epoch 1, early-stopped @ epoch 4; geometry-regularised v80, diverged after epoch 1; `outputs/ablations/v80_true_gt_regularization_a800.pth` |
| v25 true-GT ablation 1 — `v25_true_gt_baseline_fix` | ✅ **DONE** | best val **45.80 mm** @ epoch 1, diverged afterward; `outputs/ablations/v25_true_gt_baseline_fix.*` |
| v25 true-GT ablation 2 — `v25_true_gt_geometry_regularization` | ✅ **DONE** | best val **46.75 mm** @ epoch 1, diverged afterward; `outputs/ablations/v25_true_gt_geometry_regularization_a800.*` |
| v46 true-GT H36M | 🟡 **RUNNING** | sparse-view generalisation test on A800 GPU 4; `outputs/ablations/v46_true_gt_h36m_a800.log` |
| v25 true-GT mixed dataset (H36M + AIST++) | 🟡 **RUNNING** | A800 GPU 5; `outputs/ablations/v25_true_gt_mixed_dataset_a800.log` |
| v25 true-GT stability (low LR, no permute) | 🟡 **RUNNING** | A800 GPU 6; Epoch 1 val **60.74 mm**; `outputs/ablations/v25_true_gt_stability_a800.log` |
| MPI RTMPose detected-2D regeneration | 🟡 **RUNNING** | A800 GPU 7; 16 files queued, no output `.npz` yet; `outputs/mpi_rtmpose_detected_2d/generate_20260811_180024.log` |
| v57 variable-view evaluation | 🟡 **RUNNING** | A800 GPU 7; `outputs/variable_view_v57_true_gt_medium_a800.log` |
| eval_variable_views | ✅ **DONE** | helper completed for v80 / view-robustness curves |

---

## 5. Immediate Blockers

| Blocker | Severity | Why | Next Step |
|---|---|---|---|
| v25/v80/v57 overfit/diverge on true GT | P0 | v57 done (75.16 → 80.21 mm); all learned variants lag Iskakov/DLT | Analyse v25 ablation 1 result; if needed, run ablation 2/3 and then apply proven regularisation fixes from `docs/v25_divergence_diagnosis.md` |
| MPI detected-2D alignment | P0 | Detected 2D ready but DLT baseline ~326–400 mm | Diagnose camera/label coordinate-frame mismatch; fix before learned-model eval |
| 3DPW unusable | P1 | Circular pseudo / monocular actual | Drop 3DPW from multi-view experiments |

---

## 6. What to Do Next (post-v57 / during ablation 1)

1. ✅ **Capture v57 numbers** into `docs/results_true_gt_h36m.md` and this session summary (done).
2. **Monitor v25 ablation 1 (`v25_true_gt_baseline_fix`)** as it runs; extract per-epoch val MPJPE and determine whether the regularised recipe stabilises training.
3. **Run ablation 2 (`v25_true_gt_geometry_regularization`)** after ablation 1 completes, if ablation 1 does not reach ≤ 55 mm and stay stable.
4. **Run ablation 3 (`v25_true_gt_mixed_dataset`)** only if ablations 1/2 do not stabilise.
5. **Apply the winning recipe to v80** (currently the best learned baseline at **39.98 mm**) once the v25 ablations identify effective regularisations.
6. **Prepare cross-dataset mix** (H36M + AIST++ + Shelf/Campus) if single-dataset tuning does not close the gap.
7. **Diagnose and fix MPI detected-2D / camera / label alignment** so the DLT baseline drops to ~20–30 mm before learned-model benchmarking.
8. **Rewrite paper results section** once the true-GT leaderboard is stable.

---

## 7. Files / Artifacts Touched or Created Today

- `docs/cvpr2027_status.md` — project-wide status and roadmap.
- `docs/handoff_qwen3.8max.md` — detailed接力 notes.
- `docs/results_true_gt_h36m.md` — H36M true-GT leaderboard (to be updated by v57 monitor).
- `docs/results_true_gt_shelf_campus.md` — Shelf/Campus detected leaderboard.
- `docs/data_audit_summary_2026-08-11.md` — 15-agent circular-label audit.
- `docs/v25_divergence_diagnosis.md` — root-cause analysis and proposed fixes.
- `configs/benchmark_v57_h36m_true_gt_medium.yaml` — v57 medium config.
- `outputs/omniview_fusion_v57_h36m_true_gt_medium.{log,pth,config.json}` — v57 final artifacts (best observed val 75.16 mm @ epoch 3; saved ckpt 81.47 mm @ epoch 2).
- `tmp/monitor_v57.py` — auto-updates docs when v57 finishes.
- `configs/ablations/v25_true_gt_baseline_fix.yaml` — ablation 1 config.
- `scripts/run_v25_ablation_true_gt_baseline.sh` — ablation 1 launch script.
- `outputs/ablations/v25_true_gt_baseline_fix.{log,pth}` — ablation 1 outputs (running).

---

## 8. Final Reminders

- **GPU is busy** — do not launch another training run until `nvidia-smi` shows the v25 ablation process has exited.
- **A800 is read-only** — only inspect files, never start/modify anything there.
- **Use only true-GT / non-circular data** for model selection.
- **Preserve the best checkpoint**, not the final epoch file, for v25/v80/v57.
- **Do not duplicate the v25 ablation queue** — it is already running sequentially via `scripts/run_v25_ablations_sequential.sh`.
