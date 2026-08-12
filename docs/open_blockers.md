# Open P0/P1 Blockers

> Last updated: **2026-08-11**  
> Scope: MotionFlow-MultiView CVPR 2027 pipeline  
> Status convention: 🔴 Open / 🟡 In progress / 🟢 Resolved

This document tracks the open blockers that must be resolved before reliable model iteration and paper writing can continue. It synthesizes information from `docs/data_foundation_blocker.md`, `docs/cvpr2027_status.md`, `docs/handoff_qwen3.8max.md`, and the per-blocker notes under `docs/blockers/`.

---

## 🟡 P0 — H36M true-GT learned-model divergence / overfitting

| Field | Details |
|---|---|
| **ID** | P0-4 |
| **Owner** | **qwen3.8max** |
| **Impact** | All MotionFlow variants (v25, v80, v57) fail to match DLT / Iskakov on the corrected H36M standard protocol. Prevents any meaningful architecture comparison. |
| **Evidence** | v57 medium is now complete: best observed **75.16 mm** @ epoch 3 (saved ckpt **81.47 mm** from epoch 2), early stopped at epoch 5 (final 80.21 mm). v25 medium: best 72.80 mm @ epoch 2, diverges to 207.62 mm @ epoch 8. v80 medium: best 39.98 mm @ epoch 4, overfits to 133.71 mm @ epoch 8. Iskakov baseline is at 23.35 mm; confidence-weighted DLT is at 25.87 mm. Ablation 1 (`v25_true_gt_baseline_fix`) is running on the local RTX 4090 with `train_samples=4096`, `weight_decay=1e-4`, `lr=5e-4`, reduced outlier augmentation, and early-stopping patience=3. |
| **Root cause (preliminary)** | `docs/v25_divergence_diagnosis.md` / `docs/true_gt_overfitting_diagnosis.md` identify too few `train_samples` (1024/epoch), missing weight decay / early stopping, strong augmentation, and high learning rate. |
| **Next step** | 1. Wait for `v25_true_gt_baseline_fix` to finish, then compare its curve against the v25/v80/v57 baselines. 2. Run the queued follow-ups: `v25_true_gt_geometry_regularization`, then `v25_true_gt_mixed_dataset`. 3. If a fix stabilizes validation MPJPE, re-run the corresponding medium and update `docs/results_true_gt_h36m.md`. |
| **Dependencies** | None (P0-5 resolved). |

---

## 🟡 P1 — MPI-INF-3DHP detected-2D / camera / label alignment

| Field | Details |
|---|---|
| **ID** | P1-1 (formerly P0-2) |
| **Owner** | **unassigned** (data / webbridge) |
| **Impact** | Real detected-2D `.npz` exist, but the DLT baseline is ~326–400 mm, so the data cannot yet provide a meaningful lower bound for learned models. |
| **Evidence** | `data/webbridge/mpi_inf_3dhp_detected_2d/` contains 16 `_m.npz` files. `s_02_seq_02` was removed after showing ~2.76 m MPJPE. DLT baseline on the remaining files is ~326–400 mm, and a spot check shows ~189 px reprojection error for true 3D vs. stored cameras. Non-circularity check passes, so the error is a real coordinate-frame mismatch. See `docs/results_mpi_detected_dlt.md`. |
| **Next step** | 1. Diagnose why the true 3D labels do not reproject onto the stored cameras / detected 2D. 2. Fix the coordinate-frame conversion in the MPI loader/canonical builder. 3. Re-generate the affected `.npz` files. 4. Re-run `scripts/run_mpi_dlt_baseline.py` until DLT baseline drops to ~20–30 mm. |
| **Dependencies** | None |

---

## 🟡 P1 — v25 training crash on Shelf/Campus non-circular smoke

| Field | Details |
|---|---|
| **ID** | P1-2 |
| **Owner** | **unassigned** (geometry-fusion / view-embedding) |
| **Impact** | Cannot train v25 on Shelf/Campus, blocking cross-domain training and the sparse-view robustness story. |
| **Evidence** | `bash scripts/run_v25_shelf_campus_noncircular_smoke_local_4090.sh` crashes inside `motionflow_mv/fusion/epipolar_attention_bias.py::compute_epipolar_distance` with `CUDA error: device-side assert triggered` / `srcIndex < srcSelectDimSize`. The error persists even with `--use_epipolar_bias false`, suggesting a hard-coded view-count assumption elsewhere. See `docs/blockers/shelf_campus_v25_training_crash.md`. |
| **Next step** | 1. Re-run with `CUDA_LAUNCH_BLOCKING=1` to locate the exact assert. 2. Bisect by disabling `--use_deformable_cross_view_attention_v18` and other cross-view modules one-by-one. 3. As a workaround, evaluate Shelf/Campus only via `experiments/eval_shelf_campus_standard.py` until training is fixed. |
| **Dependencies** | None |

---

## 🟡 P1 — AIST++ integration pending full medium validation

| Field | Details |
|---|---|
| **ID** | P1-3 |
| **Owner** | **agent-67** (when available) |
| **Impact** | AIST++ smoke shows v25/v80 far behind DLT/Iskakov, but 3-epoch smoke is insufficient to judge convergence or cross-domain value. |
| **Evidence** | Smoke results in `docs/results_true_gt_h36m.md`: DLT (conf) 6.52 mm, Iskakov 9.31 mm, v25 71.79 mm, v80 76.34 mm. Source logs exist in `outputs/`. |
| **Next step** | 1. Run v25/v80 medium on AIST++ only (`configs/splits/aist_only_smoke.yaml`). 2. If numbers still diverge, apply the same regularization fixes from P0-4. 3. Add AIST++ to the cross-domain training mix (P1-5). |
| **Dependencies** | P0-4 (divergence fix) recommended before committing GPU to medium run |

---

##  P1 — Additional SOTA baselines not yet reproduced

| Field | Details |
|---|---|
| **ID** | P1-4 |
| **Owner** | **unassigned** (baselines / SOTA) |
| **Impact** | Paper story lacks strong comparison points beyond DLT and Iskakov. Reviewers will expect VoxelPose, MVPose, or comparable multi-view baselines. |
| **Evidence** | Iskakov ICCV 2019 is already reproduced. VoxelPose / MVPose are mentioned in `docs/handoff_qwen3.8max.md` as not yet run. |
| **Next step** | 1. Identify which SOTA methods have compatible code or checkpoints. 2. Add training/eval scripts under `scripts/sota_baselines/` or `experiments/`. 3. Run them on H36M true-GT and Shelf/Campus detected under the same protocol. 4. Update `docs/results_true_gt_h36m.md` and `docs/results_true_gt_shelf_campus.md`. |
| **Dependencies** | None |

---

##  P1 — Cross-domain training manifest and recipe

| Field | Details |
|---|---|
| **ID** | P1-5 |
| **Owner** | **unassigned** (data + modeling) |
| **Impact** | The revised paper contribution depends on sparse-view / cross-domain robustness, but no unified training mix exists yet. |
| **Evidence** | H36M true GT, AIST++, and Shelf/Campus detected are each available, but no combined manifest or training recipe has been created. `docs/cvpr2027_status.md` lists cross-domain mix as a next step. |
| **Next step** | 1. Create a combined manifest (e.g., `configs/splits/h36m_aist_shelf_campus_mix.yaml`). 2. Resolve domain-embedding sizing and view-count mismatches. 3. Run a short smoke on the mix and measure per-domain val MPJPE. |
| **Dependencies** | P1-1 (MPI detected 2D) optional but useful; P1-3 (AIST++ medium); P1-2 (Shelf/Campus v25 crash) for v25 on the mix |

---

## 🟡 P1 — Paper rewrite around sparse-view / cross-domain robustness

| Field | Details |
|---|---|
| **ID** | P1-6 |
| **Owner** | **unassigned** (paper lead) |
| **Impact** | The old narrative (absolute MPJPE record) is invalidated by the circular-label fix. The paper must be reframed before any results section is final. |
| **Evidence** | `docs/handoff_qwen3.8max.md` §6.7 and `docs/cvpr2027_status.md` §6 call for rewriting the paper story, tables, and citations. `docs/paper_draft_icra_cvpr_2027.md` has already had fabricated citations removed. |
| **Next step** | 1. Draft a new abstract/intro emphasizing sparse-view and cross-domain robustness. 2. Update tables with true-GT numbers. 3. Add real citations (Iskakov ICCV 2019, VoxelPose, MVPose, etc.). 4. Align figure captions and ablation story with the new narrative. |
| **Dependencies** | P0-4, P0-5 (H36M leaderboard stable); P1-3, P1-5 (cross-domain evidence) |

---

## Summary table

| ID | Priority | Title | Owner | Status |
|---|---|---|---|---|
| P0-4 | P0 | H36M true-GT learned-model divergence / overfitting | qwen3.8max | 🟡 In progress |
| P1-1 | P1 | MPI-INF-3DHP detected-2D / camera / label alignment | unassigned (data) | 🟡 In progress / blocked on alignment fix |
| P1-2 | P1 | v25 crash on Shelf/Campus non-circular smoke | unassigned (geometry-fusion) | 🟡 In progress |
| P1-3 | P1 | AIST++ full medium validation | agent-67 | 🟡 In progress |
| P1-4 | P1 | Additional SOTA baselines (VoxelPose, MVPose, etc.) | unassigned (baselines) | 🟡 Open |
| P1-5 | P1 | Cross-domain training manifest and recipe | unassigned (data + modeling) | 🟡 Open |
| P1-6 | P1 | Paper rewrite around sparse-view / cross-domain robustness | unassigned (paper lead) | 🟡 Open |

---

## Resolved blockers (for context)

| ID | Title | Resolution |
|---|---|---|
| P0-5 | v57 H36M true-GT medium status unconfirmed | ✅ Resolved. v57 medium completed: best observed **75.16 mm** @ epoch 3 (saved ckpt **81.47 mm** from epoch 2), early stopped at epoch 5 (final 80.21 mm). Results recorded in `docs/results_true_gt_h36m.md`. |
| P0-1 | H36M true 3D GT missing | ✅ Resolved. `data/h36m_true_gt/` is in place and validated. |
| P0-2 | MPI-INF-3DHP real detected 2D missing | ✅ Resolved: 16 `_m.npz` files generated in `data/webbridge/mpi_inf_3dhp_detected_2d/`. Outstanding issue is camera/label alignment (now P1-1). |
| P0-3 | Standard SOTA baselines not reproduced | Iskakov ICCV 2019 reproduced; VoxelPose/MVPose remain open (now P1-4). |
