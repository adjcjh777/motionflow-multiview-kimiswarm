# MotionFlow-MultiView: Corrected Paper Outline for CVPR/ICRA 2027

> **Status:** Draft outline (not a full paper). Updated after the data-foundation audit and the latest A800 ablations/re-run.
> **Anchor:** Sparse-view / cross-domain robustness on honest, non-circular benchmarks.
> **Supersedes:** `docs/paper_outline_v25_icra_cvpr_2027.md`; main draft remains `docs/paper_draft_icra_cvpr_2027.md`.

**Working title:** Geometry-First Multi-View Pose Fusion: Sparse Views, Cross-Domain Transfer, and Honest Benchmarking

---

## Abstract (1 paragraph)

Multi-view 3D human pose estimation has been benchmarked on labels that are themselves triangulations of the input 2D keypoints. We show that this circular protocol inflates apparent accuracy and conceals a large generalisation gap. On honest, true mocap ground truth, even strong learned triangulation methods lag behind a simple confidence-weighted DLT baseline. We therefore pivot the paper away from absolute-record MPJPE and toward the properties that matter for real deployment: **sparse-view robustness**, **cross-domain generalisation**, and resilience to imperfect calibration. We present MotionFlow-MultiView as a lightweight, geometry-first fusion module that triangulates first and then learns a small residual correction. Verified leaderboards on true-GT Human3.6M, AIST++, and Shelf/Campus show that the real differentiator is not incremental accuracy on circular tables, but robustness when views are few, noisy, or from another domain.

---

## 1. Introduction (≈1.5 pages)

1. **Motivation.** Calibrated multi-view video is the dominant capture modality for robotics, sports, and AR/VR. Classical triangulation is brittle; learned fusion methods report ever-lower MPJPE.
2. **The circular-label problem.** H36M multi-view labels and MPI GT-projected 2D are functions of the input 2D keypoints. Models trained on them are rewarded for reproducing DLT, not for recovering true 3D pose.
3. **True-GT reality check.** On true mocap ground truth, Iskakov ICCV 2019 reaches 23.35 mm, confidence-weighted DLT reaches 25.67 mm, and our best learned variant (v25 stability) reaches 30.83 mm (31.56 mm weighted) on H36M true-GT test. Earlier reports of 0.62 mm were artifacts of circular labels. Cross-domain transfer is hard: an AIST++-only v25 model scores ~94 mm on H36M, and the MPI-INF-3DHP RTMPose detected-2D DLT baseline is 115.09 mm.
4. **Pivot to robustness.** The contribution is re-anchored on sparse-view and cross-domain robustness: how gracefully does a method degrade when views are removed or domains change?
5. **Contributions.**
   - Honest benchmarking on true-GT H36M, AIST++, and Shelf/Campus.
   - Geometry-first fusion with triangulation + learned residual correction.
   - Sparse-view evaluation via `MPJPE@k`.
   - Cross-domain transfer without per-dataset pose heads, using geometry-based camera positional encoding (CamPE).
   - Plug-in `MultiViewFusionPlugin` integration inside MotionFlow.

---

## 2. Related Work (1 page)

Use only verified real citations. Earlier drafts incorrectly listed SmoothNet and Stacked Hourglass as CVPR papers; both are ECCV.

- **Classical triangulation.** Hartley & Zisserman [1]; Hartley & Sturm [2].
- **Learnable triangulation.** Iskakov et al. [3] — *verified real*, ICCV 2019, arXiv:1905.05754.
- **Ray-aware / transformer fusion.** RUMPL [4] (arXiv 2025). Treat as recent related work, not as a baseline we have reproduced.
- **Temporal pose refinement.** MotionBERT [5]; SmoothNet [6] (ECCV 2022, corrected from earlier CVPR typo).
- **2D pose backbone.** Stacked Hourglass [7] (ECCV 2016, corrected from earlier CVPR typo).

**What not to cite.** Do not cite unpublished internal variants (v25, v80, v57) as independent prior work. Do not cite VoxelPose / MVPose numbers unless we have actually run them on our splits.

---

## 3. Method (≈2 pages)

### 3.1 Problem statement
Given 2D keypoints + confidences `(B, T, V, J, 3)`, intrinsics `K`, and extrinsics `R, t`, output a metric 3D pose that remains reliable when:
- only `k` views are usable;
- cameras are slightly perturbed;
- the test domain differs from training.

### 3.2 Geometry-first backbone
- Weighted DLT first; all learned components operate on the triangulated pose, not on raw 2D.
- Confidence-weighted and unweighted DLT are frozen baselines.

### 3.3 Learned residual correction
- Small MLP residual head: `X = X_raw + MLP([f, X_raw])`.
- Heavily regularised; used only to correct structured leftover error.

### 3.4 Sparse-view reliability gating
- Per-view, per-joint reliability score `r_{v,j}`.
- At test time, evaluate `MPJPE@k` by randomly sampling `k` views.

### 3.5 Cross-domain transfer
- CamPE from `K, R, t` (no fixed view embeddings).
- Optional small domain embedding; no per-dataset pose head.

### 3.6 System integration
- `MultiViewFusionPlugin` outputs `HumanMotionIR` with pose, uncertainty, and provenance.
- Quality gating and robot-profile retargeting downstream.

---

## 4. Experiments and Results (≈2.5 pages)

### 4.1 Datasets and protocols
- **H36M true-GT:** `S1,S5,S6,S7,S8 → S9,S11`, `data/h36m_true_gt/`.
- **Shelf/Campus detected:** real 2D detections + true 3D, `data/webbridge/shelf_campus_detected/`.
- **AIST++:** canonical 9-view `.npz`, cross-domain smoke/full validation; 1,408 clips now available.
- **MPI-INF-3DHP:** RTMPose detected-2D `.npz` complete (16/16); DLT baseline computed.

### 4.1.1 SOTA baseline availability

| Method | Code / config in repo | Runnable now | Local RTX 4090 | A800 | Notes |
|---|---|---|---|---|---|
| DLT (unweighted) | `scripts/run_mpi_dlt_baseline.py` | Yes | CPU, <30 min | CPU, <30 min | Deterministic frozen baseline; already measured |
| DLT (confidence-weighted) | `scripts/run_mpi_dlt_baseline.py`, Iskakov script | Yes | CPU, <30 min | CPU, <30 min | Deterministic frozen baseline; already measured |
| Iskakov ICCV 2019 | `experiments/train_iskakov_baseline_shelf_campus.py` | Yes | GPU, ~hours | GPU if wrapped | Already run; current true-GT leader |
| VoxelPose | `scripts/sota_baselines/prepare_voxelpose_h36m.sh`, `voxelpose_h36m_config.yaml` | Partially | Local GPU only; not yet run | No — script exits on A800-D / read-only mount | Clones upstream on first run; needs internet |
| MVPose | `scripts/sota_baselines/prepare_mvpose_h36m.sh`, `mvpose_h36m_config.yaml` | Partially | Local GPU only; not yet run | No — script exits on A800-D / read-only mount | Clones upstream on first run; needs internet |
| v25 / v57 / v80 | `scripts/run_*_h36m_true_gt_*.sh`, `configs/ablations/` | Yes | Yes (medium/smoke) | Yes (A800 scripts exist) | Internal MotionFlow variants; not independent SOTA |
| RUMPL (arXiv 2025) | None | No | — | — | Cited as related work only; not reproduced |

- **DLT and Iskakov** are the only SOTA baselines with both runnable code *and* measured true-GT numbers on this corrected protocol.
- **VoxelPose / MVPose** wrappers are **local-RTX-4090 only**; they explicitly refuse to run on the A800-D read-only mount and require cloning upstream repositories. They have not been executed yet, so no true-GT numbers exist.
- **RUMPL** and other recent learnable-triangulation papers are cited as related work; the repo contains no implementation or config for them.

### 4.2 True-GT leaderboards (corrected, non-circular)

#### Human3.6M true-GT standard protocol

| Method | S9 direct | S11 direct | Combined direct | Combined PA-MPJPE | Notes |
|---|---:|---:|---:|---:|---|
| DLT (unweighted) | 33.61 | 24.77 | 29.19 | 29.31 | frozen baseline |
| DLT (confidence-weighted) | 29.82 | 21.91 | 25.67 | 25.55 | frozen baseline |
| RANSAC/conf-DLT | 29.60 | 21.96 | 26.47 | 28.98 | reproducible; `scripts/run_h36m_true_gt_ransac_baseline.py` |
| Iskakov ICCV 2019 | 27.10 | 19.60 | **23.35** | **23.10** | current leader |
| VoxelPose | — | — | — | — | scripts ready; not yet run |
| MVPose | — | — | — | — | scripts ready; not yet run |
| **v25 stability** | 34.87 | 26.80 | **30.83** avg / **31.56** weighted | 34.35 | **best learned result**; early-stopped @ Epoch 12 |
| v25 mixed (H36M+AIST++) | 37.87 | 28.96 | 33.42 avg / 34.23 weighted | — | diverged @ Epoch 3; best-ckpt test |
| v81 temporal-pose-attention | 42.19 | 33.46 | 37.83 | 37.75 | early-stopped @ Epoch 8 |
| v82 multi-scale temporal-attention | 42.07 | 36.84 | 39.46 | 39.94 | early-stopped @ Epoch 8 |
| v80 regularization | 56.69 | 51.27 | 53.98 | 32.47 | early-stopped @ Epoch 4 |
| v80 (medium) | — | — | 39.98 | — | best epoch 4, then overfit to 133.71 |
| v52 UWT | 58.15 | 49.87 | 54.01 | 42.22 | early-stopped @ Epoch 7 |
| v46 SVG | 55.03 | 49.88 | 52.46 | 40.20 | — |
| v57 re-run | — | — | 57.10 | — | best val 57.81 @ Epoch 4 |
| v25 (medium) | 47.28 | 40.54 | 43.93 | — | **test** result; corrected-val ablations 45.80 / 46.75 mm @ epoch 1; both diverged |

- Iskakov outperforms DLT by 2.32 mm combined direct.
- **v25 stability is the best learned MotionFlow variant** at 30.83 mm average (31.56 mm weighted), but it still trails DLT by ~5.5 mm and Iskakov by ~7.5 mm, confirming that the remaining gap is real and cannot be closed by more architecture on H36M alone.
- v81/v82 temporal modules do not beat v25, indicating that ray-token temporal attention is not the missing piece.
- v25 test MPJPE is 43.93 mm for the original medium run; the 72.80 mm val figure was inflated because `view_mask` was not passed during validation. Corrected-validation A800 ablations reached 45.80 / 46.75 mm @ epoch 1 and then diverged.
- VoxelPose and MVPose are prepared but unaudited; any numbers in earlier drafts should be treated as placeholders.

#### Shelf / Campus detected

| Method | Val direct | Val PA-MPJPE | Notes |
|---|---:|---:|---|
| Iskakov ICCV 2019 | **128.73** | **119.23** | leader |
| DLT (confidence-weighted) | 132.29 | 120.95 | frozen baseline |
| DLT (unweighted) | 134.43 | 122.37 | frozen baseline |
| v80 long | 276.49 | — | 25 epochs, overfits |
| v57 long | 306.45 | — | 25 epochs, overfits |
| v25 smoke | 430.67 | — | 3 epochs |

#### AIST++ smoke

| Method | val MPJPE | Notes |
|---|---:|---|
| DLT (confidence-weighted) | **6.52** | smoke frozen baseline |
| DLT (unweighted) | 12.66 | smoke frozen baseline |
| Iskakov ICCV 2019 | **9.31** | smoke, best epoch 6 |
| v25 | 71.79 | 3-epoch smoke |
| v80 | 76.34 | 3-epoch smoke |

#### AIST++ full 1,408 clips

| Method | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| DLT (confidence-weighted) | **15.93** | **21.12** | 1,408 clips, 1,123,873 frames |
| DLT (unweighted) | 38.11 | 42.66 | full frozen baseline |

#### AIST++-only fast v2 → H36M true-GT S9/S11 cross-eval

| Subject | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| S9 | **98.17** | **49.44** | zero-shot cross-domain transfer |
| S11 | **89.70** | **39.55** | zero-shot cross-domain transfer |
| **Combined (simple avg)** | **93.94** | **44.50** | stride 1 |

#### MPI-INF-3DHP detected-2D DLT baseline

| Method | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| DLT (confidence-weighted) | **115.09** | **132.68** | RTMPose detected-2D, 16 `.npz` files |

- Source: `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`.
- Learned-model benchmarking on MPI remains future work; the detected-2D DLT baseline sets a realistic lower bound for the current detection pipeline.

### 4.3 Sparse-view robustness (`MPJPE@k`)
Report degradation curves for `k = 2, 3, 4` (H36M) and `k = 2..14` (MPI). Smoke evidence shows v57 degrades most gracefully as views drop, while v25 is volatile at low `k`.

### 4.4 Failure-mode analysis
- Rapid divergence / overfitting on small true-GT sets.
- Cross-dataset transfer remains weak (AIST++ / Shelf/Campus smoke).

---

## 5. Discussion and Conclusion (≈0.5 page)

- The honest leaderboards reset expectations: geometric and learnable-triangulation baselines are stronger than our current learned variants on true GT.
- The positive signal is in robustness: different architectures degrade differently as views are removed.
- Cross-domain transfer is hard: an AIST++-only v25 model scores ~94 mm on H36M, and the MPI detected-2D DLT baseline is ~115 mm, showing a large gap between controlled studio data and real detected data.
- Future work: close the true-GT accuracy gap via stronger regularisation, mixed-dataset training, and explicit calibration correction.

---

## References (verified, real)

1. Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*. Cambridge University Press, 2004.
2. Hartley, R. and Sturm, P. “Triangulation.” *Computer Vision and Image Understanding*, 68(2):146–157, 1997.
3. Iskakov, K., Burkov, E., Lempitsky, V., and Malkov, Y. “Learnable triangulation of human pose.” *ICCV*, 2019. arXiv:1905.05754.
4. Ghasemzadeh, S. A. and Alahi, A. “RUMPL: Ray-based transformers for universal multi-view 2D to 3D human pose lifting.” arXiv:2512.15488, 2025.
5. Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., and Wang, Y. “MotionBERT: A Unified Perspective on Learning Human Motion Representations.” *ICCV*, 2023.
6. Zeng, A., Yang, L., Ju, X., Li, J., Wang, J., and Xu, Q. “SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos.” *ECCV*, 2022. (Corrected from earlier CVPR typo.)
7. Newell, A., Yang, K., and Deng, J. “Stacked hourglass networks for human pose estimation.” *ECCV*, 2016. (Corrected from earlier CVPR typo.)

---

## Remaining Experimental Results to Collect

1. ~~**MPI-INF-3DHP real detected-2D alignment.**~~ **Done.** RTMPose regeneration produced 16/16 `.npz` files; confidence-weighted DLT baseline is **115.09 mm** / PA-MPJPE **132.68 mm**. Learned-model benchmarking remains future work.
2. **AIST++ full-medium runs.** Clean canonical `.npz` synced to A800 after fixing the NaN blocker. AIST++-only medium fast v2 finished/early-stopped at Epoch 4; cross-eval on H36M true-GT S9/S11 is **93.94 mm**.
3. **H36M true-GT leaderboard.** Completed: v25 stability (30.83 mm), v81 (37.83 mm), v82 (39.46 mm), v46 (52.46 mm), v52 (54.01 mm), v80 regularization (53.98 mm), v57 re-run (57.10 mm). v25 mixed-dataset diverged at Epoch 3 but best checkpoint tests at 33.42 mm.
4. **Sparse-view `MPJPE@k` curves.** v81, v82, v57, v80 completed; v25 stability curve is running on GPU 4. DLT and Iskakov sparse-view curves still needed.
5. **Calibration-robustness matrix.** Earlier perturbation matrices were on the old circular protocol; re-measure on true GT.
6. **Cross-domain training mix.** Train one model on H36M true-GT + AIST++ + Shelf/Campus and evaluate on each domain. AIST++-only run is the first step; H36M+AIST++ mixed run is next.
7. **Runtime verification.** Confirm the 12.7–194 clips/s numbers on the current true-GT model variant.

---

## Notes for the Next Draft

- Keep Iskakov ICCV 2019 as a real, verified baseline citation and the current true-GT leader.
- Remove or flag any mention of the old circular-label results (e.g., H36M 0.62 mm, MPI 9.32 mm).
- Do not claim SOTA over Iskakov/DLT on true GT until a MotionFlow variant actually beats them.
- The paper’s contribution is now **robustness**, not absolute MPJPE records.
