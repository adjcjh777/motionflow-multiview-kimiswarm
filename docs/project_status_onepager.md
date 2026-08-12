# MotionFlow-MultiView Project Status

```text
  __  __           _      _   __  __      _   _ _    _ _  
 |  \/  | ___   __| | ___| | | \/  | ___ | |_(_) |__| (_) ___  _ _ 
 | |\/| |/ _ \ / _` |/ _ \ | | |\/| |/ _ \| / / | / _` | |/ _ \| '_|
 | |  | | (_) | (_| |  __/ | | |  | | (_) |>  <| | (_| | | (_) | |   
 |_|  |_\\___/ \__,_|\___|_| |_|  |_\\___/_/\_\_|\__,_|_|\___/|_|   
```

> **Date:** 2026-08-11
> **Goal:** CVPR 2027 submission (~Nov 2026) — sparse-view / cross-domain robust pose estimation
> **Phase:** true-GT protocol rebuild complete; fixing overfitting on real 3D labels
> **Repository:** `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`

---

## Compute Status

```
+------------- Local RTX 4090 --------------+   +----------- A800 / Docker -----------+
|  STATUS: IDLE  |  quick smoke / diagnostics  |   |  TRAINING ACTIVE on A800 host       |
|  v25 ablations done (45.80/46.75 mm)       |   |  GPU 5: v57 re-run; GPU 7: MPI RTMPose |
+---------------------------------------------+   +-------------------------------------+
```

| Active job | Script | Log | Status |
|---|---|---|---|
| v57 H36M true-GT medium (old) | `scripts/run_v57_h36m_true_gt_medium.sh` | `outputs/omniview_fusion_v57_h36m_true_gt_medium.log` | `DONE` (best epoch 3, observed val MPJPE 75.16 mm; saved ckpt 81.47 mm; final 80.21 mm) |
| v57 true-GT re-run | `scripts/run_v57_true_gt_medium_a800.sh` | `outputs/ablations/v57_true_gt_medium_a800.log` | `DONE` (best 57.81 mm @ epoch 4; early-stopped @ epoch 7; GPU 5 free) |
| v25 true-GT baseline fix | `scripts/run_v25_ablation_true_gt_baseline.sh` | `outputs/ablations/v25_true_gt_baseline_fix.log` | `DONE` (45.80 mm @ epoch 1; diverged; GPU 4 free) |
| v25 true-GT geometry regularization | `scripts/run_v25_ablation_true_gt_geometry_regularization_a800.sh` | `outputs/ablations/v25_true_gt_geometry_regularization_a800.log` | `DONE` (46.75 mm @ epoch 1; diverged; GPU 6 free) |
| MPI RTMPose detected-2D | `scripts/generate_mpi_detected_2d.py` / A800 | `outputs/generate_mpi_detected_2d_from_avi.log` | `RUNNING` on A800 GPU 7 (duplicate removed) |

---

## Data Foundation

```
+----------------+-------------------------------------------+-----------------+
|   Dataset      |               Status                        | True / Non-circ?| 
+----------------+-------------------------------------------+-----------------+
| H36M true GT   | READY: data/h36m_true_gt/                 |        ✅       |
| MPI-INF-3DHP   | Detected-2D `.npz` ready (16 files),    |       ⚠️        |
|                | but DLT baseline ~326–400 mm (alignment) |                |
| Shelf/Campus   | READY: shelf_campus_detected/             |        ✅       |
| AIST++         | smoke ready; full medium pending          |        ✅       |
+----------------+-------------------------------------------+-----------------+
```

---

## H36M True-GT Standard Protocol Leaderboard

```
   Standard split: S1,S5,S6,S7,S8  ->  S9,S11
   Labels: data/h36m_true_gt/*_multiview_m.npz
```

```
+-------------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| Method                  |  S9    |  S11   |  Comb  |  PA    |  Notes                                          |
|                         |  (mm)  |  (mm)  |  (mm)  |  (mm)  |                                                 |
+-------------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| Iskakov ICCV 2019       |  27.10 |  19.60 |  23.35 |  23.10 |  Leader; best epoch 4                           |
| DLT (conf-weighted)     |  29.82 |  21.91 |  25.67 |  25.55 |  Frozen ref                                     |
| DLT (unweighted)        |  33.61 |  24.77 |  29.19 |  29.31 |  Frozen ref                                     |
| v80 medium (local)      |   --   |   --   |  39.98 |   --   |  Best epoch 4; overfit to 133.71 mm             |
| v80 best conv. (v3)     |   --   |   --   |  42.60 |   --   |  Local 2-epoch best                             |
| v80 best (A800 v2)      |   --   |   --   |  39.70 |   --   |  Remote checkpoint (read-only)                  |
| v25 medium (test)       |  47.28 |  40.54 |  43.93 |   --   |  Test result; ablations 45.80/46.75 @ epoch 1  |
| v57 medium (old)        |   --   |   --   |  75.16 (obs.) / 81.47 (ckpt) |   --   |  Best epoch 3; early-stopped at epoch 5 (80.21 mm); saved ckpt is epoch 2 |
| v57 re-run              |   --   |   --   |  57.81 (epoch 4) |   --   |  Finished; early-stopped @ epoch 7; checkpoint saved correctly |
+-------------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
```

**Takeaway:** Geometric + learnable-triangulation baselines beat MotionFlow variants on true GT. v57 re-run finished at 57.81 mm @ epoch 4 (early-stopped @ epoch 7) with the checkpoint bug fixed; old run was 75.16 mm observed / 81.47 mm ckpt. v25/v80 still overfit/diverge.

---

## Shelf/Campus Detected Leaderboard

```
+-------------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| Method                  |  Direct|  PA    |  Notes                                          |
|                         |  (mm)  |  (mm)  |                                                 |
+-------------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| Iskakov ICCV 2019       | 128.73 | 119.23 |  Leader; early-stop epoch 11                    |
| DLT (conf-weighted)     | 132.29 | 120.95 |  Frozen ref                                     |
| DLT (unweighted)        | 134.43 | 122.37 |  Frozen ref                                     |
| v80 long (25 ep)        | 276.49 |   --   |  Best epoch 7, then overfit                     |
| v57 long (25 ep)        | 306.45 |   --   |  Best epoch 4, then overfit                     |
| v80 smoke               | 408.58 |   --   |  3-epoch smoke                                  |
| v57 smoke               | 424.63 |   --   |  3-epoch smoke                                  |
| v25 smoke               | 430.67 |   --   |  3-epoch smoke                                  |
+-------------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
```

---

## AIST++ Smoke (Cross-Dataset Sanity)

```
+-------------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| Method                  |  Val MPJPE (mm)  |  Notes                            |
+-------------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| DLT (conf-weighted)     |        6.52      |  Frozen ref                       |
| DLT (unweighted)        |       12.66      |  Frozen ref                       |
| Iskakov ICCV 2019       |        9.31      |  CPU smoke, best epoch 6          |
| v25                     |       71.79      |  3-epoch smoke                    |
| v80                     |       76.34      |  3-epoch smoke                    |
+-------------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
```

---

## Active Blockers & Next Steps

```
+------+--------------------------------+---------------------+---------------------+
| Pri  | Blocker                        | Why it matters      | Next action         |
+------+--------------------------------+---------------------+---------------------+
|  P0  | v25/v80/v57 overfit on true GT | Ablations done;     | Try mixed-dataset   |
|      |                                | regularisation not  | training or stronger|
|      |                                | enough              | architecture changes|
|  P1  | MPI detected-2D alignment      | RTMPose regen on    | Validate RTMPose    |
|      |                                | GPU 7; old DLT ~326 | results vs. ~20-30mm|
|  P1  | AIST++ full medium not run     | No convergence proof | GPU free -> full    |
|  P2  | VoxelPose / MVPose baselines   | SOTA comparison     | Add configs/scripts |
+------+--------------------------------+---------------------+---------------------+
```

---

## One-Liner Roadmap

1. **Done** v57 old medium (75.16 mm observed / 81.47 mm ckpt @ epoch 3) → recorded in H36M leaderboard.  
2. **Done** v25 true-GT ablations (45.80 / 46.75 mm @ epoch 1, then diverged) → GPUs 4/6 free.  
3. **Running** v57 re-run on A800 GPU 5 (57.81 mm @ epoch 4; 60.72 mm @ epoch 5).  
4. **Fix** the underlying divergence (mixed-dataset training / stronger regularisation).  
5. **Complete** MPI RTMPose detected-2D and AIST++ full medium.  
6. **Rewrite** paper story around sparse-view / cross-domain robustness.  

> Full details: `docs/handoff_qwen3.8max.md` · `docs/cvpr2027_status.md` · `docs/results_true_gt_h36m.md`
