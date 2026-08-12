# MotionFlow-MultiView: CVPR 2027 Data-Foundation Pivot

> Onboarding note for new collaborators — last updated 2026-08-12.

## TL;DR

We are rebuilding the project’s evaluation foundation for **CVPR 2027**. The previous leaderboards were contaminated by **circular 3D labels** in H36M, so all v25–v79 smoke and medium numbers are no longer model-selection criteria. The new direction is **sparse-view / cross-domain robust multi-view pose estimation**, not absolute MPJPE records.

## What changed

- **H36M labels were circular.** The old `.npz` files (`data/h36m_hf/`, `data/webbridge/h36m*.npz`) stored the DLT triangulation of the input 2D as `joints_3d`, so `direct MJE ≈ 0 mm`. Models were being scored on how well they reproduced the DLT layer, not on real pose accuracy.
- **True H36M 3D GT is now in place.** Standard protocol (train S1,5,6,7,8 → test S9/S11) is under `data/h36m_true_gt/`, with manifest `configs/splits/h36m_true_gt_standard.yaml`.
- **Shelf/Campus and AIST++ were also rebuilt with non-circular labels.**
  - Shelf/Campus detected: `data/webbridge/shelf_campus_detected/`
  - AIST++ smoke manifest: `configs/splits/aist_only_smoke.yaml`
  - AIST++ full manifest synced to A800 for cross-domain medium runs.

## Current true-GT leaderboards

### H36M true GT (S1,5,6,7,8 → S9/S11)

| Method | Combined direct (mm) | Notes |
|---|---:|---|
| **Iskakov ICCV 2019** | **23.35** | current leader |
| DLT (conf-weighted) | 25.67 | frozen geometric baseline |
| DLT (unweighted) | 29.19 | frozen geometric baseline |
| v80 medium | 39.98 | best epoch 4; overfit to 133.71 by epoch 8 |
| v25 medium (test) | 43.93 | best epoch 2; corrected-val ablations 45.80 / 46.75 mm @ epoch 1 |
| v57 medium | 80.21 | final; true best 75.16 mm @ epoch 3 was not saved; re-run 57.81 mm @ epoch 4 (in progress) |

- Full table: `docs/results_true_gt_h36m.md`
- Key implication: learned models **diverge / overfit** on the small true-GT training set. The next research priority is diagnosing and fixing this, not adding more architecture.

### Shelf/Campus detected

| Method | Val direct (mm) | Notes |
|---|---:|---|
| Iskakov ICCV 2019 | **128.73** | leader |
| DLT (conf-weighted) | 132.29 | frozen baseline |
| v80 / v57 / v25 | 408–431 | 3-epoch smoke only; far from converged |

- Full table: `docs/results_true_gt_shelf_campus.md`

### AIST++ smoke (first 3-epoch smoke)

| Method | val MPJPE (mm) |
|---|---:|
| DLT (conf-weighted) | **6.52** |
| Iskakov ICCV 2019 | 9.31 |
| DLT (unweighted) | 12.66 |
| v25 | 71.79 |
| v80 | 76.34 |

## New paper direction

- **Old claim:** “We beat DLT / SOTA on absolute MPJPE.”
- **New claim:** “Reliable multi-view pose under sparse views and cross-domain inputs.”
- Emphasis: robustness curves (variable views, camera perturbation, occlusion), cross-dataset training, and reproducible standard-protocol numbers.

## Immediate blockers

1. **v25/v80/v57 overfitting on H36M true GT.** All learned variants diverge after the first one or two epochs. The corrected-validation v25 ablations reached 45.80 / 46.75 mm @ epoch 1 and then blew up; v80 peaks at 39.98 mm @ epoch 4 and then overfits. Need mixed-dataset training or stronger regularisation.
2. **v57 H36M true-GT re-run in progress.** A fresh run with the trainer `mpjpe` checkpoint monitor reached **57.81 mm** @ epoch 4 and **60.72 mm** @ epoch 5 (A800 GPU 5), already beating the lost best of 75.16 mm.
3. **MPI-INF-3DHP detected-2D alignment.** Real detected-2D regeneration with RTMPose is running on A800 GPU 7; duplicate process removed. DLT baseline on the old MediaPipe detections is still ~326–400 mm, so learned-model MPI benchmarking remains blocked until RTMPose results are validated.
4. **Limited SOTA baselines.** Iskakov is done; VoxelPose / MVPose still pending.

## How to verify data

```bash
# Check if a .npz is circular / pseudo-GT
python scripts/diagnose_circular_labels.py <path/to/file.npz>

# Audit true-GT reprojection consistency
python scripts/check_true_gt_reprojection.py data/h36m_true_gt/s_09_act_02_multiview_m.npz
```

## Important constraints

- **Do not use old H36M `.npz` files for model selection.** Only `data/h36m_true_gt/` and `data/webbridge/shelf_campus_detected/` are trusted.
- **Local GPU (RTX 4090) runs one training task at a time.** Run `nvidia-smi` before launching any GPU job.
- **A800-D and the `motionflow` Docker service are read-only.** You may inspect files, but do not write, start, or modify anything there.
- **Do not duplicate work of other agents.** Check `docs/cvpr2027_status.md` and `docs/handoff_qwen3.8max.md` before starting a new task.

## Useful links

- Latest detailed status: `docs/cvpr2027_status.md`
- Latest handoff: `docs/handoff_qwen3.8max.md`
- H36M true-GT results: `docs/results_true_gt_h36m.md`
- Shelf/Campus results: `docs/results_true_gt_shelf_campus.md`
- Data audit summary: `docs/data_audit_summary_2026-08-11.md`
