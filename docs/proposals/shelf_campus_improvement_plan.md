# Shelf / Campus Improvement Plan

> **Status:** draft improvement roadmap, no jobs launched.  
> **Goal:** close the gap between MotionFlow-MultiView variants and the
> Iskakov/DLT triangulation baseline on the non-circular detected
> Shelf/Campus protocol (`data/webbridge/shelf_campus_detected/`).

## Current state (verified 2026-08-12)

| Method | Val direct MPJPE (mm) | Params | Notes |
|--------|----------------------|--------|-------|
| Iskakov ICCV 2019 (learnable triangulation) | **128.73** | ~1.5 k | current best |
| Conf-weighted DLT | **132.29** | — | strong frozen reference |
| Unweighted DLT | 134.43 | — | baseline triangulation |
| v80 long (25 epochs) | 276.49 | 964 k | best @ epoch 7, then overfits |
| v57 long (25 epochs) | 306.45 | — | best @ epoch 4, then overfits |
| v80 smoke (3 epochs) | 408.58 | 964 k | far behind triangulation |
| v25 smoke (3 epochs) | 430.67 | 2.7 M | far behind triangulation |

- **Protocol:** `configs/splits/shelf_campus_detected_smoke.yaml`
  - Train: Shelf seq1 (346 frames, 5 views) + Campus seq1 (652 frames, 3 views)
  - Val: Shelf seq1 val (87 frames) + Campus seq1 val (164 frames)
  - 17 COCO-style joints, units in metres, true 3D GT from `annotation_3d.json`.
- **Data size:** only 998 training frames, 251 validation frames.
- **Key problem:** heavy learned-fusion models (v25/v57/v80) severely overfit on this tiny dataset and produce worse-than-triangulation results.

## Root-cause analysis

1. **Massive capacity–data mismatch.**  
   v25/v80 have 1–3 M parameters but only ~1 k training frames. Even 3-epoch smoke runs do not have enough data to justify the model size.
2. **Architecture tuned for H36M.**  
   Many modules assume 4-view H36M rigs; Shelf has 5 views, Campus has 3 views, and the model has no room to learn robust cross-view patterns from such few examples.
3. **High detection noise.**  
   DLT itself is ~132 mm, so the 2D COCO detections are noisy. A heavy model can amplify this noise instead of filtering it.
4. **No temporal or pose prior.**  
   Frame-wise training ignores the sequential nature of the data and does not exploit kinematic or bone-length constraints.
5. **No cross-dataset transfer.**  
   The manifest `configs/splits/mix_h36m_aist_shelf.yaml` exists for mixed training, but it has not been exploited for a dedicated Shelf/Campus fine-tuning recipe.

## Prioritized improvement plan

### P0: Match the strong triangulation baseline first

**Objective:** Build a lightweight learned method that at least matches Iskakov (128.73 mm) and ideally beats DLT (~132 mm) before investing in heavier designs.

**Actions:**
1. **Iskakov + small MLP refinement.**
   - Use `outputs/iskakov_shelf_campus_detected.pth` to generate initial 3D poses.
   - Train a tiny per-joint residual MLP (≤ 10 k params) that takes the triangulated 3D pose + per-joint uncertainty + camera features and predicts a residual correction.
   - Target: < 120 mm direct MPJPE with minimal overfit risk.
   - Script: `experiments/train_iskakov_refiner_shelf_campus.py` (new).

2. **DLT + kinematic/temporal post-filter.**
   - Run confidence-weighted DLT per frame.
   - Apply a light 1D temporal smoothing/Bone-length-aware Kalman or trajectory filter per joint.
   - Target: 125–130 mm with almost zero trainable parameters.

### P1: Reduce model capacity and regularise heavily for the small dataset

**Objective:** Find the smallest MotionFlow variant that does not overfit.

**Actions:**
1. **Mini-v80 smoke sweep.**
   - Reduce `d` to 32, `residual_hidden` to 64, `n_st_layers` to 1, disable non-essential modules (epipolar bias, deformable attention, variable-view training).
   - Add heavy regularisation: `dropout=0.5`, `weight_decay=1e-3`, early-stop patience 3, EMA on validation.
   - Run a 4-GPU-free grid on local RTX 4090 (smoke, 3–5 epochs).
   - Target: reach < 200 mm before moving to cross-dataset training.

2. **Strong augmentation-only baseline.**
   - Synthetic 2D jitter, random view dropout, and scale augmentation during Shelf/Campus-only training.
   - Measure if augmentation alone closes any part of the gap.

### P2: Cross-dataset pre-training + fine-tuning

**Objective:** Use large H36M/AIST++ data to learn generic multi-view geometry, then fine-tune on Shelf/Campus.

**Actions:**
1. **Pre-train on H36M true-GT + AIST++ only.**
   - Use the proven v25/v80 H36M recipe (which reaches ~31–40 mm on H36M true-GT).
   - Do not include Shelf/Campus in pre-training yet.

2. **Fine-tune on Shelf/Campus.**
   - Load H36M/AIST++ pre-trained weights, freeze the bulk of the backbone, and fine-tune only the output/refiner layers.
   - Use very low LR (1e-5), small batch, and strong regularisation.
   - Manifest: `configs/splits/mix_h36m_aist_shelf.yaml` or a dedicated stage-wise manifest.
   - Target: beat Iskakov (128.73 mm); stretch goal < 110 mm.

### P3: Temporal and test-time refinement

**Actions:**
1. **Temporal Iskakov.**
   - Extend the Iskakov baseline to a small temporal model (e.g. 1D conv over 5–9 frames) that predicts per-view weights using neighbouring frames.
   - Target: modest but consistent gain over frame-wise Iskakov.

2. **Test-time bundle adjustment / kinematic fitting.**
   - After any 3D prediction, run a few iterations of BA over camera rays and/or bone-length constraints.
   - Should improve all methods uniformly.

### P4: Better 2D detections (if P0–P3 plateau)

**Actions:**
- Re-generate `data/webbridge/shelf_campus_detected/` using RTMPose (already used for MPI-INF-3DHP) instead of the current COCO detections.
- Re-run DLT and Iskakov baselines to quantify the raw-input ceiling.

## Suggested schedule and success criteria

| Phase | Experiment | GPU | Success criterion |
|-------|------------|-----|-------------------|
| P0 | Iskakov + residual MLP | Local 4090 | ≤ 120 mm direct MPJPE on val |
| P0 | DLT + temporal filter | CPU/4090 | ≤ 130 mm direct MPJPE on val |
| P1 | Mini-v80 smoke sweep | Local 4090 | < 200 mm without overfit |
| P2 | H36M+AIST pre-train → SC fine-tune | A800 GPU 6/7 | < 128.73 mm (beat Iskakov) |
| P3 | Temporal Iskakov | Local 4090 | improvement over frame-wise Iskakov |
| P3 | BA/kinematic post-filter | CPU/4090 | consistent gain across methods |
| P4 | RTMPose re-detection | A800 GPU 6/7 or local | lower raw DLT error |

## Immediate next steps (no jobs launched yet)

1. Create `experiments/train_iskakov_refiner_shelf_campus.py` for P0.
2. Create `experiments/dlt_temporal_filter_shelf_campus.py` for P0.
3. Define a reduced-capacity v80 config file for the P1 sweep.
4. Confirm A800 GPU 6/7 availability before scheduling the P2 cross-dataset pre-training run.

## Risks and watch-outs

- **Overfit risk remains high** until more data or stronger regularisation is introduced.
- **GPU policy:** only A800 GPUs 6/7 may be used for the heavy P2 pre-training.
- **Disk space:** A800 `/mnt/nvme0n1p1` is ~99 % full; avoid redundant checkpoint writes.
- **Detection noise ceiling:** if RTMPose does not substantially lower raw DLT error, learned gains may be limited regardless of architecture.
