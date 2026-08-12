# Final Results Snapshot — 2026-08-12

> Snapshot of all currently verified numbers across the MotionFlow-MultiView project.  
> True-GT H36M, Shelf/Campus, AIST++, and MPI-INF-3DHP detected-2D protocols are all non-circular.

## Executive summary

- **H36M true-GT standard protocol** is now the primary benchmark. The Iskakov learnable-triangulation baseline is the strongest method at **23.35 mm** combined direct MPJPE, beating both DLT variants. All MotionFlow variants (v25, v57, v80) currently lose to the geometric / learnable-triangulation baselines and overfit after a small number of epochs.
- **Shelf/Campus detected** confirms the same pattern: Iskakov/DLT are strong, and MotionFlow variants are far behind.
- **AIST++** canonical multi-view detections are clean; Iskakov reaches **29.27 mm** on the full val set, improving over strong DLT baselines.
- **MPI-INF-3DHP detected-2D** is currently blocked by a camera/label alignment issue: DLT baseline is very poor (~469–2813 mm), so learned-model benchmarking is premature until the data is inspected.

---

## H36M True-GT Standard Protocol

> Protocol: S1, S5, S6, S7, S8 → S9, S11.  
> Labels: `data/h36m_true_gt/*_multiview_m.npz` (true mocap world coordinates, non-circular).  
> Manifest: `configs/splits/h36m_true_gt_standard.yaml`.

### Main leaderboard

| Method | S9 direct (mm) | S11 direct (mm) | Combined direct (mm) | Combined PA-MPJPE (mm) | Notes |
|---|---:|---:|---:|---:|---|
| DLT (unweighted) | 33.61 | 24.77 | 29.19 | 29.31 | frozen reference |
| DLT (confidence-weighted) | 29.82 | 21.91 | 25.87 | 25.55 | frozen reference |
| **Iskakov ICCV 2019** | **27.10** | **19.60** | **23.35** | **23.10** | best run, epoch 4 |
| v80 (medium) | — | — | **39.98** | — | best epoch 4; overfit afterward |
| v80 (smoke) | — | — | **98.12** | — | 2-epoch smoke only |
| v25 (medium) | **67.92** | **77.68** | **72.80** | — | best epoch 2; diverged to 207.62 mm by epoch 8 |
| v57 (medium) | — | — | **75.16** (obs.) / **81.47** (ckpt) | — | best epoch 3; early-stopped at epoch 5 (80.21 mm); saved ckpt is epoch 2 |

- Iskakov is the current leader, ~2.5 mm better than confidence-weighted DLT.
- v80 reaches 39.98 mm at best but still lags Iskakov by ~16.6 mm.
- v25 and v57 do not currently beat any geometric baseline and diverge quickly.

### Iskakov trajectory (best run)

| Epoch | Combined direct | S9 direct | S11 direct |
|---:|---:|---:|---:|
| 1 | 23.40 | 27.12 | 19.68 |
| 2 | 23.37 | 27.11 | 19.63 |
| 3 | 23.37 | 27.12 | 19.62 |
| 4 (best) | **23.35** | **27.10** | **19.60** |
| 10 | 23.37 | 27.13 | 19.62 |

Early-stopped by patience; best epoch = 4. A larger-batch confirmation run gave **23.38 mm** (epoch 7).

### v80 trajectory (medium)

| Epoch | val MPJPE (mm) |
|---:|---:|
| 1 | ~80 |
| 2 | ~40 |
| 3 | ~45 |
| 4 (best) | **39.98** |
| 8 (final) | 133.71 |

Overfits after epoch 4. Best local result is 39.98 mm; best known result is **39.70 mm** (v2, A800 checkpoint).

### v25 trajectory (medium)

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

Diverges monotonically after epoch 2.

### v57 trajectory (medium)

| Epoch | val MPJPE (mm) |
|---:|---:|
| 1 | 98.11 |
| 2 | 81.47 |
| 3 (best) | **75.16** |
| 4 | 76.60 |
| 5 (final) | 80.21 |

Early-stopped at epoch 5. Best checkpoint: epoch 3, 75.16 mm.

---

## H36M True-GT Sparse-View MPJPE@k (Iskakov)

Evaluated with `experiments/eval_iskakov_mpjpe_at_k.py` on the same Iskakov checkpoint.

| k | Learned direct (mm) | Learned root (mm) | DLT unweighted (first subset) | DLT conf-weighted (first subset) |
|---:|---:|---:|---:|---:|
| 2 | 53.61 (±27) | 55.12 | 37.19 avg | 36.42 avg |
| 3 | **27.80** (±2) | 27.54 | 34.86 avg | 33.68 avg |
| 4 | **23.39** | 23.14 | 29.15 avg | 25.94 avg |

- k=2 is out-of-distribution for the weight-prediction head and collapses.
- For k ≥ 3, learned weights beat both frozen DLT variants.

---

## Shelf / Campus Detected Leaderboard

> Data: `data/webbridge/shelf_campus_detected/` (real 2D detections + true 3D).  
> Split: `configs/splits/shelf_campus_detected_smoke.yaml`.

| Method | Val direct MPJPE | Val PA-MPJPE | Notes |
|--------|------------------|--------------|-------|
| Iskakov ICCV 2019 learnable triangulation | **128.73** | **119.23** | early-stop at epoch 11 |
| DLT (confidence-weighted) | 132.29 | 120.95 | frozen reference |
| DLT (unweighted) | 134.43 | 122.37 | frozen reference |
| v80 long run (25 epochs) | 276.49 | — | best epoch 7, then overfits |
| v57 long run (25 epochs) | 306.45 | — | best epoch 4, then overfits |
| v80 smoke | 408.58 | — | 3-epoch smoke |
| v57 smoke | 424.63 | — | 3-epoch smoke |
| v25 smoke | 430.67 | — | 3-epoch smoke |

- Per-dataset Iskakov: Shelf 123.76 mm / 120.16 mm PA; Campus 133.71 mm / 118.29 mm PA.
- Same pattern as H36M: Iskakov/DLT strong; MotionFlow variants need substantial re-tuning.

---

## AIST++

### Full val set — Iskakov

| Method | Val direct MPJPE (mm) | Val root MPJPE (mm) | Notes |
|---|---:|---:|---|
| Unweighted DLT (frozen) | 53.72 | 46.63 | Baseline triangulation |
| Confidence-weighted DLT (frozen) | 34.64 | 32.79 | Weights = detection confidences |
| **Iskakov learned weights** | **29.27** | **26.03** | Best epoch 10/10 |

Training trajectory (val direct): 45.50 → 38.39 → 34.09 → 31.66 → 30.53 → 29.88 → 29.50 → 29.37 → 29.28 → **29.27**.

### DLT baseline only

| Metric | Value |
|---|---|
| Clips | 128 |
| Total frames | 105,269 |
| MPJPE (mm) | 22.92 |
| PA-MPJPE (mm) | 29.35 |
| Reprojection error (px) | 18.05 |

Note: the standalone DLT baseline (22.92 mm) is evaluated with a different procedure than the Iskakov DLT reference (53.72 mm); the Iskakov result is the fair within-run comparison.

### H36M-style smoke (cross-dataset sanity)

| Method | val MPJPE (mm) | Notes |
|---|---:|---|
| DLT (unweighted) | **12.66** | frozen reference |
| DLT (confidence-weighted) | **6.52** | frozen reference |
| Iskakov ICCV 2019 | **9.31** | best epoch 6, CPU smoke |
| v25 | **71.79** | 3-epoch smoke |
| v80 | **76.34** | 3-epoch smoke |

---

## MPI-INF-3DHP Detected-2D

> Input: detected 2D keypoints + calibrated cameras.  
> Target: true 3D mocap (`joints_3d`).  
> Status: **blocked by camera/label alignment issue**.

### Smoke set (15 files)

| Method | Mean MPJPE (mm) | Mean PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| DLT (confidence-weighted) | **2813.19** | **454.57** | very poor |
| DLT (unweighted) | **2812.43** | **454.26** | very poor |

### Full detected-2D set (17 files)

| Method | Mean MPJPE (mm) | Mean PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| DLT (confidence-weighted) | **539.01** | **361.53** | `s_02_seq_02` outlier ~2.76 m |
| DLT (unweighted) | **469.05** | **317.95** | `s_02_seq_02` outlier ~2.76 m |

- The non-circularity check passes, so the high error is real for the current detected-2D / camera pairing.
- A spot check shows ~189 px reprojection error for true 3D vs. stored cameras, indicating a calibration-frame or coordinate-frame mismatch.
- Learned-model benchmarking on MPI is premature until `s_02_seq_02` and the camera/label alignment are inspected.

---

## Blockers and next steps

1. **MPI-INF-3DHP detected-2D alignment** (P0): DLT baseline is ~469–2813 mm, so the data cannot yet provide a meaningful lower bound for learned models. Inspect `s_02_seq_02` and camera/label frames.
2. **MotionFlow overfitting on true-GT**: v80 reaches 39.98 mm but overfits; v25/v57 diverge. Needs re-tuning (weight decay, train samples, augmentation, lr schedule) before another medium run.
3. **Sparse-view k=2**: Iskakov weight-prediction head collapses at 2 views; needs a stronger per-view feature backbone or residual refiner.
4. **Cross-dataset generalization**: MotionFlow variants are far behind on AIST++ and Shelf/Campus smoke; confirm whether the same architectural fixes improve all protocols.

---

## Sources

- `docs/results_true_gt_h36m.md`
- `docs/results_true_gt_shelf_campus.md`
- `docs/results_iskakov_h36m_true_gt.md`
- `docs/results_v80_h36m_true_gt.md`
- `docs/results_aistpp_iskakov_full.md`
- `docs/results_aistpp_dlt_baseline.md`
- `docs/results_mpi_detected_dlt.md`
