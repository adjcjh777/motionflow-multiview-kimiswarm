# True-GT Leaderboard Summary — 2026-08-11 (updated 2026-08-12)

> **Status:** H36M true-GT standard protocol complete; v25 ablations and v57 re-run updated; AIST++ / Shelf/Campus / MPI-INF-3DHP smoke / partial results available.  
> **Key takeaway:** On non-circular labels, geometric/learnable-triangulation baselines (DLT, Iskakov) are the current leaders. MotionFlow variants (v25/v57/v80) still lag and overfit on the small true-GT protocols; the v57 re-run with a fixed checkpoint monitor is already beating the old lost best.

---

## 1. H36M True-GT Standard Protocol

**Protocol:** S1, S5, S6, S7, S8 → S9, S11  
**Labels:** `data/h36m_true_gt/*_multiview_m.npz` (true mocap world coords, non-circular)  
**Metric:** Combined direct MPJPE (mm) on S9+S11

| Rank | Method | Combined direct (mm) | S9 direct | S11 direct | PA-MPJPE | Notes |
|:---:|:---|---:|---:|---:|---:|:---|
| 1 | **Iskakov ICCV 2019** | **23.35** | 27.10 | 19.60 | 23.10 | best epoch 4 |
| 2 | DLT (confidence-weighted) | **25.67** | 29.54 | 21.81 | 25.55 | frozen reference (full-set)
| 3 | DLT (unweighted) | 29.19 | 33.61 | 24.77 | 29.31 | frozen reference |
| 4 | **v80 (medium)** | **39.98** | — | — | — | best epoch 4; overfit to 133.71 mm by epoch 8 |
| 5 | v80 (v3 reg) | 42.60 | — | — | — | local 2-epoch best; A800 v2 best 39.70 |
| 6 | **v25 (medium)** | **43.93** | 47.28 | 40.54 | — | **test** result; corrected-val ablations 45.80 / 46.75 mm @ epoch 1 |
| 7 | **v57 (medium)** | **75.16** (obs.) / **81.47** (ckpt) | — | — | — | best epoch 3; early-stopped @ epoch 5 (final **80.21** mm); saved ckpt is epoch 2 |
| 8 | **v57 (re-run)** | **57.81** (epoch 4) | — | — | — | in progress on A800 GPU 5; 60.72 mm @ epoch 5; checkpoint monitor fixed |

- **Current leader:** Iskakov ICCV 2019 (23.35 mm).
- **DLT baselines are very strong:** conf-weighted DLT is only 2.32 mm behind Iskakov.
- **MotionFlow variants overfit:** v80/v25 improve early but diverge quickly on the corrected H36M protocol.

---

## 2. AIST++ Cross-Dataset Sanity (Smoke)

**Protocol:** 9-view AIST++, H36M skeleton, smoke split  
**Metric:** Val MPJPE (mm)

| Method | Val MPJPE (mm) | Notes |
|:---|---:|:---|
| DLT (confidence-weighted) | **6.52** | frozen reference; very strong baseline |
| Iskakov ICCV 2019 | **9.31** | CPU smoke, best epoch 6 |
| DLT (unweighted) | **12.66** | frozen reference |
| v25 | **71.79** | 3-epoch smoke |
| v80 | **76.34** | 3-epoch smoke |

- AIST++ full val (128 clips): Iskakov = **29.27 mm** direct, **26.03 mm** root-aligned.
- Learned MotionFlow models are far behind the geometric baselines on AIST++ smoke; full medium runs still needed.

---

## 3. Shelf / Campus Detected (Non-Circular)

**Protocol:** COCO detections + true 3D annotation  
**Metric:** Val direct MPJPE (mm)

| Rank | Method | Val direct (mm) | Val PA-MPJPE | Notes |
|:---:|:---|---:|---:|:---|
| 1 | **Iskakov ICCV 2019** | **128.73** | 119.23 | early-stop epoch 11 |
| 2 | DLT (confidence-weighted) | 132.29 | 120.95 | frozen reference |
| 3 | DLT (unweighted) | 134.43 | 122.37 | frozen reference |
| 4 | v80 long (25 ep) | 276.49 | — | best epoch 7, then overfits |
| 5 | v57 long (25 ep) | 306.45 | — | best epoch 4, then overfits |
| 6 | v80 smoke | 408.58 | — | 3-epoch smoke |
| 7 | v57 smoke | 424.63 | — | 3-epoch smoke |
| 8 | v25 smoke | 430.67 | — | 3-epoch smoke |

- **Current leader:** Iskakov (128.73 mm), followed closely by DLT variants.
- **MotionFlow variants need substantial re-tuning / more data** to match geometric baselines on this small detected protocol.

---

## 4. MPI-INF-3DHP Non-Circular Smoke

**Protocol:** MPI-INF-3DHP only, true 3D GT, 3-epoch smoke  
**Metric:** Best val MPJPE (mm)

| Method | Best val MPJPE (mm) | Notes |
|:---|---:|:---|
| DLT baseline | **23.79** | geometric lower bound |
| v25 geometry fusion | **26.15** | closest learned model to DLT |
| v57 DC-PSC (128 samples, 3 ep) | **33.26** | domain-conditional physical-space calibration |
| v57 DC-PSC (512 samples, 5 ep) | **33.96** | more training did not close gap |
| v46 SVG | **34.94** | sparse-view generalization |
| v80 VRBT | **35.22** | learned view-reliability before triangulation |

- **Current leader:** DLT baseline (23.79 mm) on GT-projected 2D smoke.
- Complex modules do not automatically win on this tiny smoke; longer training / more data / sparse-view evaluation needed.
- **Blocker:** Real detected-2D `.npz` exist (16 files), but DLT baseline is ~326–400 mm due to camera/label alignment. Learned-model benchmarking on MPI is blocked until alignment is fixed.

---

## Cross-Dataset Snapshot

| Dataset | Leader | Best MPJPE (mm) | MotionFlow best (mm) | Gap to leader |
|:---|:---|---:|---:|---:|
| H36M true-GT | Iskakov | 23.35 | v80 39.98 (v25 **43.93** test) | +16.63 |
| AIST++ smoke | DLT conf-weighted | 6.52 | v25 71.79 | +65.27 |
| Shelf/Campus detected | Iskakov | 128.73 | v80 long 276.49 | +147.76 |
| MPI-INF-3DHP smoke | DLT baseline | 23.79 | v25 26.15 | +2.36 |

---

## Takeaways

1. **True-GT protocol is now reliable.** H36M numbers are in the expected 15–30 mm range, unlike the old circular-label 0.62 mm.
2. **Iskakov and DLT are the baselines to beat.** On every non-circular dataset, geometric or learnable-triangulation baselines lead.
3. **MotionFlow variants overfit on small protocols.** v80 reaches competitive early numbers but diverges; v25/v57/v80 all need regularization, SWA, or more data. The v57 re-run with a fixed `mpjpe` checkpoint monitor is already at 57.81 mm @ epoch 4, showing the old 75.16 mm "best" was partly a checkpoint artifact.
4. **Paper contribution must shift to sparse-view / cross-domain robustness**, not absolute MPJPE records.

---

## Blockers & Next Steps

| Blocker | Impact | Next Step |
|:---|:---|:---|
| MPI-INF-3DHP detected-2D alignment | RTMPose regeneration running on A800 GPU 7; old MediaPipe DLT baseline ~326–400 mm | Validate RTMPose results; re-run DLT until ~20–30 mm |
| v25/v80/v57 overfit on H36M | Cannot assess full model potential | v25 ablations done (45.80/46.75 mm @ epoch 1, diverged); v57 re-run in progress; try mixed-dataset training |
| AIST++ only smoke results | Full medium leaderboard incomplete | Run v25/v57/v80 medium on AIST++ |
| Shelf/Campus small dataset | Severe overfitting even after 25 epochs | Collect more detected data or focus on sparse-view / cross-domain metrics |

---

## Source Docs

- `docs/results_true_gt_h36m.md`
- `docs/results_true_gt_shelf_campus.md`
- `docs/results_aistpp_iskakov_full.md`
- `docs/noncircular_mpi_smoke_results.md`
- `docs/cvpr2027_status.md`

---

*Generated: 2026-08-11. No GPU work started; RTX 4090 was busy at summary time (nvidia-smi: 68% util, ~14.1 GB VRAM, active `python.exe` compute processes).*
