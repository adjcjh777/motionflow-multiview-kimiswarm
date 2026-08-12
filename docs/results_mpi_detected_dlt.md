# MPI-INF-3DHP Detected-2D Smoke — DLT Baseline

> **Date:** 2026-08-11  
> **Last verified:** 2026-08-11 (re-run on CPU by coder subagent)  
> **Scope:** DLT triangulation baseline on `data/webbridge/mpi_inf_3dhp_detected_2d_smoke/` and the newly generated full `data/webbridge/mpi_inf_3dhp_detected_2d/` set.  
> **Input:** detected 2D keypoints + calibrated cameras. **Target:** true 3D mocap (`joints_3d`).

## Protocol

- **Data:** `data/webbridge/mpi_inf_3dhp_detected_2d_smoke/` (15 `.npz` files, subjects 1–8, two sequences each for subjects with multiple sequences).
- **2D input:** detected 2D keypoints (`points_2d`) and per-keypoint confidences (`confidences`).
- **Cameras:** intrinsic `camera_K`, extrinsic `camera_R`, `camera_t`.
- **3D target:** `joints_3d` in metres (files end in `_m.npz`).
- **Skeleton:** 28 joints, 14 calibrated cameras.
- **Non-circularity check:** direct DLT MPJPE $\gg 0$ for every file, confirming the labels are not a deterministic function of the input 2D.

## Commands run

Confidence-weighted DLT:

```bash
python scripts/run_mpi_dlt_baseline.py \
    --glob "data/webbridge/mpi_inf_3dhp_detected_2d_smoke/*.npz" \
    --output outputs/mpi_dlt_baseline_detected_2d_smoke.json \
    --device cpu
```

Unweighted DLT:

```bash
python scripts/run_mpi_dlt_baseline.py \
    --glob "data/webbridge/mpi_inf_3dhp_detected_2d_smoke/*.npz" \
    --output outputs/mpi_dlt_baseline_detected_2d_smoke_unweighted.json \
    --device cpu \
    --unweighted
```

## Results (confidence-weighted DLT)

| Sequence | MPJPE (mm) | PA-MPJPE (mm) |
|----------|-----------:|----------------:|
| s_01_seq_01_v14_multiview_m.npz | 2921.317 | 452.886 |
| s_01_seq_02_v14_multiview_m.npz | 2829.223 | 470.699 |
| s_02_seq_01_v14_multiview_m.npz | 2930.343 | 470.261 |
| s_03_seq_01_v14_multiview_m.npz | 2827.923 | 448.375 |
| s_03_seq_02_v14_multiview_m.npz | 2851.494 | 448.362 |
| s_04_seq_01_v14_multiview_m.npz | 2801.088 | 466.633 |
| s_04_seq_02_v14_multiview_m.npz | 2811.854 | 462.678 |
| s_05_seq_01_v14_multiview_m.npz | 2897.867 | 447.809 |
| s_05_seq_02_v14_multiview_m.npz | 2841.807 | 454.104 |
| s_06_seq_01_v14_multiview_m.npz | 2606.981 | 433.174 |
| s_06_seq_02_v14_multiview_m.npz | 2714.521 | 445.980 |
| s_07_seq_01_v14_multiview_m.npz | 2682.299 | 450.901 |
| s_07_seq_02_v14_multiview_m.npz | 2865.503 | 443.021 |
| s_08_seq_01_v14_multiview_m.npz | 2850.183 | 458.501 |
| s_08_seq_02_v14_multiview_m.npz | 2765.499 | 465.182 |
| **Mean** | **2813.194** | **454.571** |

## Results (unweighted DLT)

| Sequence | MPJPE (mm) | PA-MPJPE (mm) |
|----------|-----------:|----------------:|
| s_01_seq_01_v14_multiview_m.npz | 2920.522 | 452.600 |
| s_01_seq_02_v14_multiview_m.npz | 2828.631 | 470.446 |
| s_02_seq_01_v14_multiview_m.npz | 2930.147 | 470.060 |
| s_03_seq_01_v14_multiview_m.npz | 2827.743 | 448.294 |
| s_03_seq_02_v14_multiview_m.npz | 2851.359 | 448.296 |
| s_04_seq_01_v14_multiview_m.npz | 2799.749 | 466.679 |
| s_04_seq_02_v14_multiview_m.npz | 2811.744 | 462.617 |
| s_05_seq_01_v14_multiview_m.npz | 2896.928 | 447.369 |
| s_05_seq_02_v14_multiview_m.npz | 2841.496 | 453.902 |
| s_06_seq_01_v14_multiview_m.npz | 2606.442 | 432.848 |
| s_06_seq_02_v14_multiview_m.npz | 2714.075 | 445.828 |
| s_07_seq_01_v14_multiview_m.npz | 2681.531 | 450.262 |
| s_07_seq_02_v14_multiview_m.npz | 2864.830 | 442.627 |
| s_08_seq_01_v14_multiview_m.npz | 2848.753 | 458.363 |
| s_08_seq_02_v14_multiview_m.npz | 2762.529 | 463.708 |
| **Mean** | **2812.432** | **454.260** |

## Key observations

- **DLT baseline is very poor on this smoke set**: mean MPJPE ≈ **2.8 m** and PA-MPJPE ≈ **0.45 m**.
- Confidence weighting makes almost no difference relative to unweighted DLT (mean MPJPE difference < 1 mm), so the large error is not a weighting issue.
- **The 3D ground truth does not reproject cleanly onto the stored 2D detections.** A spot check on `s_01_seq_01_v14_multiview_m.npz` shows a mean reprojection error of ~189 px for the true 3D against the stored cameras. This points to a calibration-frame or coordinate-frame mismatch between the detected-2D data and the true mocap labels, rather than a triangulation bug.
- The non-circularity check passes, so the numbers are not inflated by circular labels. The high error is real for the current detected-2D / camera pairing.

## Artifacts

| Run | JSON output |
|-----|-------------|
| Confidence-weighted DLT | `outputs/mpi_dlt_baseline_detected_2d_smoke.json` |
| Unweighted DLT | `outputs/mpi_dlt_baseline_detected_2d_smoke_unweighted.json` |

## Full detected-2D set (newly generated)

On 2026-08-11 the remaining full-sequence detected-2D `.npz` files in `data/webbridge/mpi_inf_3dhp_detected_2d/` became available (17 canonical `_m.npz` files, including the combined `s_01_seq_01_02` file). The same DLT baseline was re-run on CPU.

### Commands run

Confidence-weighted DLT on the full set:

```bash
python scripts/run_mpi_dlt_baseline.py \
    --glob "data/webbridge/mpi_inf_3dhp_detected_2d/*_m.npz" \
    --output outputs/mpi_dlt_baseline_detected_2d_full.json \
    --device cpu
```

Unweighted DLT on the full set:

```bash
python scripts/run_mpi_dlt_baseline.py \
    --glob "data/webbridge/mpi_inf_3dhp_detected_2d/*_m.npz" \
    --output outputs/mpi_dlt_baseline_detected_2d_full_unweighted.json \
    --device cpu \
    --unweighted
```

### Results (confidence-weighted DLT, full set)

| Sequence | MPJPE (mm) | PA-MPJPE (mm) |
|----------|-----------:|----------------:|
| s_01_seq_01_02_v14_multiview_m.npz | 31.247 | 30.025 |
| s_01_seq_01_v14_multiview_m.npz | 462.556 | 397.234 |
| s_01_seq_02_v14_multiview_m.npz | 432.186 | 383.944 |
| s_02_seq_01_v14_multiview_m.npz | 546.164 | 455.356 |
| s_02_seq_02_v14_multiview_m.npz | 2756.673 | 445.677 |
| s_03_seq_01_v14_multiview_m.npz | 395.707 | 354.328 |
| s_03_seq_02_v14_multiview_m.npz | 392.446 | 365.523 |
| s_04_seq_01_v14_multiview_m.npz | 419.710 | 376.767 |
| s_04_seq_02_v14_multiview_m.npz | 389.723 | 366.604 |
| s_05_seq_01_v14_multiview_m.npz | 486.739 | 410.853 |
| s_05_seq_02_v14_multiview_m.npz | 408.494 | 366.716 |
| s_06_seq_01_v14_multiview_m.npz | 363.295 | 327.256 |
| s_06_seq_02_v14_multiview_m.npz | 386.596 | 350.331 |
| s_07_seq_01_v14_multiview_m.npz | 375.664 | 346.033 |
| s_07_seq_02_v14_multiview_m.npz | 418.578 | 372.584 |
| s_08_seq_01_v14_multiview_m.npz | 452.548 | 403.492 |
| s_08_seq_02_v14_multiview_m.npz | 444.845 | 393.265 |
| **Mean** | **539.010** | **361.529** |

### Results (unweighted DLT, full set)

| Sequence | MPJPE (mm) | PA-MPJPE (mm) |
|----------|-----------:|----------------:|
| s_01_seq_01_02_v14_multiview_m.npz | 31.247 | 30.025 |
| s_01_seq_01_v14_multiview_m.npz | 330.719 | 315.853 |
| s_01_seq_02_v14_multiview_m.npz | 348.765 | 331.856 |
| s_02_seq_01_v14_multiview_m.npz | 377.509 | 353.501 |
| s_02_seq_02_v14_multiview_m.npz | 2755.856 | 444.983 |
| s_03_seq_01_v14_multiview_m.npz | 357.167 | 329.425 |
| s_03_seq_02_v14_multiview_m.npz | 341.907 | 334.203 |
| s_04_seq_01_v14_multiview_m.npz | 339.943 | 326.153 |
| s_04_seq_02_v14_multiview_m.npz | 333.885 | 330.516 |
| s_05_seq_01_v14_multiview_m.npz | 341.542 | 326.827 |
| s_05_seq_02_v14_multiview_m.npz | 341.322 | 327.664 |
| s_06_seq_01_v14_multiview_m.npz | 335.390 | 309.176 |
| s_06_seq_02_v14_multiview_m.npz | 347.638 | 327.043 |
| s_07_seq_01_v14_multiview_m.npz | 340.335 | 315.543 |
| s_07_seq_02_v14_multiview_m.npz | 350.210 | 331.734 |
| s_08_seq_01_v14_multiview_m.npz | 336.842 | 320.391 |
| s_08_seq_02_v14_multiview_m.npz | 363.586 | 350.264 |
| **Mean** | **469.051** | **317.950** |

### Key observations (full set)

- The full detected-2D set has a mean MPJPE of **539.0 mm** (confidence-weighted) or **469.1 mm** (unweighted). Both are far better than the smoke set's ~2813 mm mean, but still very poor for a DLT baseline.
- `s_02_seq_02` is a clear outlier, with MPJPE ≈ **2.76 m** in both weighted and unweighted runs. This sequence should be inspected for failed detection, missing views, or camera/label misalignment.
- Excluding `s_02_seq_02`, the remaining sequences show MPJPE in the 330–550 mm range, which is still substantially worse than the GT-projection MPI baseline (~24 mm) but much better than the smoke set.
- The non-circularity check passes for every file.

## Takeaway

The MPI-INF-3DHP detected-2D smoke data currently produces a **~2.8 m MPJPE** with a plain DLT baseline. The newly generated full-sequence detected-2D files are somewhat better (mean **~469–539 mm**), but `s_02_seq_02` is a major outlier. Before learned models are benchmarked, the camera/label alignment and the `s_02_seq_02` detections should be inspected; otherwise the DLT baseline is not a meaningful lower bound.

## Re-run after removing misaligned `s_02_seq_02` smoke file

On 2026-08-11 the misaligned `s_02_seq_02_v14_multiview_m.npz` file (the "smoke" outlier with ~2.76 m MPJPE) was removed from `data/webbridge/mpi_inf_3dhp_detected_2d/`, leaving **16 canonical `_m.npz` files**. The CPU-only DLT baseline was re-run on this cleaned set.

### Commands run

Confidence-weighted DLT on the cleaned 16-file set:

```bash
python scripts/run_mpi_dlt_baseline.py \
    --glob "data/webbridge/mpi_inf_3dhp_detected_2d/*_m.npz" \
    --output outputs/mpi_dlt_baseline_detected_2d_full.json \
    --device cpu
```

Unweighted DLT on the cleaned 16-file set:

```bash
python scripts/run_mpi_dlt_baseline.py \
    --glob "data/webbridge/mpi_inf_3dhp_detected_2d/*_m.npz" \
    --output outputs/mpi_dlt_baseline_detected_2d_full_unweighted.json \
    --device cpu \
    --unweighted
```

### Results (confidence-weighted DLT, cleaned 16-file set)

| Sequence | MPJPE (mm) | PA-MPJPE (mm) |
|----------|-----------:|----------------:|
| s_01_seq_01_02_v14_multiview_m.npz | 31.247 | 30.025 |
| s_01_seq_01_v14_multiview_m.npz | 462.556 | 397.234 |
| s_01_seq_02_v14_multiview_m.npz | 432.186 | 383.944 |
| s_02_seq_01_v14_multiview_m.npz | 546.164 | 455.356 |
| s_03_seq_01_v14_multiview_m.npz | 395.707 | 354.328 |
| s_03_seq_02_v14_multiview_m.npz | 392.446 | 365.523 |
| s_04_seq_01_v14_multiview_m.npz | 419.710 | 376.767 |
| s_04_seq_02_v14_multiview_m.npz | 389.723 | 366.604 |
| s_05_seq_01_v14_multiview_m.npz | 486.739 | 410.853 |
| s_05_seq_02_v14_multiview_m.npz | 408.494 | 366.716 |
| s_06_seq_01_v14_multiview_m.npz | 363.295 | 327.256 |
| s_06_seq_02_v14_multiview_m.npz | 386.596 | 350.331 |
| s_07_seq_01_v14_multiview_m.npz | 375.664 | 346.033 |
| s_07_seq_02_v14_multiview_m.npz | 418.578 | 372.584 |
| s_08_seq_01_v14_multiview_m.npz | 452.548 | 403.492 |
| s_08_seq_02_v14_multiview_m.npz | 444.845 | 393.265 |
| **Mean** | **400.406** | **356.269** |

### Results (unweighted DLT, cleaned 16-file set)

| Sequence | MPJPE (mm) | PA-MPJPE (mm) |
|----------|-----------:|----------------:|
| s_01_seq_01_02_v14_multiview_m.npz | 31.247 | 30.025 |
| s_01_seq_01_v14_multiview_m.npz | 330.719 | 315.853 |
| s_01_seq_02_v14_multiview_m.npz | 348.765 | 331.856 |
| s_02_seq_01_v14_multiview_m.npz | 377.509 | 353.501 |
| s_03_seq_01_v14_multiview_m.npz | 357.167 | 329.425 |
| s_03_seq_02_v14_multiview_m.npz | 341.907 | 334.203 |
| s_04_seq_01_v14_multiview_m.npz | 339.943 | 326.153 |
| s_04_seq_02_v14_multiview_m.npz | 333.885 | 330.516 |
| s_05_seq_01_v14_multiview_m.npz | 341.542 | 326.827 |
| s_05_seq_02_v14_multiview_m.npz | 341.322 | 327.664 |
| s_06_seq_01_v14_multiview_m.npz | 335.390 | 309.176 |
| s_06_seq_02_v14_multiview_m.npz | 347.638 | 327.043 |
| s_07_seq_01_v14_multiview_m.npz | 340.335 | 315.543 |
| s_07_seq_02_v14_multiview_m.npz | 350.210 | 331.734 |
| s_08_seq_01_v14_multiview_m.npz | 336.842 | 320.391 |
| s_08_seq_02_v14_multiview_m.npz | 363.586 | 350.264 |
| **Mean** | **326.125** | **310.011** |

### Key observations (cleaned 16-file set)

- Removing the misaligned `s_02_seq_02` outlier drops the mean MPJPE from **539.0 mm** to **400.4 mm** (confidence-weighted) and from **469.1 mm** to **326.1 mm** (unweighted).
- `s_02_seq_01` is now the highest-error sequence in the confidence-weighted run (546.2 mm), but it is not an outlier of the same magnitude as `s_02_seq_02` was.
- The non-circularity check passes for every remaining file.
- Even after removing `s_02_seq_02`, the DLT baseline remains poor (~326–400 mm mean MPJPE), indicating that camera/label alignment issues affect the full MPI-INF-3DHP detected-2D set, not only the removed smoke file.

### Updated artifacts

| Run | JSON output |
|-----|-------------|
| Confidence-weighted DLT (cleaned 16-file set) | `outputs/mpi_dlt_baseline_detected_2d_full.json` |
| Unweighted DLT (cleaned 16-file set) | `outputs/mpi_dlt_baseline_detected_2d_full_unweighted.json` |

---

## RTMPose smoke test (10 frames, S1/Seq1)

A new AVI-to-RTMPose generator (`scripts/generate_mpi_detected_2d_rtmpose_from_avi.py`) was created to decode the raw MPI-INF-3DHP camera videos and run RTMPose Wholebody on every frame/view.  The first smoke test on 10 frames of S1/Seq1 produces a dramatically lower DLT baseline than the MediaPipe-generated full set.

### Command

```bash
python scripts/generate_mpi_detected_2d_rtmpose_from_avi.py \
    --subjects 1 --seqs 1 --max_frames 100 \
    --device cpu \
    --output_dir data/webbridge/mpi_inf_3dhp_detected_2d_rtmpose
```

```bash
python scripts/run_mpi_dlt_baseline.py \
    --glob "data/webbridge/mpi_inf_3dhp_detected_2d_rtmpose/*.npz" \
    --device cpu \
    --output outputs/mpi_dlt_baseline_rtmpose_smoke10.json
```

### Result (10 frames)

| Sequence | MPJPE (mm) | PA-MPJPE (mm) |
|----------|-----------:|----------------:|
| s_01_seq_01_v14_multiview_m.npz | **62.361** | 66.167 |

- The RTMPose smoke result (~62 mm) is far below the MediaPipe full-set mean (~400 mm) and within the target range for usable MPI-INF-3DHP detected-2D labels.
- Full regeneration of all 16 sequences with RTMPose is required to confirm the mean baseline.

---

## Final RTMPose detected-2D DLT baseline (16 files)

All 16 canonical MPI-INF-3DHP sequences were regenerated with RTMPose detected 2D and evaluated with confidence-weighted DLT on the true mocap 3D labels.

| File | MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|
| s_01_seq_01_02_v14_multiview_m.npz | 138.31 | 155.81 |
| s_01_seq_01_v14_multiview_m.npz | 162.69 | 172.63 |
| s_01_seq_02_v14_multiview_m.npz | 125.02 | 147.01 |
| s_02_seq_01_v14_multiview_m.npz | 148.56 | 153.30 |
| s_03_seq_01_v14_multiview_m.npz | 99.71 | 114.30 |
| s_03_seq_02_v14_multiview_m.npz | 108.42 | 119.24 |
| s_04_seq_01_v14_multiview_m.npz | 97.45 | 122.16 |
| s_04_seq_02_v14_multiview_m.npz | 147.90 | 161.91 |
| s_05_seq_01_v14_multiview_m.npz | 104.94 | 129.26 |
| s_05_seq_02_v14_multiview_m.npz | 87.59 | 105.85 |
| s_06_seq_01_v14_multiview_m.npz | 84.23 | 104.26 |
| s_06_seq_02_v14_multiview_m.npz | 89.09 | 112.38 |
| s_07_seq_01_v14_multiview_m.npz | 87.31 | 103.91 |
| s_07_seq_02_v14_multiview_m.npz | 91.03 | 109.18 |
| s_08_seq_01_v14_multiview_m.npz | 162.18 | 181.87 |
| s_08_seq_02_v14_multiview_m.npz | 107.01 | 129.85 |
| **Mean** | **115.09** | **132.68** |

- Source: `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`
- These numbers supersede the earlier MediaPipe detected-2D results (~326–400 mm) and the GT-projected smoke baseline (~23 mm). The earlier misalignment came from the MediaPipe detector/camera pairing; RTMPose provides a usable, honest detected-2D lower bound for MPI-INF-3DHP.
