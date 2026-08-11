# MPI-INF-3DHP Detected-2D Smoke — DLT Baseline

> **Date:** 2026-08-11  
> **Scope:** DLT triangulation baseline on `data/webbridge/mpi_inf_3dhp_detected_2d_smoke/`.  
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

## Takeaway

The MPI-INF-3DHP detected-2D smoke data currently produces a **~2.8 m MPJPE** with a plain DLT baseline. Before learned models are benchmarked, the camera/label alignment should be inspected; otherwise the DLT baseline is not a meaningful lower bound.
