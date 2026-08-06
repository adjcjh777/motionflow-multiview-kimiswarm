# Failure-Case Analysis: Temporal Ray-Attention Baseline on MPI-INF-3DHP S2/Seq1

## Setup

* Dataset: `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`
* Checkpoint: `outputs/ray_attention_temporal_smoke.pth`
* Inference window (clip_len): 13
* Metric: MPJPE (mm) and PA-MPJPE (mm)
* Note: The requested ``outputs/ray_attention_temporal_baseline.pth`` did not exist; the smoke-run checkpoint ``outputs/ray_attention_temporal_smoke.pth`` was used instead (same 25.26 mm result).

## Overall Results

* **MPJPE**: 25.26 mm
* **PA-MPJPE**: 24.17 mm
* Frames evaluated: 6502

## Worst Joints

| Rank | Joint | Mean MPJPE (mm) |
|------|-------|------------------|
| 1 | r_eye (27) | 42.71 |
| 2 | l_thumb (22) | 42.45 |
| 3 | l_ear (26) | 42.39 |
| 4 | l_hand_tip (21) | 42.14 |
| 5 | spine (20) | 38.86 |
| 6 | l_eye (25) | 38.81 |
| 7 | l_hand (7) | 35.52 |
| 8 | r_knee (17) | 29.45 |
| 9 | l_hip (12) | 28.48 |
| 10 | l_wrist (6) | 27.65 |
| 11 | r_hip (16) | 26.55 |
| 12 | r_hand (11) | 25.53 |
| 13 | l_elbow (5) | 23.88 |
| 14 | r_foot (19) | 23.58 |
| 15 | r_elbow (9) | 23.55 |
| 16 | r_thumb (24) | 23.47 |
| 17 | l_ankle (14) | 23.26 |
| 18 | l_foot (15) | 22.55 |
| 19 | r_shoulder (8) | 22.18 |
| 20 | l_knee (13) | 22.05 |
| 21 | r_wrist (10) | 21.96 |
| 22 | thorax (1) | 21.60 |
| 23 | pelvis (0) | 16.79 |
| 24 | neck (2) | 13.18 |
| 25 | head (3) | 11.09 |
| 26 | r_hand_tip (23) | 7.00 |
| 27 | r_ankle (18) | 6.25 |
| 28 | l_shoulder (4) | 4.20 |

## Worst Frames

| Rank | Frame | MPJPE (mm) |
|------|-------|------------|
| 1 | 2606 | 32.93 |
| 2 | 2531 | 32.92 |
| 3 | 2530 | 32.90 |
| 4 | 2607 | 32.89 |
| 5 | 2532 | 32.86 |
| 6 | 2605 | 32.85 |
| 7 | 2529 | 32.82 |
| 8 | 2608 | 32.81 |
| 9 | 2533 | 32.80 |
| 10 | 2534 | 32.72 |
| 11 | 2609 | 32.71 |
| 12 | 2528 | 32.70 |
| 13 | 2535 | 32.65 |
| 14 | 2604 | 32.64 |
| 15 | 2610 | 32.62 |
| 16 | 2536 | 32.61 |
| 17 | 2537 | 32.58 |
| 18 | 2538 | 32.55 |
| 19 | 2613 | 32.54 |
| 20 | 2611 | 32.53 |

## Worst Views (by reprojection error)

Mean can be dominated by a few frames where a predicted joint lands behind a camera. Median is more robust.

| Rank | View | Mean (px) | Median (px) |
|------|------|-----------|-------------|
| 1 | 7 | 0.24 | 0.21 |
| 2 | 0 | 135654.67 | 0.18 |
| 3 | 9 | 0.19 | 0.17 |
| 4 | 2 | 0.22 | 0.17 |
| 5 | 4 | 0.21 | 0.16 |
| 6 | 3 | 0.17 | 0.15 |
| 7 | 13 | 0.17 | 0.15 |
| 8 | 5 | 0.16 | 0.13 |
| 9 | 8 | 0.18 | 0.13 |
| 10 | 12 | 0.13 | 0.10 |
| 11 | 6 | 0.11 | 0.09 |
| 12 | 1 | 0.11 | 0.09 |
| 13 | 10 | 0.09 | 0.09 |
| 14 | 11 | 0.07 | 0.05 |

## Artifacts

* Numerical arrays: `outputs\failure_analysis_temporal\failure_arrays.npz`
* Per-joint plot: `outputs\failure_analysis_temporal\per_joint_error.png`
* Per-frame plot: `outputs\failure_analysis_temporal\per_frame_error.png`
* Joint heatmap: `outputs\failure_analysis_temporal\joint_heatmap.png`
* Per-view plot: `outputs\failure_analysis_temporal\per_view_error.png`

## Observations / Failure Modes

* **Worst joints:** r_eye, l_thumb, l_ear, l_hand_tip and l_hand dominate the error budget. These are small or distal joints where 2D detectors are noisy and multi-view triangulation is most sensitive.
* **Worst frames:** The highest errors cluster around frames 2528-2613, suggesting a short, challenging activity segment (e.g., fast motion, turning, or self-occlusion).
* **Per-view reprojection:** Median reprojection errors are extremely low (<0.25 px), indicating the model preserves multi-view consistency. View 0 has a handful of outlier frames (predicted joints behind the camera) that inflate its mean.
* **PA-MPJPE vs MPJPE:** PA-MPJPE is only slightly lower than MPJPE, suggesting the remaining errors are mostly local joint offsets rather than global misalignment.
