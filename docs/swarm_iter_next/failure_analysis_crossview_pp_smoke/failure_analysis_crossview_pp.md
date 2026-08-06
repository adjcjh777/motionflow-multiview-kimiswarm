# Failure-Case Analysis: Cross-View Residual + PP Model

## Setup

* Dataset: `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz`
* Checkpoint: `outputs/crossview_pp_smoke.pth`
* clip_len: 13

## Overall metrics

* MPJPE: **45.38 mm**
* PA-MPJPE: **28.76 mm**
* Mean residual correction: **50.57 mm**

## Worst joints

| Rank | Joint | MPJPE (mm) |
|---|---:|---:|
| 1 | l_hip | 74.52 |
| 2 | r_hand | 71.84 |
| 3 | r_wrist | 64.92 |
| 4 | l_hand | 60.16 |
| 5 | r_elbow | 58.22 |
| 6 | l_wrist | 56.48 |
| 7 | r_shoulder | 55.29 |
| 8 | l_elbow | 54.98 |
| 9 | thorax | 53.88 |
| 10 | l_knee | 53.17 |

## Worst frames

1. Frame 499: 300.54 mm
2. Frame 498: 261.91 mm
3. Frame 497: 216.24 mm
4. Frame 496: 167.25 mm
5. Frame 495: 113.81 mm
6. Frame 494: 58.56 mm
7. Frame 205: 49.42 mm
8. Frame 283: 49.31 mm
9. Frame 36: 49.23 mm
10. Frame 270: 49.22 mm

## Per-view reprojection error

| View | Mean (px) | Median (px) | PP delta (px) | Mean weight |
|---|---|---|---|---|
| 13 | 18.18 | 18.44 | 28.28 | 0.299 |
| 3 | 17.59 | 17.78 | 28.28 | 0.142 |
| 2 | 17.34 | 17.65 | 28.28 | 0.186 |
| 7 | 16.80 | 17.29 | 28.28 | 0.237 |
| 8 | 15.32 | 15.48 | 28.28 | 0.141 |
| 5 | 14.83 | 15.10 | 28.28 | 0.155 |
| 0 | 14.90 | 13.97 | 28.28 | 0.197 |
| 6 | 13.35 | 13.88 | 28.28 | 0.129 |
| 4 | 13.89 | 12.73 | 28.28 | 0.119 |
| 9 | 12.70 | 12.49 | 28.28 | 0.124 |
| 11 | 11.67 | 12.25 | 28.28 | 0.261 |
| 12 | 12.30 | 12.07 | 28.28 | 0.253 |
| 1 | 11.25 | 10.61 | 28.28 | 0.126 |
| 10 | 8.70 | 8.43 | 28.28 | 0.167 |

## Residual correction

* Overall mean residual correction magnitude: **50.57 mm**
| Rank | Joint | Mean residual correction (mm) |
|---|---:|---:|
| 1 | l_eye | 68.66 |
| 2 | spine | 67.91 |
| 3 | l_ear | 66.79 |
| 4 | r_eye | 66.04 |
| 5 | l_hand_tip | 64.74 |
| 6 | r_foot | 64.66 |
| 7 | l_thumb | 64.57 |
| 8 | r_thumb | 63.66 |
| 9 | r_ankle | 57.11 |
| 10 | l_shoulder | 55.60 |

Figures saved to: `outputs\failure_analysis_crossview_pp_smoke`
