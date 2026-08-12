# Failure-Case Analysis: Cross-View Residual + PP Model

## Setup

* Dataset: `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz`
* Checkpoint: `outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth`
* clip_len: 13

## Overall metrics

* MPJPE: **10.10 mm**
* PA-MPJPE: **7.53 mm**
* Mean residual correction: **60.50 mm**

## Worst joints

| Rank | Joint | MPJPE (mm) |
|---|---:|---:|
| 1 | l_hip | 17.18 |
| 2 | r_hand | 15.70 |
| 3 | r_knee | 13.22 |
| 4 | r_wrist | 11.99 |
| 5 | spine | 11.55 |
| 6 | r_hip | 11.39 |
| 7 | r_foot | 10.95 |
| 8 | l_hand_tip | 10.84 |
| 9 | l_thumb | 10.59 |
| 10 | l_hand | 10.51 |

## Worst frames

1. Frame 499: 301.59 mm
2. Frame 498: 261.98 mm
3. Frame 497: 215.90 mm
4. Frame 496: 165.72 mm
5. Frame 495: 110.84 mm
6. Frame 494: 52.06 mm
7. Frame 256: 12.00 mm
8. Frame 255: 11.95 mm
9. Frame 258: 11.93 mm
10. Frame 254: 11.85 mm

## Per-view reprojection error

| View | Mean (px) | Median (px) | PP delta (px) | Mean weight |
|---|---|---|---|---|
| 7 | 17.02 | 17.14 | 28.28 | 0.760 |
| 2 | 16.45 | 16.22 | 28.28 | 0.338 |
| 8 | 15.71 | 15.55 | 28.28 | 0.680 |
| 5 | 15.31 | 15.29 | 28.28 | 0.788 |
| 3 | 16.21 | 15.07 | 28.28 | 0.296 |
| 13 | 13.52 | 13.12 | 28.28 | 0.310 |
| 4 | 14.43 | 13.00 | 28.28 | 0.105 |
| 9 | 13.36 | 12.52 | 28.28 | 0.447 |
| 6 | 12.77 | 12.46 | 28.28 | 0.535 |
| 0 | 13.92 | 11.96 | 28.28 | 0.114 |
| 12 | 11.08 | 10.92 | 28.28 | 0.274 |
| 11 | 10.84 | 10.87 | 28.28 | 0.080 |
| 10 | 10.98 | 10.38 | 28.28 | 0.560 |
| 1 | 10.69 | 9.83 | 28.28 | 0.076 |

## Residual correction

* Overall mean residual correction magnitude: **60.50 mm**
| Rank | Joint | Mean residual correction (mm) |
|---|---:|---:|
| 1 | l_hand | 84.43 |
| 2 | r_knee | 79.43 |
| 3 | r_hip | 78.65 |
| 4 | l_wrist | 78.50 |
| 5 | l_elbow | 75.42 |
| 6 | l_foot | 74.91 |
| 7 | l_knee | 74.87 |
| 8 | l_ankle | 74.72 |
| 9 | r_shoulder | 74.18 |
| 10 | r_elbow | 72.67 |

Figures saved to: `outputs\failure_analysis_crossview_pp_smoke`
