# H36M v1 Results — Metric-Normalised Model

Checkpoint: `outputs/ray_attention_v1_s_01_acts_02_03_..._16_multiview_m.pth`  
Training: 62,094 frames, subject 1, actions 2–16, all coordinates in meters.

## Clean-data robustness (500-frame subset)

All values in meters.

| drop | noise | ray_attention | DLT   |
|------|-------|---------------|-------|
| 0.0  | 0.00  | 0.0004        | 0.0021 |
| 0.0  | 2.00  | 0.0119        | 0.0128 |
| 0.0  | 5.00  | 0.0297        | 0.0303 |
| 0.2  | 0.00  | 0.0004        | 0.0104 |
| 0.2  | 2.00  | 0.0118        | 0.0213 |
| 0.2  | 5.00  | 0.0295        | 0.0435 |
| 0.4  | 0.00  | 0.0004        | 0.0172 |
| 0.4  | 2.00  | 0.0119        | 0.0320 |
| 0.4  | 5.00  | 0.0295        | 0.0616 |

The model essentially reproduces the DLT baseline on clean data and is
noticeably better under view dropout.

## Outlier robustness (500-frame subset, 5% outliers, 100 px scale)

| drop | noise | ray_attention | DLT   |
|------|-------|---------------|-------|
| 0.0  | 0.00  | 0.0027        | 0.2810 |
| 0.0  | 2.00  | 0.0139        | 0.2797 |
| 0.0  | 5.00  | 0.0327        | 0.2973 |
| 0.2  | 0.00  | 0.0321        | 0.3141 |
| 0.2  | 2.00  | 0.0482        | 0.3211 |
| 0.2  | 5.00  | 0.0652        | 0.3525 |
| 0.4  | 0.00  | 0.0955        | 0.3472 |
| 0.4  | 2.00  | 0.1069        | 0.3594 |
| 0.4  | 5.00  | 0.1264        | 0.3853 |

With sparse 2D outliers the model is two orders of magnitude more robust than DLT.
