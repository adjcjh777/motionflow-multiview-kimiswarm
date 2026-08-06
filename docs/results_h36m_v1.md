# H36M v1 Results — `ray_attention_v1`

Checkpoint: `outputs/ray_attention_v1_s_01_acts_02_03_..._16_multiview.pth`  
Training: 62,094 frames, subject 1, actions 2–16, stopped after 5 epochs because
validation had already converged.

## Clean-data robustness (500-frame subset)

All values in millimetres.

| drop | noise | ray_attention | DLT   |
|------|-------|---------------|-------|
| 0.0  | 0.00  | 2.12          | 2.12  |
| 0.0  | 2.00  | 12.82         | 12.82 |
| 0.0  | 5.00  | 30.38         | 30.39 |
| 0.2  | 0.00  | 11.20         | 11.20 |
| 0.2  | 2.00  | 22.05         | 22.05 |
| 0.2  | 5.00  | 44.06         | 44.06 |
| 0.4  | 0.00  | 19.05         | 19.05 |
| 0.4  | 2.00  | 29.89         | 29.89 |
| 0.4  | 5.00  | 59.97         | 59.97 |

On clean data v1 reproduces the DLT baseline almost exactly.

## Outlier robustness (500-frame subset, 5% outliers, 100 px scale)

| drop | noise | ray_attention | DLT   |
|------|-------|---------------|-------|
| 0.0  | 0.00  | 4.01          | 283.68 |
| 0.0  | 2.00  | 15.19         | 298.71 |
| 0.0  | 5.00  | 33.54         | 306.71 |
| 0.2  | 0.00  | 40.71         | 316.30 |
| 0.2  | 2.00  | 54.31         | 332.39 |
| 0.2  | 5.00  | 74.71         | 337.29 |
| 0.4  | 0.00  | 103.94        | 328.51 |
| 0.4  | 2.00  | 117.45        | 343.29 |
| 0.4  | 5.00  | 149.21        | 389.16 |

The v1 model is an order of magnitude more robust to sparse outliers than DLT.

## Cross-subject transfer (S1 -> S5, action 2)

Clean: ray_attention = 1.48 mm, DLT = 1.49 mm.
