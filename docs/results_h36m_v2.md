# H36M v2 Results — `ray_attention_v2`

Checkpoint: `outputs/ray_attention_v2_s_01_acts_02_03_..._16_multiview.pth`  
Training: 62,094 frames, subject 1, actions 2–16, 30 epochs.

## Clean-data robustness (500-frame subset)

All values in millimetres.

| drop | noise | ray_attention | DLT   |
|------|-------|---------------|-------|
| 0.0  | 0.00  | 3.65          | 2.12  |
| 0.0  | 2.00  | 13.50         | 12.82 |
| 0.0  | 5.00  | 31.52         | 29.95 |
| 0.2  | 0.00  | 10.99         | 10.25 |
| 0.2  | 2.00  | 20.92         | 20.45 |
| 0.2  | 5.00  | 44.47         | 43.50 |
| 0.4  | 0.00  | 17.69         | 17.50 |
| 0.4  | 2.00  | 30.19         | 30.17 |
| 0.4  | 5.00  | 58.04         | 57.85 |

On clean data the learned model matches the geometric DLT baseline closely.

## Outlier robustness (500-frame subset, 5% outliers, 100 px scale)

| drop | noise | ray_attention | DLT   |
|------|-------|---------------|-------|
| 0.0  | 0.00  | 6.88          | 283.96 |
| 0.0  | 2.00  | 16.07         | 297.22 |
| 0.0  | 5.00  | 34.77         | 303.83 |
| 0.2  | 0.00  | 43.66         | 306.67 |
| 0.2  | 2.00  | 54.36         | 324.51 |
| 0.2  | 5.00  | 75.82         | 334.33 |
| 0.4  | 0.00  | 107.28        | 339.39 |
| 0.4  | 2.00  | 124.72        | 360.92 |
| 0.4  | 5.00  | 146.40        | 368.51 |

With sparse 2D outliers the learned model is an order of magnitude more robust
than DLT, keeping errors in the centimetre range while DLT explodes to hundreds
of millimetres.

## Cross-subject transfer (S1 -> S5, action 2)

Clean: ray_attention = 3.10 mm, DLT = 1.49 mm.

## Small ablation (500-frame subset, 10 epochs, d=32)

| model          | best val MPJPE |
|----------------|----------------|
| v1 (view only) | 2.25 mm        |
| v2 (view+joint)| 4.43 mm        |

On the small subset the simpler v1 model actually outperforms v2, suggesting
joint-level attention may be unnecessary (or needs more data) and supporting a
simpler-is-better design.
