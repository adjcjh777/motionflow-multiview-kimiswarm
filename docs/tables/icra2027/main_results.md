> ⚠️ **DEPRECATED.** This table reports numbers from the old circular-label protocol (H36M `joints_3d = DLT(points_2d, cameras)`), which inflated apparent accuracy. For the corrected true-GT leaderboard, see `docs/results_true_gt_h36m.md`. The table is kept only as a historical snapshot.

| Model | Params | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw DLT | — | 25.21 | 24.08 | 0.990 | 0.999 | 1.000 | 0.832 |
| Robust IRLS | — | 25.20 | 24.07 | 0.990 | 0.999 | 1.000 | 0.832 |
| **Ray-attention + PP correction (ours)** | **243 k** | **9.32** | **5.37** | **1.000** | **1.000** | **1.000** | **0.938** |