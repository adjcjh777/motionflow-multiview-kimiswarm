# Robustness Matrix: PP Robust Re-Train Checkpoint

Checkpoint: `outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth`
Dataset: `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`

## Results (partial, eval timed out before occlusion/view-dropout conditions)

| Condition              | MPJPE  | PA-MPJPE | PCK@50 | PCK@100 | PCK@150 | AUC   |
|------------------------|--------|----------|--------|---------|---------|-------|
| clean                  | 8.76   | 5.12     | 1.000  | 1.000   | 1.000   | 0.942 |
| rot_0.5_deg            | 23.55  | 9.80     | 0.938  | 0.999   | 1.000   | 0.843 |
| rot_1.0_deg            | 43.32  | 17.05    | 0.680  | 0.946   | 0.994   | 0.712 |
| trans_5mm              | 9.91   | 5.36     | 1.000  | 1.000   | 1.000   | 0.934 |
| trans_10mm             | 12.59  | 5.95     | 1.000  | 1.000   | 1.000   | 0.916 |
| focal_1pct             | 9.65   | 5.60     | 1.000  | 1.000   | 1.000   | 0.936 |
| focal_2pct             | 11.13  | 6.33     | 1.000  | 1.000   | 1.000   | 0.926 |
| cxcy_3px               | 1824.68| 432.68   | 0.000  | 0.000   | 0.001   | 0.000 |
| cxcy_5px               | 2072.62| 400.08   | 0.000  | 0.000   | 0.000   | 0.000 |
| distortion_k1_0.01     | 9.88   | 6.66     | 0.993  | 0.998   | 0.999   | 0.934 |

## Observations

- Clean accuracy is **8.76 mm MPJPE** / **5.12 mm PA-MPJPE**, confirming the 8.75 mm training best.
- Translation and focal-length perturbations are handled well.
- Rotation remains the main geometric failure mode; beyond 1° it degrades quickly.
- Principal-point (cxcy) perturbations remain catastrophic, even with direct PP supervision. This is the same failure mode that motivated the robust re-train and is the target of ongoing camera-robustness work.
- Radial distortion (k1=0.01) is handled well.
