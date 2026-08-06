# Iter16 Results – MotionFlow-MultiView Candidate Exploration

This document tracks the iter16 exploration while the Bayesian Triangulation full run was in progress on the RTX 4090.

## Anchor

- Model: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
- Checkpoint: `outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth`
- MPI-INF-3DHP S2/Seq1 clean MPJPE: **8.75 mm**
- PA-MPJPE: **4.95 mm**

## Bayesian Triangulation (Tier-2, uncertainty-aware triangulation)

Full run on RTX 4090.

| Epoch | val_MPJPE (mm) | Notes |
|------:|---------------:|-------|
| 1  | 34.98 | — |
| 2  | 16.05 | saved |
| 3  | 13.09 | saved |
| 4  | 11.54 | saved |
| 5  | 11.16 | saved |
| 6  | 10.89 | saved |
| 7  | 11.50 | — |
| 8  | 10.62 | saved |
| 9  | 11.89 | — |
| 10 | 11.00 | — |
| 11 | 11.51 | — |
| 12 | 11.25 | — |
| 13 |  9.81 | best so far |
| 14 | 12.14 | — |
| 15 | 16.56 | — |
| 16 | 11.70 | — |
| 17 | 11.38 | — |
| 18 | 18.50 | diverging |

Status: **still running** (Epoch 18/20). Early best = 9.81 mm, which is **worse than the 8.75 mm anchor**. Final clean evaluation pending.

## Next-candidate CPU smoke results

All run on CPU (RTX 4090 reserved for Bayesian Tri). Two epochs only.

| Model | smoke val MPJPE (mm) | Status |
|-------|---------------------:|--------|
| `epipolar_bias_v2_pp` | 27.69 | wired, full GPU script queued |
| `camera_conditioned_pp` | 69.19 | wired, fallback candidate |
| `hierarchical_view_temporal_joint_pp` | 20.16 | wired, full GPU script queued |
| `splat_pp` | 28.82 | wired, Cholesky robustness verified |
| `WebBridgeMixedDataset` prototype | — | clean CPU smoke on H36M+MPI+AIST |

## Next steps

1. Wait for Bayesian Tri 20-epoch run to finish.
2. Run `scripts/eval_bayesian_tri_pp_full_wsl.sh` for clean MPJPE.
3. If final clean MPJPE > 8.75 mm, start `epipolar_bias_v2_pp` GPU full run (`scripts/run_epipolar_bias_v2_pp_full_wsl.sh`).
4. If `epipolar_bias_v2_pp` also fails, run `hierarchical_attention_pp` GPU full run.
5. Apply robustness matrix to any model that beats the anchor.
