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
| 19 | 10.83 | — |
| 20 | 12.30 | — |

Status: **completed**. Best val MPJPE = **9.81 mm**, clean eval MPJPE = **9.81 mm**, PA-MPJPE = **5.84 mm**. Did **not** beat the 8.75 mm anchor.

## Next-candidate CPU smoke results

All run on CPU (RTX 4090 reserved for Bayesian Tri). Two epochs only.

| Model | smoke val MPJPE (mm) | Status |
|-------|---------------------:|--------|
| `epipolar_bias_v2_pp` | 27.69 | wired, full GPU script queued |
| `camera_conditioned_pp` | 69.19 | wired, fallback candidate |
| `hierarchical_view_temporal_joint_pp` | 20.16 | wired, full GPU script queued |
| `splat_pp` | 28.82 | wired, Cholesky robustness verified |
| `WebBridgeMixedDataset` prototype | — | clean CPU smoke on H36M+MPI+AIST |

## Current run

- `epipolar_bias_v2_pp` GPU full run was stopped after the first end-to-end epoch did not complete within ~5 min (CPU-bound epipolar geometry computation); optimization needed.
- `hierarchical_attention_pp` GPU full run started on RTX 4090 (background task `bash-s2tuyaga`).
- Log: `outputs/hierarchical_attention_pp_full_mpiinf3dhp.log`
- Checkpoint target: `outputs/hierarchical_attention_pp_full_mpiinf3dhp.pth`

## Next steps

1. Monitor `hierarchical_attention_pp` full run to completion.
2. Run `scripts/eval_hierarchical_attention_pp_full_wsl.sh` for clean MPJPE.
3. If `hierarchical_attention_pp` > 8.75 mm, optimize and rerun `epipolar_bias_v2_pp`, or try `splat_pp` / `camera_conditioned_pp`.
4. Apply robustness matrix to any model that beats the anchor.
