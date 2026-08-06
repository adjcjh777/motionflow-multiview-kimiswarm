# Iter15 Results Tracker

This document tracks smoke/full results for the iter15 20-agent swarm proposals.

## Current anchor

- Model: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
- Clean MPJPE: **8.75 mm**
- Clean PA-MPJPE: **4.95 mm**
- Checkpoint: `outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth`

## Tier-1 smoke results

| # | Direction | Best val MPJPE (5-epoch smoke) | Checkpoint | Notes |
|---|-----------|-------------------------------|------------|-------|
| 1 | Gaussian-Splatting Pose Regularizer | 25.46 mm | `outputs/splat_pp_smoke.pth` | Converged; loss finite; auxiliary covariance head works. |
| 2 | Kinematic-Chain Graph Refiner | 26.10 mm | `outputs/kinematic_chain_pp_smoke.pth` | Fixed forward tuple ordering; smoke passed. |
| 3 | Cross-View Contrastive Pose Representation | 27.74 mm | `outputs/crossview_contrast_pp_smoke.pth` | Contrastive loss added; training stable. |

## Integration plan

1. Run full training on the most promising Tier-1 direction (Gaussian-Splatting or Kinematic-Chain) once GPU is free.
2. Compare clean/robustness metrics against the 8.75 mm anchor.
3. If a Tier-1 module improves or matches the anchor, integrate it into the factorized/visibility-v2 variants.
4. Continue with Tier-2 directions (uncertainty-aware triangulation, multi-scale spatial pyramid, etc.).
