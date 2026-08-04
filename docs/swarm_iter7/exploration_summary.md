# Exploration Summary: Swarm Iteration 7

**Date:** 2026-08-05

## Current best model

- **Model:** `RayAttentionFusionModelTemporalResidual` (temporal-only ray-attention + residual refinement)
- **Checkpoint:** `outputs/ray_attention_temporal_residual_final5.pth`
- **MPI-INF-3DHP (S1→S2/Seq1):** MPJPE **11.17 mm**, PA-MPJPE **8.24 mm**, AUC **0.9256**
- **Params:** 243,428
- **Human3.6M (S1→S5, h=128):** MPJPE **5.74 mm**, PA-MPJPE **3.99 mm**, AUC **0.9618**

## Explored directions

| Direction | Best result | Status |
|---|---|---|
| Temporal residual (final5) | 11.17 mm | **Best** |
| Lightweight residual (d=32, h=64) | 13.22 mm / 66k params | Strong lightweight baseline |
| Uncertainty-aware residual | 12.89 mm (epoch 2) | No improvement over final5 |
| Cross-view residual (d=64, h=128, n_st=2) | 15.29 mm | Worse than temporal-only; may need larger capacity |
| Longer temporal window (clip_len=27, smoke) | 27.15 mm (1 epoch) | Undertrained; not promising in smoke |

## Key takeaways

1. The residual refinement head on top of weighted DLT is the strongest component.
2. Explicit uncertainty or cross-view attention in the current capacity does not beat the temporal-only model.
3. The temporal-only model is also the fastest; cross-view attention over T·V tokens is much slower.
4. Robustness to occlusion is excellent; Gaussian noise is the main failure mode.

## Remaining high-potential directions

- Scale cross-view attention (d=128, n_st_layers=3) — higher compute, uncertain gain.
- Real-world GVHMR multi-view demo — requires multi-view video data.
- Iterative residual refinement with learned damping — planned, but high implementation effort.
- Longer temporal window with more epochs and smaller learning rate — smoke was unpromising.

## Publication artifacts

- Paper draft: `docs/paper_draft_icra_cvpr_2027.md`
- Architecture figure: `docs/figures/architecture.png`
- Verified results: `docs/swarm_iter7/verified_results.json`
- Robustness: `outputs/robustness_residual_final5/`
- Visualizations: `outputs/visualize_residual_mpi_final5/`

## Blockers

- GitHub issue/PR automation needs a token for `gh auth login`.
