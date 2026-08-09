# v51 Domain-Agnostic Ensemble — Implementation Notes

**Tracking issue:** #178 (closed)  
**Module:** `motionflow_mv/fusion/domain_agnostic_ensemble_v51.py`

## What it does

v51 DAE adds a learned pose-expert ensemble after the residual refinement block in `OmniMultiViewFusionV5`.  It takes multiple candidate 3-D poses produced by upstream branches and blends them per joint with weights derived from geometric evidence:

- reprojection residual magnitude
- temporal jump magnitude
- epipolar / cross-view projection consistency
- available view count
- learned per-joint bias

The first expert (geometry-only pose) is biased to dominate at init, so enabling the flag preserves the existing full-view baseline.

## Current experts (smoke version)

1. `pred_3d_gn` — pose after adaptive Gauss-Newton refinement (geometry expert).
2. `pred_3d` — pose after the residual MLP (residual/refinement expert).

Future extensions can add v47 temporal, v48 domain-conditioned, and v50 SEFH-corrected poses as separate experts.

## Key design decisions

- **Identity-at-init.** The geometry expert logit is initialized to 2.0, giving it ~88% weight for a 2-expert ensemble at startup. A small uniform bypass (10%) prevents hard switching early in training.
- **Temporal-aware evidence.** The model passes `(B, T, V, ...)` camera tensors to DAE so the temporal-residual evidence is meaningful. Earlier versions flattened batch and time, which zeroed the temporal signal.
- **Bounded per-expert weights.** Weights are clamped to `[v51_dae_min_weight, ...]` after softmax to avoid dropping experts entirely.
- **Diversity auxiliary loss.** A tiny negative-variance loss (weight 0.005 by default) encourages the experts to disagree, but its magnitude is small enough that the supervised MPJPE loss dominates.

## Integration

- `motionflow_mv/fusion/omniview_fusion_v5.py`: instantiate `DomainAgnosticEnsembleV51`, call it after the residual MLP, add the diversity loss to `epi_loss`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`: add CLI flags and pass them through `model_kwargs`.
- `configs/benchmark_v51_domain_agnostic_ensemble_smoke.yaml`: smoke config with v46 + v51 enabled.
- `scripts/run_v51_domain_agnostic_ensemble_smoke_local_4090.sh`: local RTX 4090 smoke.
- `scripts/run_v51_domain_agnostic_ensemble_full_local_4090.sh`: full 4090 run (5 epochs, 5k samples).
- `scripts/launch_v33_a800_queue.py`: two A800 entries (v51 on v46, v51 on v50).
- `tests/test_v51_domain_agnostic_ensemble.py`: forward + gradient tests.

## Smoke acceptance

- `val_MPJPE@full` within 1 mm of the v46 baseline (32.97 mm).
- `MPJPE@2` improves by ≥2 mm over the v46 baseline.
- No NaN/OOM with `v51_dae_loss_weight=0.005`.

## Risks / mitigation

| Risk | Mitigation |
|------|------------|
| Gate collapses to one expert | identity-at-init strongly favors geometry; uniform bypass; min-weight clamp |
| Diversity loss destabilizes training | very low weight (0.005); gradient clipping |
| Temporal residual noisy at small `clip_len` | kept as evidence but not used as a hard constraint |
| Adding more experts increases compute | expert poses are already computed; DAE itself is a tiny 2-layer MLP |

## Expected MPJPE impact

- `MPJPE@2`: −3 to −5 mm vs v46 baseline
- `MPJPE@3`: −2 to −4 mm
- `val_MPJPE@full`: within ±0.5 mm of v46 baseline
