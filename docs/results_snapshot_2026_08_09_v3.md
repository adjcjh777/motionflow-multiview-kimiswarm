# Results Snapshot 2026-08-09 v3

Snapshot time: 2026-08-09 (active).

## v51 CDSVR local RTX 4090 smoke results

| Run | Epochs | Best val_MPJPE | Notes |
|-----|--------|----------------|-------|
| v51 CDSVR tiny smoke (losses disabled) | 2 | **104.09 mm** | v50/v51 SEFH/CDSVR loss weights 0.0; stable baseline |
| v51 CDSVR medium smoke 200 samples (losses disabled) | 4 | **52.33 mm** | v50/v51 heads wired in, losses disabled; early stopped after no val improvement for 3 epochs |
| v51 CDSVR tiny smoke (v50 loss 0.01) | 2 | nan | Main model emits NaN `pred_3d` and cameras when v50/v51 losses are enabled; under investigation (issue #181) |

## Key findings

- `SelfEvolutionFeedbackHeadV50` and `CrossDomainSparseViewReliabilityV51` heads can be wired into `OmniMultiViewFusionV5` without breaking the baseline when their auxiliary losses are disabled (`loss_weight=0.0`).
- With v50 loss weight > 0, the main model produces NaN `pred_3d` and camera inputs from the first training step, causing the v50/v51 losses to become NaN and validation to fail.
- Detaching `pred_3d` (and even all v50 inputs: `K`, `R`, `t`, `points_2d`) did not prevent the NaN, suggesting the instability is not from back-prop through the residuals.
- The v50 SEFH NLL loss was stabilized by clamping the depth `z` and masking NaN `points_2d` in the reprojection residual, but the upstream NaN `pred_3d` issue remains.

## Code changes

- Commit `864772e`: detach `pred_3d` in v50 SEFH forward, clamp projection z, add NaN/Inf guards in trainer for v50/v51/aleatoric losses, disable unstable losses in smoke script and A800 queue.

## Next gates

1. Root-cause why enabling v50/v51 auxiliary losses makes the main model emit NaN `pred_3d`/cameras.
2. Re-enable v50/v51 losses once stable and rerun smoke/full A800 runs.
3. Compare v51 CDSVR (heads only) against v46/v50 baselines on the same medium/full config.
