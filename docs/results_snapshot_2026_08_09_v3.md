# Results Snapshot 2026-08-09 v3

Snapshot time: 2026-08-09 (active).

## v51 CDSVR local RTX 4090 smoke results

| Run | Epochs | Best val_MPJPE | Notes |
|-----|--------|----------------|-------|
| v51 CDSVR tiny smoke (losses disabled) | 2 | **104.09 mm** | v50/v51 SEFH/CDSVR loss weights 0.0; stable baseline |
| v51 CDSVR medium smoke 200 samples (losses disabled) | 4 | **52.33 mm** | v50/v51 heads wired in, losses disabled; early stopped after no val improvement for 3 epochs |
| v51 CDSVR tiny smoke (v50/v51 MSE loss 0.01) | 2 | **104.51 mm** | Replaced unstable NLL with MSE targets; v50/v51 losses enabled and stable |
| v51 CDSVR medium smoke 200 samples (v50/v51 MSE loss 0.01) v1 | 5 | **duspended** | Loss exploded after ~800 steps (1.8k → 3.4k); v50 only detached pred_3d, allowing gradients through K/R/t/points_2d to poison main model |
| v51 CDSVR medium smoke 200 samples (v50/v51 MSE loss 0.01) v2 | 5 | **suspended** | Loss still exploded (107 → 15k) even with all v50 inputs detached; MSE auxiliary loss is unstable beyond tiny smoke |
| v51 CDSVR medium smoke 200 samples (heads only, losses 0.0) | TBD | TBD | Re-running with v50/v51 auxiliary losses disabled; heads remain wired in for future inference use |

## Key findings

- `SelfEvolutionFeedbackHeadV50` and `CrossDomainSparseViewReliabilityV51` heads can be wired into `OmniMultiViewFusionV5` without breaking the baseline when their auxiliary losses are disabled (`loss_weight=0.0`).
- The original v50 SEFH loss used a Gaussian negative-log-likelihood with `exp(log_var)` and division by `sigma^2`, which produced NaNs as soon as the loss weight was > 0.
- Replacing the NLL with stable MSE targets removes the NaN:
  - `target_reliability = exp(-reproj / 5.0)`
  - `target_log_var = log(reproj + 1.0).mean(dim=2)`
  - `loss = MSE(reliability, target_reliability) + 0.1 * MSE(log_var, target_log_var) + 0.01 * temporal_smoothness`
- The v50/v51 auxiliary losses are now re-enabled in the local smoke script and in the A800 queue.

## Code changes

- Commit `864772e`: detach `pred_3d` in v50 SEFH forward, clamp projection z, add NaN/Inf guards in trainer for v50/v51/aleatoric losses, disable unstable losses in smoke script and A800 queue.
- Commit `6d29b91`: replace unstable NLL with MSE-based SEFH loss, re-enable v50/v51 auxiliary losses, fix a stray double backslash in the smoke script, add medium smoke script.

## Next gates

1. Wait for the v51 CDSVR medium smoke (200 samples / 5 epochs) to finish.
2. Compare v51 CDSVR (heads + MSE losses enabled) against the heads-only baseline (52.33 mm).
3. If medium smoke is promising, revert the smoke script to 500 samples or launch the full A800 run.
