# Results Snapshot 2026-08-09 v3

Snapshot time: 2026-08-09 (active).

## v51 CDSVR local RTX 4090 smoke results

| Run | Epochs | Best val_MPJPE | Notes |
|-----|--------|----------------|-------|
| v51 CDSVR tiny smoke (losses disabled) | 2 | **104.09 mm** | v50/v51 SEFH/CDSVR loss weights 0.0; stable baseline |
| v51 CDSVR medium smoke 200 samples (losses disabled) | 4 | **52.33 mm** | v50/v51 heads wired in, losses disabled; early stopped after no val improvement for 3 epochs |
| v51 CDSVR tiny smoke (v50/v51 MSE loss 0.01) | 2 | **104.51 mm** | Replaced unstable NLL with MSE targets; v50/v51 losses enabled and stable |
| v51 CDSVR medium smoke 200 samples (v50/v51 MSE loss 0.01) v1 | 5 | **suspended** | Loss exploded after ~800 steps (1.8k → 3.4k); v50 only detached pred_3d, allowing gradients through K/R/t/points_2d to poison main model |
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

## v52 Uncertainty-Weighted Triangulation status

| Run | Epochs | Best val_MPJPE | Notes |
|-----|--------|----------------|-------|
| v52 UWT tiny smoke (v50/v51 losses disabled) | 2 | **102.70 mm** | Identity-at-init; training and validation complete without NaN/OOM; comparable to v51 tiny smoke (104.51 mm) |
| v52 UWT medium smoke 200 clips (v50/v51 losses disabled) | 2 | **60.09 mm** | Epoch 1 validation; loss decreased 19.98 → 6.56. Epoch 2 val_MPJPE=76.17 mm. Run is continuing (5 epochs total). |

## v53 Physical-Space Calibration status

| Run | Epochs | Best val_MPJPE | Notes |
|-----|--------|----------------|-------|
| v53 PSC tiny smoke (v50/v51 losses disabled) | 2 | **78.76 mm** | Tiny smoke completed, but `use_v53_physical_space_calibration` was not forwarded to the model due to a trainer `model_kwargs` bug. The bug is now fixed; the next v53 run will exercise the module. |

## Key findings (v52)

- `UncertaintyWeightedTriangulationV52` is implemented and wired into `OmniMultiViewFusionV5` after v25/v45 triangulation.
- The module is identity-at-init: the precision MLP and residual correction are zero-initialized, so a v51 checkpoint loads unchanged.
- The tiny smoke runs to completion and reports `val_MPJPE=102.70 mm`, which is on par with the v51 tiny smoke baseline.

## v53 Physical-Space Calibration status

| Run | Epochs | Best val_MPJPE | Notes |
|-----|--------|----------------|-------|
| v53 PSC tiny smoke (v50/v51 losses disabled) | TBD | **TBD** | Identity-at-init; waiting for first RTX 4090 smoke |

## Key findings (v53)

- `PhysicalSpaceCalibrationV53` is implemented and wired into `OmniMultiViewFusionV5` after the v52 UWT block.
- The module is identity-at-init: the final residual MLP and gate are zero-initialized, so a v52 checkpoint loads unchanged.
- Three auxiliary terms are folded into `psc_loss`: floor-plane calibration, canonical bone-length calibration, and reprojection consistency.
- Per-domain canonical bone lengths are selected via `domain_id` when available.

## v54 Physical-Space Calibration v2 status

| Run | Epochs | Best val_MPJPE | Notes |
|-----|--------|----------------|-------|
| v54 PSC-v2 integration CPU smoke (synthetic, v52+v53+v54) | 1 | **82.02 mm** | Module wired after v53; training/validation complete without NaN/OOM; identity-at-init verified in unit tests |

## Key findings (v54)

- `PhysicalSpaceCalibrationV2V54` is implemented and wired into `OmniMultiViewFusionV5` after the v53 PSC block.
- The module is identity-at-init: the final GNN/MLP residual projection, bone-scale output, and residual gate are zero-initialized, so a v53 checkpoint loads unchanged.
- Four loss terms are folded into `psc2_loss`: floor-plane calibration, velocity-gated foot-floor contact, per-domain canonical bone-length calibration, and temporal smoothness of the correction.
- A shallow skeleton-graph refiner (default `gnn_layers=1`) propagates physical hints along the kinematic chain.

## Trainer fix

- Fixed a bug where `use_v52_uncertainty_weighted_triangulation`, `use_v53_physical_space_calibration`, and `use_v54_physical_space_calibration_v2` were not forwarded to `OmniMultiViewFusionV5` via `model_kwargs`, so the CLI flags had no effect. All three modules are now correctly enabled.

## Next gates

1. Wait for the v52 UWT medium smoke to finish on the RTX 4090, then run a v53 medium smoke and compare against the v52 baseline.
2. Run `bash scripts/run_v54_psc_v2_smoke_local_4090.sh` and verify `val_MPJPE@full` is within 1 mm of the v53 baseline.
3. If v53/v54 smokes are stable, launch the A800 full runs (`v53_physical_space_calibration_on_v52` and `v54_physical_space_calibration_v2_on_v53`) from `scripts/launch_v33_a800_queue.py`.
4. Continue v54 ablations (floor/contact/bone-scale/GNN) once the baseline is validated.
