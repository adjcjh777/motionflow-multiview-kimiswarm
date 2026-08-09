# v55 Probabilistic Pose Forecasting (PPF)

## 1. Module name and one-line purpose

- **Module:** `ProbabilisticPoseForecastingV55` → `motionflow_mv/fusion/probabilistic_pose_forecasting_v55.py`
- **One-line purpose:** A causal, uncertainty-aware probabilistic forecast head that predicts a Gaussian distribution over the next-frame pose correction from the v54 physically-calibrated pose sequence, and refines the current-frame estimate through a gated residual.

## 2. Placement in the OmniMultiViewFusionV5 forward pass

PPF sits **after the v54 PSC-v2 block** and **before the final residual MLP / v47/v49 temporal / v50 SEFH heads**.

```text
points_2d, confidences, K, R, t
    
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss
    ↓
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, psc_loss
    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_loss
    ↓
v55 ProbabilisticPoseForecastingV55
    (consumes pred_3d_psc2 history, uwt_weights, view_mask, domain_id)
    → pred_3d_ppf, ppf_loss, ppf_mean, ppf_std
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

PPF does **not** replace v54; it adds a short-horizon temporal forecast on top of the already physically-calibrated pose, giving downstream heads a temporally-smoothed, uncertainty-aware input.

## 3. Inputs, outputs, and shapes

### Inputs

| Tensor | Shape | Description |
|---|---|---|
| `pred_3d_psc2` | `(B, T, J, 3)` | Current and past v54-calibrated 3D poses. `T` is the causal forecast window. |
| `uwt_weights` | `(B, T, V, J)` | v52 UWT per-view/joint reliability weights. |
| `view_mask` | `(B, T, V)` | Binary visibility mask for each view. |
| `domain_id` | `(B,)` or `(B, T)` | Domain index for optional per-domain normalization. |
| `pred_3d_init` | `(B, T, J, 3)` | Initial triangulated pose from v25/v45 (for optional reprojection term). |

### Outputs

| Tensor | Shape | Description |
|---|---|---|
| `pred_3d_ppf` | `(B, J, 3)` | Current-frame pose after the gated forecast correction. |
| `ppf_loss` | `()` | Scalar auxiliary loss (forecast NLL + temporal consistency + optional reprojection). |
| `ppf_mean` | `(B, J, 3)` | Predicted Gaussian mean for the next-frame correction (diagnostic). |
| `ppf_std` | `(B, J, 3)` | Predicted Gaussian std for the next-frame correction (diagnostic). |

## 4. Architecture

### Per-joint feature extraction

For each frame in the window, build a per-joint feature vector by concatenating:

- The v54 pose: `pred_3d_psc2` `(B, T, J, 3)`
- Velocity: finite difference over the window, padded: `vel_t = pred_3d_psc2_t - pred_3d_psc2_{t-1}` `(B, T, J, 3)`
- Visibility/uncertainty cue: per-joint mean and variance of `uwt_weights` across views, masked by `view_mask` → `(B, T, J, 2)`

A -joint MLP (`hidden=64`, 2 layers, ReLU) maps the concatenated feature to `(B, T, J, D)` with `D=64`.

### Causal temporal encoder

A single-layer causal Transformer (or 1D causal conv with kernel 3 as fallback) runs over the `T` time steps for each joint independently, with causal masking. Output: `(B, T, J, D)`. Only the last time step is used for prediction.

### Probabilistic forecast head

From the current-frame token, two small MLPs predict:

- `ppf_mean`: predicted correction to the current-frame pose, shape `(B, J, 3)`.
- `ppf_log_std`: predicted log-standard-deviation of the next-frame pose, shape `(B, J, 3)`.

The final layer of the mean MLP is **zero-initialized**, so the initial correction is zero. The log_std layer is initialized to `log(0.05 m)` to start from a sensible, bounded uncertainty.

### Gated residual

```
gate = sigmoid(v55_ppf_residual_gate_init)            # default ≈ 0.0025
pred_3d_ppf = pred_3d_psc2[:, -1] + gate * tanh(ppf_mean)
```

The `tanh` clamps the correction; the gate starts near zero, preserving the v54 checkpoint exactly at initialization.

### Losses

| Loss | Description | Weight |
|---|---|---|
| `L_forecast_nll` | Negative log-likelihood of the true next-frame pose under `N(ppf_mean, softplus(ppf_log_std))`, masked to valid joints. | `v55_ppf_nll_weight` (default `1.0`) |
| `L_temporal_consistency` | L2 smoothness of `pred_3d_ppf` against the previous frame, weighted by inverse `uwt_weights` uncertainty. | `v55_ppf_temporal_weight` (default `0.1`) |
| `L_reproj` | Optional reprojection consistency of `pred_3d_ppf` to 2D keypoints (only when `pred_3d_init` is provided). | `v55_ppf_reproj_weight` (default `0.0`) |

Total auxiliary loss:

```
ppf_loss = v55_ppf_loss_weight * (
    v55_ppf_nll_weight * L_forecast_nll
  + v55_ppf_temporal_weight * L_temporal_consistency
  + v55_ppf_reproj_weight * L_reproj
)
```

### Identity-at-init mechanism

- Final projection of the **mean MLP** is zero-initialized.
- **Gate logit** is initialized to `-6.0` so `gate ≈ 0.0025`.
- **Log_std** is initialized to a constant (no pose correction effect). `ppf_std` is bounded by `softplus` to keep the distribution valid.
- When `T=1` (single-frame input) the module is designed to be a no-op: all temporal computations return the current token unchanged and `L_forecast_nll` is not computed.

## 5. Expected MPJPE impact and main risks

| View setting | Expected MPJPE impact |
|---|---|
| Full views | smoke `−0.3 to −0.8 mm`; full `−1.0 to −2.5 mm` |
| Sparse `@2/3` | `−1.0 to −3.0 mm` (larger gain because forecasting borrows information across frames) |

### Main risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| **Forecast horizon too short / no temporal context** | Zero or negative gain on static sequences. | Default window `T=5`; skip `L_forecast_nll` when `T<2`. |
| **NLL loss dominated by fast-motion joints** | MPJPE rises on high-velocity actions. | Per-joint weighting by inverse UWT uncertainty; clamp `ppf_std` to `[1e-3, 1.0]` m. |
| **Over-smoothing / latency** | Joints lag behind true motion. | Gate initialized near zero; small hidden dim (`64`); auxiliary loss weight kept small. |
| **Identity-at-init failure** | v54 checkpoint changes by `>0.1 mm` when v55 enabled. | Zero-init mean MLP, gate logit `−6.0`; unit test `||pred_ppf - pred_psc2||_∞ < 1e-4`. |
| **Memory / OOM from temporal window** | RTX 4090 OOM during smoke. | Default `T=5`; provide causal-conv fallback; window can be reduced to `3`. |

## 6. Smoke acceptance criteria

- `val_MPJPE@full` is within `1 mm` of the v54-PSC-v2 baseline on the same smoke config and seed.
- Identity-at-init: loading a v54 checkpoint with v55 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
- No NaN, Inf, or OOM through at least one full epoch.
- Predicted `ppf_std` stays in `[1e-3, 1.0]` m for `≥95%` of joints.
- `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.
- When `T=1`, `ppf_loss` is zero and `pred_3d_ppf` equals `pred_3d_psc2[:, -1]` within `1e-4`.

## 7. Required new files and files to modify

### New files

- `motionflow_mv/fusion/probabilistic_pose_forecasting_v55.py` — `ProbabilisticPoseForecastingV55` module.
- `configs/benchmark_v55_ppf_smoke.yaml` — smoke config copied from `configs/benchmark_v54_psc_v2_smoke.yaml` with v55 flags enabled.
- `scripts/run_v55_ppf_smoke_local_4090.sh` — smoke launch script that warm-starts from the best v54 checkpoint.
- `tests/test_probabilistic_pose_forecasting_v55.py` — unit tests for identity-at-init, std bounds, single-frame no-op, and gradient flow.

### Files to modify

- `motionflow_mv/fusion/omniview_fusion_v5.py` — add the v55 flag block in `__init__`, instantiate `ProbabilisticPoseForecastingV55` when enabled, call it after the v54 PSC-v2 block, and add `ppf_loss` to `epi_loss`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — aggregate `loss_dict["v55_ppf"]` with `v55_ppf_loss_weight` and the warmup guard.
- `scripts/launch_v33_a800_queue.py` — add the v55 full-run entry on top of the best v54 checkpoint.

### Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_probabilistic_pose_forecasting` | bool | `False` | Master toggle |
| `v55_ppf_hidden` | int | `64` | Feature / MLP hidden dimension |
| `v55_ppf_n_layers` | int | `2` | Per-joint MLP depth |
| `v55_ppf_window` | int | `5` | Causal temporal window length `T` |
| `v55_ppf_n_heads` | int | `4` | Causal Transformer heads (if used) |
| `v55_ppf_identity_init` | bool | `True` | Zero-initialize final mean layer and gate |
| `v55_ppf_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init |
| `v55_ppf_loss_weight` | float | `1.0` | Multiplier on total `L_ppf` |
| `v55_ppf_nll_weight` | float | `1.0` | Weight of forecast NLL loss |
| `v55_ppf_temporal_weight` | float | `0.1` | Weight of temporal consistency loss |
| `v55_ppf_reproj_weight` | float | `0.0` | Weight of optional reprojection term |
| `v55_ppf_std_min` | float | `1e-3` | Lower bound on predicted std |
| `v55_ppf_std_max` | float | `1.0` | Upper bound on predicted std |
| `v55_ppf_warmup_epochs` | int | `0` | Epochs before `ppf_loss` contributes to total loss |
