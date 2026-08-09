# v54: Probabilistic Pose Forecasting (PPF)

**Task identifier:** `design_v54_probabilistic_pose_forecasting`  
**Status:** Proposal (no code yet)  
**Depends on:** v52 (`UncertaintyWeightedTriangulationV52`), v53 (`PhysicalSpaceCalibrationV53`), v49-Lite temporal conventions

## 1. Motivation

After v52 uncertainty-weighted triangulation and v53 physical-space calibration, the pipeline has a strong per-frame 3D pose, but each frame is still treated as an independent point estimate. **v54** adds a lightweight, causal probabilistic forecasting head that predicts a Gaussian distribution over the next pose from the calibrated history and uses that forecast as a learned motion prior to refine the current pose. This closes the paper flow: after multi-view fusion and physical-space alignment, the optimized motionflow pipeline exploits temporal dynamics.

The module is **identity at initialization**, so the output equals the v53 input until training opens the correction gate.

## 2. Module overview

**File:** `motionflow_mv/fusion/probabilistic_pose_forecasting_v54.py`

```text
ProbabilisticPoseForecastingV54(
    j=17, hidden=64, n_layers=2, temporal_kernel_size=3,
    forecast_horizon=1, distribution="gaussian_diagonal",
    use_pose_features=True, use_view_count_conditioning=True,
    identity_init=True, residual_gate_init=-6.0,
    loss_weight=0.01, warmup_epochs=0, detach_input=True,
)
```

### 2.1 Inputs / outputs

```python
pred_3d_ref, ppf_dict = ppf(
    pred_3d_init,  # (B, T, J, 3)  -- v53 calibrated pose sequence
    features,        # (B, T, V, J, d) -- optional per-view feature tokens
    view_mask,       # (B, T, V)
    domain_id=None,  # (B,)
)
```

* `pred_3d_ref`: `(B, T, J, 3)` -- refined 3D pose (identity at init).
* `forecast_mean`: `(B, T, J, 3)` -- predicted mean for `t+1`.
* `forecast_log_var`: `(B, T, J, 3)` -- predicted diagonal log-variance for `t+1`.
* `ppf_loss`: scalar -- negative log-likelihood of the observed next pose plus a consistency regularizer.

### 2.2 Architecture and equations

Let `X_t ∈ R^{J×3}` be the v53 pose at frame `t`. Build a per-joint token

```
z_t = mean_v(features_t[v])                 # (B, T, J, d) if use_pose_features, else zero
h_t = Linear_zero_init( concat(X_t, z_t) )  # (B, T, J, hidden)
```

A causal temporal convolution stack processes `h_t` over time:

```
c_t = CausalConv1D_stack( h_1, ..., h_t )   # (B, T, J, hidden)
```

All conv kernels/biases and the input projection are zero-initialized, so `c_t = 0` at init. Two zero-initialized heads predict the next-pose distribution:

```
μ_{t→t+1} = X_t + W_μ c_t + b_μ           # (B, T, J, 3)
Σ_{t→t+1} = diag( exp(W_Σ c_t + b_Σ) )    # (B, T, J, 3)
```

The negative log-likelihood is

```
L_forecast = - Σ_{t=1}^{T-1} log N( X_{t+1} ; μ_{t→t+1}, Σ_{t→t+1} )
```

At init this reduces to `Σ ||X_{t+1} - X_t||^2`, a smoothness term that provides gradients without changing the estimate.

A gated correction uses the previous-frame forecast to refine the current frame:

```
g = sigmoid(γ)    # γ = -6.0 => g ≈ 0.0025
r_t = g * MLP_zero_final( concat( X_t, μ_{t-1→t}, L_{t-1} ) )
X_t' = X_t + r_t
```

At init `r_t = 0`, so `X_t' = X_t`; training then pulls the current pose toward the predicted future.

The final loss is

```
L_ppf = loss_weight * ( L_forecast + λ_cons * Σ_t ||r_t||^2 )
```

with `λ_cons = 0.01`, active only after `warmup_epochs`.

### 2.3 Composability

v54 sits **after** v53 and receives the calibrated pose plus v52 feature tokens. It does not modify triangulation weights; it only adds a temporal motion prior, so it composes cleanly with v45/v46/v47/v48/v49/v50/v51/v52/v53.

## 3. Integration into `OmniMultiViewFusionV5`

### 3.1 New toggles

```python
use_v54_probabilistic_pose_forecasting: bool = False,
v54_ppf_hidden: int = 64,
v54_ppf_n_layers: int = 2,
v54_ppf_temporal_kernel_size: int = 3,
v54_ppf_forecast_horizon: int = 1,
v54_ppf_distribution: str = "gaussian_diagonal",
v54_ppf_use_pose_features: bool = True,
v54_ppf_use_view_count_conditioning: bool = True,
v54_ppf_identity_init: bool = True,
v54_ppf_residual_gate_init: float = -6.0,
v54_ppf_loss_weight: float = 0.01,
v54_ppf_warmup_epochs: int = 0,
v54_ppf_detach_input: bool = True,
```

### 3.2 Wiring

In `OmniMultiViewFusionV5.__init__`, instantiate the module when the flag is true. In `forward`, after the v53 block and before test-time self-evolution / final output:

```python
if self.use_v54_probabilistic_pose_forecasting:
    pred_3d_gn, ppf_loss = self.probabilistic_pose_forecasting_v54(
        pred_3d_init=pred_3d_gn.view(B, T, J, 3),
        features=feat.view(B, T, V, J, self.d),
        view_mask=view_mask_flat.view(B, T, V),
        domain_id=domain_id,
    )
    pred_3d_gn = pred_3d_gn.view(B * T, J, 3)
    self._v54_ppf_loss = ppf_loss
```

Add `self._v54_ppf_loss` to `epi_loss` after `v54_ppf_warmup_epochs`, following the v52/v53 pattern.

## 4. Expected MPJPE impact

* **Smoke (RTX 4090, 50–100 samples):** ≤ 0.5 mm change; identity-at-init preserves the v53 baseline.
* **Medium (500–2k samples):** 1–2 mm improvement by smoothing noisy single-frame estimates.
* **Full (mixed, 10k+ samples):** 2–4 mm improvement over v52/v53, especially on fast motion, self-occlusion, and sparse-view sequences where temporal coherence is most informative.

## 5. Risks

See `docs/swarm_iter28/reports/agent_probabilistic_pose_forecasting_risks.md` for detailed risks and mitigations.

## 6. Implementation plan

1. **Temporal helper:** Add a causal depthwise-separable 1-D convolution helper in `motionflow_mv/fusion/prototypes/causal_conv1d.py` with zero-initialization support.
2. **Module file:** Implement `ProbabilisticPoseForecastingV54` with zero-initialized input projection, forecast heads, correction MLP, and residual gate.
3. **Model wiring:** Add the v54 toggle block to `OmniMultiViewFusionV5.__init__` and `forward`, placing it immediately after v53.
4. **Smoke test:** Create `configs/benchmark_v54_ppf_smoke.yaml` and `scripts/run_v54_ppf_smoke_local_4090.sh`; verify identity-at-init (`||pred_3d_ref - pred_3d_init||_2 < 1e-4`) and finite loss.
5. **Unit tests + ablation:** Add `tests/test_probabilistic_pose_forecasting_v54.py` covering variable sequence length, masked views, gradient flow, and ablation against v53 alone.
