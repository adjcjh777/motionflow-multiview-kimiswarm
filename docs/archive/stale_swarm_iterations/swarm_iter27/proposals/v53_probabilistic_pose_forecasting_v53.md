# v53 Probabilistic Pose Forecasting (PPF)

**Module:** `motionflow_mv/fusion/probabilistic_pose_forecasting_v53.py`  
**Integration:** `motionflow_mv/fusion/omniview_fusion_v5.py` after `v52` uncertainty-weighted triangulation and the residual MLP, before the final kinematic/physical refiner.  
**Tracking issue:** #185 (placeholder)

## 1. Motivation

The current pipeline (`v25` geometry fusion → `v46` sparse-view reliability → `v47`/`v49` temporal heads → `v48` domain adaptation → `v50`/`v51` self-evolution → `v52` uncertainty-weighted triangulation) produces per-frame 3-D poses but does not explicitly model the **temporal distribution** of the human pose.  Physical-space alignment downstream is sensitive to jitter and to single-frame triangulation failures.  By forecasting a per-joint Gaussian pose distribution from the recent uncertainty-weighted history, the v53 module can:

1. smooth the current frame with a learned Kalman-like correction that respects per-joint uncertainty,
2. regularize training via a next-frame negative log-likelihood (NLL) loss, and
3. feed a temporally coherent pose into the physical/kinematic refiner.

Crucially, the module is **warm-startable/identity-at-init**: its correction gate and final projection are zero-initialized, so loading a `v52` checkpoint with `use_v53_probabilistic_pose_forecasting=True` leaves the baseline MPJPE unchanged.

## 2. Architecture

The module is a lightweight causal forecaster.  It consumes the current residual pose sequence `X ∈ R^(B×T×J×3)` and the per-view/joint precision weights `W ∈ R^(B×T×V×J)` produced by `v52`.  It outputs a smoothed pose sequence `X' ∈ R^(B×T×J×3)` and an auxiliary forecasting loss.

### 2.1 Per-joint precision from v52 weights

For every frame `t`, joint `j`:

```
λ_tj = clamp( Σ_v W_tvj + ε, min=1e-3, max=1e3 )   ∈ R^{B×T×J}
```

`λ_tj` acts as the inverse observation variance (precision).  When `v52` weights are at their init value of ~0.5, all joints receive equal precision.

### 2.2 Temporal feature encoding

For a causal window `w` (default `w=5`, flag `v53_ppfc_window`), build a per-joint feature vector:

```
h_t = MLP( concat[ X_{t-w+1:t},  log(λ_{t-w+1:t}),  ΔX_{t-w+1:t} ] )   ∈ R^{B×T×J×d}
```

where `ΔX_t = X_t - X_{t-1}` is the finite-difference velocity and `d = v53_ppfc_hidden`.  The encoder is a 2-layer MLP with ReLU.  At the sequence start, missing frames are zero-padded and masked.

### 2.3 Probabilistic head

A small per-joint MLP predicts the parameters of a Gaussian distribution for the **next** frame:

```
μ_{t+1} = MLP_μ(h_t)                         ∈ R^{B×T×J×3}
log σ_{t+1} = MLP_σ(h_t)                     ∈ R^{B×T×J×3}
σ_{t+1} = softplus(log σ_{t+1}) + v53_ppfc_min_std
```

`MLP_μ` and `MLP_σ` each have `v53_ppfc_n_layers` hidden layers.  Their final layers are **zero-initialized**, so at init `μ_{t+1} = X_t` (zero residual forecast) and `σ_{t+1} = min_std`.

### 2.4 Uncertainty-gated smoothing

The forecast is fused back into the current frame via a learnable gate:

```
δ_t  = gate( h_t ) · (μ_{t+1} - X_t)        ∈ R^{B×T×J×3}
X'_t = X_t + δ_t
```

`gate(·) ∈ (0,1)` is produced by a sigmoid-activated per-joint MLP whose final layer is zero-initialized; hence `gate ≈ 0.5` at init and the smoothing residual is tiny.  To make the module strictly identity-at-init, the gate scalar parameter `γ` is introduced:

```
X'_t = X_t + γ · gate(·) · (μ_{t+1} - X_t),   γ initialized to 0.0
```

## 3. Loss

Two auxiliary terms are added to the existing `epi_loss`:

1. **Negative log-likelihood (NLL) on the next true pose** (treat the supervision `Y_{t+1}` as the ground-truth 3-D pose):

```
L_nll = -Σ_t Σ_j log N( Y_{t+1,j} | μ_{t+1,j}, diag(σ²_{t+1,j}) )
```

2. **Reprojection-aware precision loss** that encourages the predicted precision `1/σ²` to correlate with reprojection error `ρ` of the current frame:

```
L_precision = Σ_t Σ_j | exp(-ρ_tj / 5.0) - 1/(1 + σ²_{t+1,j}) |
```

Total auxiliary loss: `L_v53 = v53_ppfc_loss_weight · (L_nll + 0.01 · L_precision)`.  `v53_ppfc_loss_weight` starts at `0.0` for `v53_ppfc_warmup_epochs` to preserve the warm start.

## 4. Inputs / Outputs

| Symbol | Tensor shape | Description |
|--------|--------------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Per-frame 3-D pose from upstream fusion/residual MLP. |
| `v52_weights` | `(B, T, V, J)` | Per-view/joint triangulation weights from `v52` UWT. |
| `v52_log_precision` | `(B, T, V, J)` | Raw log-precision from `v52` (optional, for richer features). |
| `view_mask` | `(B, T, V)` | Boolean mask for valid views. |
| `pred_3d_ppfc` | `(B, T, J, 3)` | Smoothed/forecasted pose sequence. |
| `ppfc_loss` | `()` | Scalar auxiliary NLL + precision loss. |
| `ppfc_std` | `(B, T, J, 3)` | Predicted per-joint standard deviations. |

## 5. Config flags

```yaml
use_v53_probabilistic_pose_forecasting: false
v53_ppfc_hidden: 64
v53_ppfc_n_layers: 2
v53_ppfc_window: 5
v53_ppfc_min_std: 0.005          # ~5 mm floor on predicted std
v53_ppfc_loss_weight: 1.0
v53_ppfc_identity_init: true
v53_ppfc_use_uncertainty_gate: true
v53_ppfc_warmup_epochs: 1
```

## 6. Expected MPJPE impact

| Scenario | Expected Δ MPJPE | Rationale |
|----------|------------------|-----------|
| Local RTX 4090 smoke | **−0.8 to −1.5 mm** | Small clips benefit most from temporal smoothing and NLL regularization. |
| Full A800 benchmark | **−0.3 to −0.8 mm** | Diminishing returns on already strong `v52` baseline, but temporal coherence reduces outlier frames. |
| Variable-view eval (`MPJPE@2`) | **−1.0 to −2.0 mm** | Uncertainty-weighted forecasting is most valuable when only 2 views are available. |

## 7. Risks (see linked report for mitigations)

1. **Over-smoothing** of fast motions due to the temporal gate.
2. **Variance collapse** if the NLL loss drives `σ → min_std` too aggressively.
3. **Redundancy** with `v47`/`v49` temporal modules; gated stacking may hurt rather than help.
4. **Memory overhead** from the extra temporal window and MLP heads.

## 8. 5-step implementation plan

1. **Create `motionflow_mv/fusion/probabilistic_pose_forecasting_v53.py`** implementing the causal encoder, probabilistic head, and identity-at-init gate.  Add unit tests that verify `X' == X` at init and that output shapes match the table above.
2. **Wire into `omniview_fusion_v5.py`** after the residual MLP block: accept `pred_3d`, `v52_weights`, and `view_mask`; return the smoothed pose and `ppfc_loss`; add the loss to `epi_loss` with warmup gating.
3. **Add YAML smoke config** `configs/benchmark_v53_probabilistic_pose_forecasting_smoke.yaml` by copying `benchmark_v52_uwt_smoke.yaml` and enabling the v53 flags.  Run `scripts/run_v53_ppfc_smoke_local_4090.sh`.
4. **Verify warm-start identity**: load the latest `v52` checkpoint with `use_v53=True`; confirm `val_MPJPE` drift < 0.1 mm before training begins, and after `v53_ppfc_warmup_epochs` the loss turns on smoothly.
5. **Ablate and queue**: run a 2-epoch comparison against `v52` on local smoke; if ΔMPJPE < −0.5 mm and no NaN/OOM, append an entry to `scripts/launch_v33_a800_queue.py` for the full A800 run.
