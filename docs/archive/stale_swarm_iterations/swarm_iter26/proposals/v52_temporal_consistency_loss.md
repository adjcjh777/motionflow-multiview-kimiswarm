# v52: Multi-Scale Temporal Consistency Loss (v52-MSTCL)

**Direction:** `temporal_consistency_loss`

## Motivation

The current `OmniMultiViewFusionV5` pipeline produces a 3-D pose sequence `(B, T, J, 3)` after triangulation, geometry fusion, and optional temporal heads. While v32/v33 add a trajectory refiner and v47/v49 add temporal aggregation, the training objective itself still relies on a per-frame supervised L2 loss. A per-frame loss cannot penalise jitter or implausible velocity spikes that arise from noisy views, dropped frames, or sparse-view training. v52 therefore introduces a **learned, multi-scale, confidence-aware temporal consistency loss** that is added directly to the training objective. Because it is implemented as a loss rather than a refiner, it adds no extra inference latency and is naturally identity-at-init by ramping its weight from zero.

## Architecture

`v52_MultiScaleTemporalConsistencyLoss` is a small `nn.Module` instantiated inside `OmniMultiViewFusionV5`. It consumes the per-frame 3-D pose `pred_3d` before any final kinematic head and returns a scalar auxiliary loss.

### Inputs and shapes

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Triangulated/refined 3-D pose sequence. |
| `conf` | `(B, T, J)` | Per-joint visibility/confidence, already computed from `x[..., 2]` and `view_mask`. |
| `reproj_err` | `(B, T, V, J)` | Optional per-view-joint reprojection error used to derive temporal uncertainty. |
| `domain_id` | `(B,)` | Optional domain label for per-domain loss scaling. |

### Learned multi-scale weights

For each temporal scale `τ ∈ v52_tcl_scales`, the module predicts a per-joint weight `w_τ ∈ (0, 1)^J`:

```
logit_τ = MLP_τ( mean_j(reproj_err[:, :, :, j]) )   # (B, T, J) -> aggregate over views
w_τ     = sigmoid(logit_τ)                            # (B, T, J)
```

The final MLP layer is initialised to zero so that `w_τ ≈ 0.5` at start and the loss is dominated by the global scalar weight `v52_tcl_weight`, which is ramped linearly for `v52_tcl_warmup_epochs` epochs.

### Loss terms

**1. Multi-scale smoothness loss**

For a scale `τ`, define the `τ`-step finite difference:

```
Δ_τ pred_3d(b, t, j) = pred_3d(b, t+τ, j) - pred_3d(b, t, j)
```

Then:

```
L_smooth(τ) = Σ_t Σ_j w_τ(t, j) · c(t, j) · ρ( ||Δ_τ pred_3d(b, t, j)||_2 , δ_τ )
```

where `c(t, j)` is the confidence mask, `δ_τ = v52_tcl_huber_delta · τ` (larger scale -> larger Huber threshold), and `ρ` is the Huber loss.

**2. Bone-length temporal consistency loss**

Using the kinematic parent list `parents`:

```
bone(t, j)  = pred_3d(t, child(j)) - pred_3d(t, parent(j))
L_bone      = Σ_t Σ_j || ||bone(t, j)||_2 - ||bone(t-1, j)||_2 ||_2^2
```

**Total v52 loss**

```
L_v52 = ramp(λ) · [ L_smooth + α · L_bone ]
```

with `λ = v52_tcl_weight`, `α = v52_tcl_bone_weight`, and `ramp` linear from 0 to 1 over `v52_tcl_warmup_epochs`.

## Config flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v52_temporal_consistency_loss` | bool | `False` | Master switch. |
| `v52_tcl_weight` | float | `0.01` | Overall loss weight `λ`. |
| `v52_tcl_scales` | list | `[1, 2, 4]` | Temporal finite-difference scales. |
| `v52_tcl_huber_delta` | float | `0.05` | Base Huber delta in metres. |
| `v52_tcl_warmup_epochs` | int | `1` | Linear ramp-up of `λ`. |
| `v52_tcl_bone_weight` | float | `0.1` | Weight `α` for bone-length temporal term. |
| `v52_tcl_learned_scale_weights` | bool | `True` | Use per-joint learned scale weights. |
| `v52_tcl_min_views_for_loss` | int | `2` | Only apply loss when the clip has at least this many active views. |
| `v52_tcl_use_reproj_confidence` | bool | `True` | Derive `w_τ` from reprojection errors. |
| `v52_tcl_per_domain_scale` | bool | `False` | Learn a separate `λ` per domain. |

## Expected MPJPE impact

Local smoke (d=64, 50 samples, 2 epochs) on top of the v46/v49 baseline should show `val_MPJPE` remain within `±0.5 mm` while reducing per-joint acceleration error by `≥15 %`. On the full A800 run, the largest gains are expected on sequences with fast motion or sparse views: `MPJPE@2` and `MPJPE@3` should improve by `1.0–2.0 mm`, while `MPJPE@full` improves by `0.3–0.8 mm` by suppressing frame-to-frame jitter.

## Risks

See `docs/swarm_iter26/reports/agent_temporal_consistency_loss_risks.md` for the full risk register.

## 5-step implementation plan

1. **Module stub.** Create `motionflow_mv/losses/temporal_consistency_loss_v52.py` with `MultiScaleTemporalConsistencyLossV52` implementing the two loss terms and the warmup ramp. Initialise the learned scale-weight MLP to zero.
2. **Wire into V5.** Add the nine config flags to `motionflow_mv/fusion/omniview_fusion_v5.py`, instantiate the loss module in `__init__`, and call it inside `forward` immediately after the `pred_3d = pred_3d.view(B, T, J, 3)` reshape. Add the returned scalar to `epi_loss`.
3. **Trainer plumbing.** Expose the flags in `experiments/train_omniview_fusion_v5_webbridge_multi.py` and pass them through `build_model_from_args`. Log `loss_v52_smooth` and `loss_v52_bone` separately.
4. **Smoke test.** Add `configs/benchmark_v52_temporal_consistency_loss_smoke.yaml` and a shell script. Run a 50-sample smoke on the local RTX 4090; verify that the loss starts at zero (warmup) and that training does not NaN.
5. **Ablate and queue.** Sweep `v52_tcl_weight ∈ {0.001, 0.01, 0.05}` and `v52_tcl_scales` against the v46/v49 baseline. Commit the winning variant to `scripts/launch_v33_a800_queue.py` and start the A800 full run.
