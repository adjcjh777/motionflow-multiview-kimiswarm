# v53: Uncertainty-Guided Temporal Consistency Loss (v53-UGTCL)

**Direction:** `temporal_consistency_loss_v53`
**Depends on:** v25 geometry fusion, v45 adaptive geometry fusion, v46 sparse-view generalization, v47 temporal aggregation, v49-Lite causal temporal, v50 self-evolution feedback head, v51 cross-domain sparse-view reliability, v52 uncertainty-weighted triangulation

## 1. Motivation

The MotionFlow-MultiView pipeline triangulates per-view 2-D keypoints into a 3-D pose, then refines it through v45/v46/v47/v49/v50/v51 and v52 uncertainty-weighted triangulation. While these modules improve per-frame pose quality, the final training objective still relies heavily on per-frame supervised L2 loss. This objective cannot directly penalise frame-to-frame jitter, implausible velocity spikes, or bone-length flicker that arise from noisy views, sparse-view dropout, or cross-domain distribution shifts.

v52 UWT produces per-view, per-joint precision weights `w_btvj ∈ [0,1]` that measure how much each view contributes to the triangulated 3-D point. These weights encode a natural per-joint temporal uncertainty: a joint with consistently low precision across views is less trustworthy, and its temporal neighbours should be regularised more strongly. v53 therefore introduces an **Uncertainty-Guided Temporal Consistency Loss (UGTCL)** that reuses the v52 precision weights to modulate temporal smoothness, velocity, and bone-length consistency. It is implemented as a pure loss term, so it adds no inference-time module and remains **identity-at-init** by ramping its global weight from zero.

## 2. Architecture

`v53_TemporalConsistencyLossV53` is instantiated inside `OmniMultiViewFusionV5`. It consumes the refined 3-D pose `pred_3d` and the v52 UWT weights `v52_weights`, then returns a scalar auxiliary loss that is added to the total training objective.

### 2.1 Inputs and shapes

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Refined 3-D pose sequence after v52 UWT. |
| `v52_weights` | `(B, T, V, J)` | Per-view, per-joint triangulation weights from v52 UWT. |
| `view_mask` | `(B, T, V)` | Boolean mask for valid views. |
| `domain_id` | `(B,)` | Optional integer domain labels for per-domain loss scaling. |

### 2.2 Per-joint temporal uncertainty from v52 weights

For each joint `j` at time `t`, compute the effective precision as the mean v52 weight across visible views:

```
precision(t, j) = sum_v v52_weights(t, v, j) * view_mask(t, v) / (sum_v view_mask(t, v) + eps)   # (B, T, J)
conf(t, j)      = sqrt(precision(t, j))                                                          # (B, T, J)
```

`conf` is clamped to `[v53_tcl_min_conf, 1.0]`. A small MLP optionally predicts per-joint scale weights from the precision sequence, but the final layer is zero-initialised so that all weights start at `0.5` and the loss is dominated by the global ramped weight.

### 2.3 Loss terms

**1. Multi-scale smoothness loss**

For temporal scale `τ ∈ v53_tcl_scales`, define the finite difference:

```
Δ_τ pred_3d(b, t, j) = pred_3d(b, t+τ, j) - pred_3d(b, t, j)
```

Then the smoothness loss is:

```
L_smooth(τ) = Σ_t Σ_j w_τ(t, j) · conf(t, j) · ρ( ||Δ_τ pred_3d(b, t, j)||_2 , δ_τ )
```

where `ρ` is the Huber loss and `δ_τ = v53_tcl_huber_delta · τ`. The learned per-joint scale weights `w_τ` are initialised to zero so `sigmoid(0) = 0.5` at start.

**2. Velocity consistency loss**

Penalise changes in joint velocity across consecutive frames:

```
vel(t, j)      = pred_3d(t, j) - pred_3d(t-1, j)
L_velocity     = Σ_t Σ_j conf(t, j) · || vel(t, j) - vel(t-1, j) ||_2^2
```

**3. Bone-length temporal consistency loss**

Using the kinematic parent list `parents`:

```
bone(t, j)   = pred_3d(t, child(j)) - pred_3d(t, parent(j))
L_bone       = Σ_t Σ_j conf(t, j) · ( ||bone(t, j)||_2 - ||bone(t-1, j)||_2 )^2
```

**Total v53 loss**

```
L_v53 = ramp(λ) · [ L_smooth + α · L_velocity + β · L_bone ]
```

with `λ = v53_tcl_weight`, `α = v53_tcl_velocity_weight`, `β = v53_tcl_bone_weight`, and `ramp` linearly increasing from `0` to `1` over `v53_tcl_warmup_epochs`.

## 3. Config flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v53_temporal_consistency_loss` | bool | `False` | Master switch. |
| `v53_tcl_weight` | float | `0.01` | Overall loss weight `λ`. |
| `v53_tcl_warmup_epochs` | int | `1` | Linear ramp-up of `λ`. |
| `v53_tcl_scales` | list | `[1, 2, 4]` | Temporal finite-difference scales. |
| `v53_tcl_huber_delta` | float | `0.05` | Base Huber delta in metres. |
| `v53_tcl_velocity_weight` | float | `0.1` | Weight `α` for velocity consistency. |
| `v53_tcl_bone_weight` | float | `0.1` | Weight `β` for bone-length consistency. |
| `v53_tcl_min_conf` | float | `0.05` | Floor for per-joint confidence from v52 weights. |
| `v53_tcl_learned_scale_weights` | bool | `True` | Use learned per-joint scale weights. |
| `v53_tcl_min_views_for_loss` | int | `2` | Only apply loss when at least this many views are active. |
| `v53_tcl_per_domain_scale` | bool | `False` | Learn a separate `λ` per domain. |

## 4. Expected MPJPE impact

* **Smoke (RTX 4090, 50–100 samples):** `val_MPJPE` should change by `≤ 0.5 mm` because the loss is ramped from zero and operates on the already-refined v52 output.
* **Medium/full runs:** `MPJPE@2` and `MPJPE@3` should improve by `1.0–2.5 mm` on fast-motion and sparse-view sequences by suppressing jitter. `MPJPE@full` is expected to improve by `0.4–1.0 mm`.
* **Acceleration error:** Per-joint acceleration root-mean-square should drop by `≥ 15 %` without visible over-smoothing on static poses.

## 5. Risks

See `docs/swarm_iter27/reports/agent_temporal_consistency_loss_v53_risks.md` for the full risk register.

## 6. 5-step implementation plan

1. **Module stub.** Create `motionflow_mv/losses/temporal_consistency_loss_v53.py` with `TemporalConsistencyLossV53` implementing the three loss terms, the warmup ramp, and the per-joint confidence derived from v52 weights. Zero-initialise the learned scale-weight MLP final layer.
2. **Wire into V5.** Add the config flags to `motionflow_mv/fusion/omniview_fusion_v5.py`, instantiate the loss module in `__init__`, and call it inside `forward` after the v52 UWT block returns `pred_3d` and `v52_weights`. Add the returned scalar to the auxiliary loss dictionary.
3. **Trainer plumbing.** Expose the flags in `experiments/train_omniview_fusion_v5_webbridge_multi.py` and pass them through `build_model_from_args`. Log `loss_v53_smooth`, `loss_v53_velocity`, and `loss_v53_bone` separately.
4. **Smoke test.** Create `configs/benchmark_v53_tcl_smoke.yaml` and `scripts/run_v53_tcl_smoke_local_4090.sh`. Run a 50-sample smoke on the local RTX 4090; verify the loss starts at zero and that `v52_weights` flow into the confidence computation without NaN.
5. **Ablate and queue.** Sweep `v53_tcl_weight ∈ {0.001, 0.01, 0.05}` and `v53_tcl_scales` against the v52 UWT baseline. Commit the winning variant to `scripts/launch_v33_a800_queue.py` and start the A800 full run.
