# v54 — Learned View Selection Policy for Robust Multi-View Triangulation

## 1. Motivation

The current `v45/v52/v53` fusion chain produces per-view, per-joint **continuous weights** for triangulation, but it does not explicitly solve the **combinatorial view-selection problem**: for each joint, which subset of the available views carries reliable geometric evidence? A small number of noisy/occluded views can dominate the weighted DLT solve because continuous weights still retain every view. Earlier modules (v46 sparse-view dropout, v51 cross-domain reliability) estimate quality, yet none of them learn an explicit, differentiable policy that *decides* whether a view is worth including.

`v54_learned_view_selection_policy` closes this gap. It formulates view selection as a differentiable policy-optimization problem on top of the `v52` uncertainty-weighted triangulation and `v53` physical-space calibration outputs. At training time the policy learns to drop views whose 2-D evidence would degrade the 3-D estimate; at inference it selects a compact, interpretable subset of views with negligible overhead.

## 2. Architecture

### 2.1 Placement in `OmniMultiViewFusionV5`

The module is inserted **after** `v53_physical_space_calibration_v53` and before the optional test-time refiners (v27/v29). It re-uses the per-view features `feat` shaped `(B, T, V, J, d)` and the refined 3-D pose `pred_3d_gn` shaped `(B, T, J, 3)`. Its output is a set of selection weights `s ∈ [0,1]^{B×T×V×J}` that gate the triangulation weights produced by v52, allowing a downstream re-triangulation or residual correction to benefit from a cleaner view subset.

### 2.2 Inputs

| Symbol | Shape | Description |
|--------|-------|-------------|
| `feat` | `(B, T, V, J, d)` | Transformer features produced by the ST transformer, same tensor used by v52. |
| `points_2d` | `(B, T, V, J, 2)` | Undistorted 2-D joint detections per view. |
| `K, R, t` | `(B, T, V, 3, 3)`, `(B, T, V, 3, 3)`, `(B, T, V, 3)` | Calibrated camera parameters. |
| `pred_3d` | `(B, T, J, 3)` | Current 3-D pose estimate from v53. |
| `current_weights` | `(B, T, V, J)` | Continuous per-view weights from `v52` UWT (or uniform if v52 is disabled). |
| `view_mask` | `(B, T, V)` | Binary mask of actually available views. |

### 2.3 Policy Network

For each `(view, joint)` token we build a state vector `z_{b,t,v,j}` by concatenating four sources:

1. Feature evidence: `feat[b,t,v,j]` → `(d,)`.  
2. Geometry evidence: ray direction `d_{b,t,v,j} = R_{b,t,v}^{-1}(p_{b,t,v,j} - t_{b,t,v})` where `p` is the back-projected unit-depth ray, normalized → `(3,)`.  
3. Residual evidence: `r_{b,t,v,j} = ||Π_{b,t,v}(pred_3d) - points_2d|`_2 ` → `(1,)`.  
4. Current weight: `current_weights[b,t,v,j]` → `(1,)`.

The state is fed through an MLP:

```
z = MLP([d + 3 + 1 + 1, hidden, ..., hidden])    # (B*T*V*J, hidden)
logits = Linear(hidden, 1)                         # unnormalized logit per (view,joint)
```

The final selection probability is

```
s = σ(logits / temperature)        # soft selection, or
s = StraightThroughGumbel(logits)  # hard selection with straight-through gradient
```

### 2.4 Differentiable Top-K / Budget Constraint

Because a simple per-view binary mask is under-constrained, the policy supports an **optional** per-joint top-K constraint. We compute a per-joint softmax over views, add Gumbel noise, and select the top `K` views. The straight-through estimator passes the gradient through the hard `argmax` by treating the forward pass as hard masking and the backward pass as the softmax probabilities:

```
π = softmax_v((logits + gumbel_noise) / τ)          # forward: one-hot top-K via straight-through
s = π_forward.detach() + (hard_top_k_mask - hard_top_k_mask.detach())
```

If `v54_lvsp_top_k == 0`, the continuous sigmoid is used and no hard top-K is enforced. The selected weights are multiplied with the existing triangulation weights:

```
weights_selected = weights_in * (s * (1 - min_weight) + min_weight)
```

where `min_weight > 0` guarantees every visible view still contributes at least a small amount (warm-start / stability).

### 2.5 Warm-Start / Identity at Initialization

To satisfy the warm-start requirement, all final linear layers that produce `logits` are initialized to **zero**, and the temperature is set moderately high (`τ = 1.0`) at the start. Consequently `σ(0) = 0.5`, so `s ≈ 0.5` and `weights_selected  weights_in * 0.5 * (1 - min_weight) + min_weight)`. A learned scalar rescale after the policy restores the magnitude, making the network initially equivalent to the v52/v53 baseline. An explicit `v54_lvsp_identity_init=True` flag controls this behavior.

### 2.6 Losses

The module contributes an auxiliary loss `L_policy` with three terms:

1. **Reprojection-guided reward** (maximized, implemented as negative loss):
   ```
   L_reproj = mean( s * r )               # encourage selecting low-residual views
   ```
2. **Sparsity / entropy bonus**:
   ```
   L_entropy = -entropy(s)                # discourage trivial uniform selection
   ```
3. **MPJPE shaping loss** (only when hard top-K is used):
   ```
   L_pose = MPJPE( triangulate(points_2d, weights_selected), gt_3d )
   ```

The combined auxiliary loss is `L = v54_lvsp_loss_weight * (L_reproj - v54_lvsp_entropy_weight * L_entropy) + L_pose`.

## 3. Configuration Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v54_learned_view_selection_policy` | `bool` | `False` | Master toggle. |
| `v54_lvsp_hidden` | `int` | `64` | Hidden dimension of the policy MLP. |
| `v54_lvsp_n_layers` | `int` | `2` | Number of MLP layers. |
| `v54_lvsp_top_k` | `int` | `0` | If `> 0`, enforce hard top-K selection per joint. |
| `v54_lvsp_temperature` | `float` | `1.0` | Gumbel-softmax / sigmoid temperature. |
| `v54_lvsp_hard` | `bool` | `True` | Use straight-through hard selection; otherwise sigmoid. |
| `v54_lvsp_min_weight` | `float` | `0.05` | Minimum residual weight for any selected view. |
| `v54_lvsp_entropy_weight` | `float` | `0.01` | Coefficient of the entropy bonus. |
| `v54_lvsp_loss_weight` | `float` | `0.01` | Weight of the policy auxiliary loss. |
| `v54_lvsp_identity_init` | `bool` | `True` | Zero-initialize policy logits for identity warm-start. |
| `v54_lvsp_use_reproj_residual` | `bool` | `True` | Include reprojection residual in the state vector. |

## 4. Expected MPJPE Impact

- **Smoke (RTX 4090, 2 epochs, 100 samples):** expect `val_MPJPE@full` to stay within **±0.5 mm** of the v53 baseline thanks to identity init; if training converges, a **0.5–1.0 mm** improvement is realistic.  
- **Full A800 run:** a stable v54 policy should reduce `val_MPJPE@full` by **1.0–2.0 mm** over v53 by suppressing residual-heavy views.  
- **Sparse-view metrics (`MPJPE@2`, `MPJPE@3`):** largest gains, estimated **2–5 mm**, because selecting the correct subset matters more when few views are available.  
- **No new data loader** is required; the module consumes the same inputs as v52/v53.

## 5. Risks

1. **Gradient instability through the straight-through top-K estimator.**  
   Mitigation: keep a continuous sigmoid warm-up path (`v54_lvsp_hard=False`) for the first epoch, then switch to hard selection.  
2. **Over-selection: the policy drops too many views and under-constrains triangulation.**  
   Mitigation: enforce `min_weight > 0` and an entropy bonus, and clamp the effective selection to at least `max(2, min_visible_views)` views per joint.  
3. **Noisy reprojection residual at early training misleads the policy.**  
   Mitigation: use `pred_3d` from the v53 baseline (detached) as the residual reference, and schedule the policy loss weight to ramp up after epoch 1.  
4. **Inference-time top-K selection is sensitive to `K` and may hurt rare-camera setups.**  
   Mitigation: keep a soft fallback mode at inference and expose `v54_lvsp_top_k` as a tunable hyper-parameter per dataset.  
5. **Interaction with v51/v50 auxiliary heads could compound losses.**  
   Mitigation: gate the policy loss with `v54_lvsp_loss_weight` and validate on the v51-CDSVR baseline smoke before a full run.

## 6. Implementation Plan

1. Create `motionflow_mv/fusion/learned_view_selection_policy_v54.py` with class `LearnedViewSelectionPolicyV54` implementing the state encoder, MLP, and differentiable selection.
2. Wire the module into `OmniMultiViewFusionV5.__init__` and `forward`, placed immediately after `v53_physical_space_calibration_v53`, and multiply the returned selection weights with the existing triangulation weights.
3. Add the configuration flags to `OmniMultiViewFusionV5.__init__`, `configs/benchmark_v54_lvsp_smoke.yaml`, and the A800 launcher.
4. Add a smoke test script `scripts/run_v54_lvsp_smoke_local_4090.sh` that runs a short 2-epoch validation and checks identity-at-init (loading a v53 checkpoint with v54 enabled should change `val_MPJPE@full` by ≤ 0.1 mm).
5. Update `AGENTS.md` status table and create `docs/swarm_iter28/reports/agent_learned_view_selection_policy_risks.md` with the risk register below.
