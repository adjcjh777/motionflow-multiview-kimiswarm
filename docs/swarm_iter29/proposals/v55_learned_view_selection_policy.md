# v55 Learned View Selection Policy (LVSP)

## 1. Module name and one-line purpose

- **Module:** `LearnedViewSelectionPolicyV55` → `motionflow_mv/fusion/learned_view_selection_policy_v55.py`
- **Class:** `LearnedViewSelectionPolicyV55`
- **One-line purpose:** A differentiable, physically-grounded view-subset selection policy that reweights camera views before a final gated re-triangulation, reducing the impact of outlier and missing views after the v54 physical-space calibration stage.

## 2. Where it sits in the OmniMultiViewFusionV5 forward pass

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss
    ↓
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, psc_loss
    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_loss
    ↓
v55 LearnedViewSelectionPolicyV55
    (consumes pred_3d_psc2, uwt_weights, points_2d, K, R, t, view_mask, domain_id)
    → pred_3d_lvsp, lvsp_loss, view_selection_score
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

LVSP is placed **after v54 PSC-v2** so it can use the physically calibrated 3D pose as reliable context for judging which views are trustworthy. It does not replace v54; it adds a view-selection stage that refines how the calibrated pose is fused from the original multi-view observations.

## 3. Inputs, outputs, and shapes

| Symbol | Tensor | Shape | Description |
|--------|--------|-------|-------------|
| `pred_3d_psc2` | float32 | `(B, T, J, 3)` | Physically calibrated 3D pose from v54. |
| `uwt_weights` | float32 | `(B, T, J, V)` | v52 per-view joint uncertainty weights. |
| `points_2d` | float32 | `(B, T, J, V, 2)` | Input 2D keypoints. |
| `K` | float32 | `(B, T, V, 3, 3)` | Camera intrinsics. |
| `R` | float32 | `(B, T, V, 3, 3)` | Camera rotations. |
| `t` | float32 | `(B, T, V, 3)` | Camera translations. |
| `view_mask` | bool / float32 | `(B, T, V)` | Valid-view mask. |
| `domain_id` | int64 | `(B,)` | Domain label for per-domain selection bias. |

**Outputs:**

| Symbol | Tensor | Shape | Description |
|--------|--------|-------|-------------|
| `pred_3d_lvsp` | float32 | `(B, T, J, 3)` | Final triangulated pose after view selection. |
| `lvsp_loss` | float32 | scalar | Auxiliary loss encouraging sensible selection. |
| `view_selection_score` | float32 | `(B, T, V)` | Per-view selection probability in `[0, 1]`. |

## 4. Architecture: layers, heads, losses, identity-at-init mechanism

### 4.1 Selection policy

1. **Reprojection feature.** Reproject `pred_3d_psc2` into every view using `K, R, t`. Compute per-joint reprojection error
   ```
   e_{b,t,j,v} = ||Π_v(p_{b,t,j}) - x_{b,t,j,v}||_2  *  c_{b,t,j,v}
   ```
   where `c` is the input confidence.

2. **Per-view feature vector.** Pool joint errors into a per-view feature
   ```
   f_v = concat( mean_j(e_v), std_j(e_v), max_j(e_v), mean_j(uwt_weights_v) )
   ```
   shape `(B, T, V, D)` with `D = 4` (or larger if domain embedding is appended).

3. **Selection MLP.** A -layer MLP maps `f_v` to a logit
   ```
   logit_v = MLP(f_v)  ∈ (B, T, V)
   ```
   Final layer is **zero-initialized** so logits are zero at initialization.

4. **Selection probability.**
   ```
   π_v = sigmoid( logit_v / τ )
   ```
   where `τ = v55_lvsp_temperature` starts high (soft uniform selection) and is annealed during training.

### 4.2 Re-triangulation and gated residual

1. **Refined view weights.** Multiply the v52 UWT weights by the selection score:
   ```
   w'_{b,t,j,v} = uwt_weights_{b,t,j,v} * π_{b,t,v}
   ```

2. **Lightweight DLT re-triangulation.** Run a small v52-style DLT on `w'` to obtain `pred_3d_reweighted`.

3. **Gated combination.**
   ```
   pred_3d_lvsp = pred_3d_psc2 + σ(gate) * (pred_3d_reweighted - pred_3d_psc2)
   ```
   where the gate logit is initialized to `v55_lvsp_residual_gate_init = -6.0` so `σ(gate) ≈ 0.0025` at start.

### 4.3 Losses

All losses are weighted by `v55_lvsp_loss_weight` and are only active after `v55_lvsp_warmup_epochs`.

| Loss | Formula | Purpose |
|------|---------|---------|
| `L_reproj` | Reprojection loss on the selected (top-K or weighted) views. | Train the policy to pick views that reproject well. |
| `L_entropy` | `− Σ_v [π_v log π_v + (1−π_v) log(1−π_v)]` per valid view. | Prevent collapse to all-0 or all-1; encourage decision. |
| `L_consistency` | Variance of triangulated pose across different top-K subsets. | Selected subsets should agree on the 3D pose. |
| `L_sparsity` | `mean(π_v)` regularizer (optional). | Encourage selecting fewer views if accuracy permits. |

```
lvsp_loss = v55_lvsp_loss_weight * (
              v55_lvsp_reproj_weight   * L_reproj +
              v55_lvsp_entropy_weight  * L_entropy +
              v55_lvsp_consistency_weight * L_consistency +
              v55_lvsp_sparsity_weight * L_sparsity
            )
```

### 4.4 Identity-at-init mechanism

- **Zero-initialized final MLP layer:** logits are zero at init, so `π_v = 0.5` for all views (uniform weighting after multiplication with UWT weights).
- **High-temperature annealing:** `v55_lvsp_temperature` starts at `5.0` and decays to `1.0`, keeping selection soft during warm-up.
- **Residual gate:** `v55_lvsp_residual_gate_init = -6.0`, so the gated re-triangulation term is effectively zero.
- **Warm-up guard:** `v55_lvsp_warmup_epochs` epochs during which `lvsp_loss` is not added to the total loss.

At initialization, the module therefore passes `pred_3d_psc2` through almost unchanged, satisfying the identity-at-init requirement.

## 5. Expected MPJPE impact (full/sparse views) and main risks

| View setting | Expected impact | Rationale |
|--------------|-----------------|-----------|
| Full views  | `−0.8` to `−1.5 mm` | Downweights occasional outlier views; marginal on clean full-view data. |
| `@4` views  | `−1.0` to `−2.0 mm` | Better rejection of a single bad view. |
| `@3` views  | `−2` to `−4 mm` | Selection policy can drop the weakest view and rely on the best two. |
| `@2` views  | `−2` to `−5 mm` | Correctly identifying the better of two views has large sparse-view payoff. |

### Main risks and mitigations

| Risk | Symptom | Mitigation |
|------|---------|------------|
| **Policy collapses** | All `π_v` near 0 or 1 regardless of quality; loss plateaus. | Entropy regularization, temperature annealing, and sparsity weight clamping. |
| **Re-triangulation instability** | NaN/Inf when selected views are too few. | Enforce `min_views >= 2` via mask; use weighted DLT rather than hard top-K. |
| **Overfit to training camera setups** | Validation gain smaller than training gain. | Generalize selection MLP with dropout and per-domain input embedding. |
| **Identity-at-init regression** | v54 checkpoint changes by `>0.1 mm` when v55 enabled. | Unit test asserting `||pred_3d_lvsp − pred_3d_psc2||_∞ < 1e-4`; gate logit `−6.0`. |
| **Slow training / OOM** | Extra re-triangulation is expensive. | Cache v52 DLT matrices; use `v55_lvsp_topk_mode='soft'` by default; keep MLP hidden dim `64`. |

## 6. Smoke acceptance criteria

Run on the local RTX 4090 using `configs/benchmark_v55_learned_view_selection_policy_smoke.yaml`, warm-started from the best available v54 checkpoint.

- `val_MPJPE@full` is within `1 mm` of the v54-PSC-v2 baseline on the same smoke config.
- `val_MPJPE@2` and `val_MPJPE@3` are not worse than the v54 baseline.
- No NaN, Inf, or OOM through at least one full epoch.
- Identity-at-init: loading a v54 checkpoint with v55 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
- Selection sanity: mean selection entropy per view is in `(0.1, 0.9)` bits in at least `80%` of frames.
- Coverage sanity: at least `min_views` selected in every frame; no frame drops below `min_views` due to policy.
- Loss sanity: `lvsp_loss` is finite and decreasing or stable after the first 100 steps.

## 7. Required new files and files to modify

### New files

- `motionflow_mv/fusion/learned_view_selection_policy_v55.py` — `LearnedViewSelectionPolicyV55` module.
- `configs/benchmark_v55_learned_view_selection_policy_smoke.yaml` — smoke config, copied from the v54 smoke with v55 flags enabled.
- `scripts/run_v55_learned_view_selection_policy_smoke_local_4090.sh` — smoke launch script, warm-starting from the best v54 checkpoint.
- `tests/test_learned_view_selection_policy_v55.py` — unit tests for identity-at-init, selection shape/sanity, gradient flow, and min-views enforcement.

### Files to modify

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add constructor flag `use_v55_learned_view_selection_policy`.
  - Instantiate `LearnedViewSelectionPolicyV55` when enabled.
  - Insert the call in `forward` immediately after the v54 PSC-v2 block and before the final residual MLP / v47/v49 temporal / v50 SEFH heads.
  - Add `lvsp_loss` to the `epi_loss` dictionary with key `v55_lvsp`.

- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Forward `domain_id` to the model (already done for v54).
  - Aggregate `loss_dict["v55_lvsp"]` into the total loss with weight `v55_lvsp_loss_weight` only after `v55_lvsp_warmup_epochs`.

- `scripts/launch_v33_a800_queue.py`
  - Add a full-run entry `v55_learned_view_selection_policy_on_v54` on top of the best v54 checkpoint.

### Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_learned_view_selection_policy` | bool | `False` | Master toggle |
| `v55_lvsp_hidden` | int | `64` | Selection MLP hidden dim |
| `v55_lvsp_n_layers` | int | `2` | Selection MLP depth |
| `v55_lvsp_num_domains` | int | `8` | Number of domains for per-domain input embedding |
| `v55_lvsp_temperature` | float | `5.0` | Initial softmax temperature (annealed to `1.0`) |
| `v55_lvsp_min_views` | int | `2` | Minimum views the policy is allowed to keep |
| `v55_lvsp_topk_mode` | str | `"soft"` | `"soft"` (weighted) or `"hard"` (straight-through top-K) |
| `v55_lvsp_topk` | int | `0` | If `>0`, use hard top-K selection (riskier; default 0 disables it) |
| `v55_lvsp_identity_init` | bool | `True` | Zero-initialize final MLP layer and gate |
| `v55_lvsp_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init |
| `v55_lvsp_loss_weight` | float | `1.0` | Multiplier on total `L_lvsp` |
| `v55_lvsp_reproj_weight` | float | `1.0` | Weight of `L_reproj` |
| `v55_lvsp_entropy_weight` | float | `0.1` | Weight of `L_entropy` |
| `v55_lvsp_consistency_weight` | float | `0.1` | Weight of `L_consistency` |
| `v55_lvsp_sparsity_weight` | float | `0.01` | Weight of `L_sparsity` |
| `v55_lvsp_warmup_epochs` | int | `0` | Epochs before `lvsp_loss` contributes |

## Notes

- Do not implement any code; this proposal is for design review and agent assignment only.
- Keep the module optional: `OmniMultiViewFusionV5` must still load and run when `use_v55_learned_view_selection_policy=False`.
- If the smoke shows hard top-K is unstable, keep `v55_lvsp_topk_mode="soft"` as the default and deprecate the hard option.
- Strong synergy with v52 UWT: the policy operates on top of existing uncertainty weights, so it is inherently low-risk and preserves the warm-start checkpoint.
