# v55 Test-Time Self-Refinement (TTSR)

**Status:** proposal / design-only  
**Tracking issue:** #208 (proposed)  
**Base branch:** `v55-ttsr`  
**Depends on:** v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT, v53-PSC, v54-PSC-v2

## 1. Module name and one-line purpose

**Module:** `motionflow_mv/fusion/test_time_self_refinement_v55.py`  
**Class:** `TestTimeSelfRefinementV55`

Iteratively refine the v54 PSC-v2 pose at test/train time by feeding back reprojection residuals and physical-space cues (floor distance, bone-length residual, joint velocity) into a lightweight learned corrector, producing a better-calibrated 3-D pose without changing the v54 checkpoint at initialization.

## 2. Forward-pass placement

```textnpoints_2d, confidences, K, R, t    ↓v25/v45 geometry fusion → pred_3d_init, weights_init    ↓v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss    ↓v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, psc_loss    v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_loss    ↓v55 TestTimeSelfRefinementV55    (consumes pred_3d_psc2, uwt_weights, points_2d, K, R, t, view_mask, domain_id)    → pred_3d_ttsr, ttsr_loss    ↓final residual MLP / v47/v49 temporal / v50 SEFH heads
```

v55 sits **after** the v54 PSC-v2 block and **before** any final residual MLP or temporal/SEFH heads, so downstream modules receive a pose that has been locally physically calibrated by v54 and then iteratively self-corrected by v55.

## 3. Inputs, outputs, and shapes

| Tensor | Shape | Description |
|---|---|---|
| `pred_3d_psc2` | `(B, T, J, 3)` | Pose after v54 PSC-v2. |
| `uwt_weights` | `(B, T, J, V)` or `(B, T, J)` | v52 per-(view,joint) triangulation weights or pooled reliability. |
| `points_2d` | `(B, T, J, V, 2)` | Input 2-D keypoints. |
| `confidences` | `(B, T, J, V)` | Input 2-D confidence scores. |
| `K` | `(B, V, 3, 3)` | Camera intrinsics. |
| `R` | `(B, V, 3, 3)` | Camera rotations. |
| `t` | `(B, V, 3, 1)` | Camera translations. |
| `view_mask` | `(B, T, V)` | Valid-view mask. |
| `domain_id` | `(B,)` or int | Domain index for per-domain bone/floor priors. |
| `bone_scale_v2` | `(B, T, B)` | v54 per-bone canonical scale (optional, used as prior). |
| `floor_height_v2` | `(B, T)` or scalar | v54 estimated floor height (optional). |

**Outputs:**

| Tensor | Shape | Description |
|---|---|---|
| `pred_3d_ttsr` | `(B, T, J, 3)` | Refined pose. |
| `ttsr_loss` | scalar | Auxiliary loss summed over refinement steps. |

## 4. Architecture

### 4.1 Core idea

Run a small, fixed number of refinement steps `N` (default `3`). At each step `i`, compute feedback signals from the current pose estimate `x_i`, map them to a per-joint correction `Δx_i`, and gate the update with a near-zero initialized scalar gate so that `x_{i+1} = x_i + σ(gate) · Δx_i`. The final output `x_N` is returned as `pred_3d_ttsr`.

### 4.2 Feedback features per joint

For each joint `j` at each step `i`:

1. **Reprojection residual:** project `x_i[j]` into each view, measure 2-D distance to detected keypoint, weighted by `uwt_weights` and confidence. Output: `(B, T, J, 1)`.
2. **Bone-length residual:** compare bone lengths to v54 canonical bone scale prior; output signed residual `(B, T, B, 1)` mapped to child joint.
3. **Floor residual:** signed distance of foot joints to `floor_height_v2`; output `(B, T, J, 1)` (zeros for non-foot joints).
4. **Velocity residual:** finite-difference velocity of `x_i` along `T`; encourages temporal smoothness.
5. **Uncertainty feature:** `−log(uwt_weights)` pooled over views, giving a per-joint confidence feature.

These features are concatenated per joint into a vector of dimension `F` (≈ 12) and fed into the correction head.

### 4.3 Correction head

```textnfeedback_features (B, T, J, F)
    ↓
LayerNorm + Linear(F → hidden) + ReLU
    ↓
Repeat per refinement step:  N × [Linear(hidden → hidden) + ReLU]
    ↓
Linear(hidden → 3)  → per-joint correction Δx  (zero-initialized)
```

- `hidden = v55_ttsr_hidden` (default `64`).
- The final `Linear(hidden → 3)` is zero-initialized so that `Δx_i = 0` at initialization.

### 4.4 Residual gate

```python
import torch.nn as nn

gate_logit = nn.Parameter(torch.tensor(v55_ttsr_residual_gate_init))  # default -6.0
gate = torch.sigmoid(gate_logit)  # ≈ 0.0025 at init
x_{i+1} = x_i + gate * Δx_i
```

At initialization `gate ≈ 0`, so `x_N = x_0` and a v54 checkpoint loaded with v55 enabled remains unchanged.

### 4.5 Losses

Each loss is computed on the **final** refined pose `x_N` and, optionally for stability, averaged over intermediate steps. All loss weights are defaulted conservatively so that the v54 baseline is preserved at initialization.

| Loss | Symbol | Description | Weight |
|---|---|---|---|
| Reprojection consistency | `L_reproj` | 2-D reprojection error of `x_N`, weighted by UWT weights and confidence. | `v55_ttsr_reproj_weight = 0.1` |
| Bone-length consistency | `L_bone` | Soft squared residual to v54 canonical bone scale. | `v55_ttsr_bone_weight = 0.05` |
| Floor consistency | `L_floor` | Soft penalty for foot joints below estimated floor. | `v55_ttsr_floor_weight = 0.01` |
| Temporal smoothness | `L_temporal` | Finite-difference velocity of `x_N`. | `v55_ttsr_temporal_weight = 0.01` |
| Correction regularizer | `L_corr` | L2 norm of the correction `Δx_N`, prevents over-shooting. | `v55_ttsr_corr_weight = 0.01` |

Total:

```
ttsr_loss = v55_ttsr_loss_weight * (
    v55_ttsr_reproj_weight * L_reproj +
    v55_ttsr_bone_weight * L_bone +
    v55_ttsr_floor_weight * L_floor +
    v55_ttsr_temporal_weight * L_temporal +
    v55_ttsr_corr_weight * L_corr
)
```

All losses are computed with **detached** intermediate supervision: the network is trained to produce a refined pose, but the iterative loop is unrolled only once during backpropagation to keep memory low. In inference, the same loop runs for `v55_ttsr_num_steps` steps.

## 5. Expected MPJPE impact and main risks

| View setting | Expected MPJPE change |
|---|---|
| Full views | `−0.8 to −2.0 mm` |
| Sparse `@2/3` | `−1.5 to −3.5 mm` (larger because reprojection feedback redistributes uncertainty) |
| 3DPW actual / cross-domain | `−2.0 to −4.0 mm` |

**Main risks and mitigations:**

| Risk | Symptom | Mitigation |
|---|---|---|
| **Iterative drift / overshoot** | MPJPE rises after enabling v55; corrections grow with each step. | Gate logit `−6.0` and correction L2 regularizer keep updates tiny at init; clamp per-step correction magnitude to `0.1 m` during first epoch. |
| **Reprojection loss dominates, bones/floor ignored** | Bone lengths worsen, feet clip floor. | Use loss weights that are ≤ v54 weights; optionally freeze v55 for first warmup epoch (`v55_ttsr_warmup_epochs`). |
| **Memory blow-up from unrolled loop** | OOM on RTX 4090 during smoke. | Fix `v55_ttsr_num_steps=3`; share the correction-head weights across steps; backpropagate only through the final step with intermediate steps detached. |
| **Test-time / train-time mismatch** | Smoke val improves but full-run val regresses. | Train and eval use the same fixed `num_steps`; no adaptive stopping. |
| **v54 identity-at-init broken** | v54 checkpoint changes by `>0.1 mm` when v55 enabled. | Zero-initialize final correction `Linear`, gate logit `−6.0`, and unit-test `||pred_ttsr − pred_psc2||_∞ < 1e-4`. |

## 6. Smoke acceptance criteria

- `val_MPJPE@full` is within `1 mm` of the v54-PSC-v2 baseline on the same smoke config.
- No NaN, Inf, or OOM through at least one full epoch.
- Identity-at-init: loading a v54 checkpoint with v55 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
- Reprojection sanity: mean 2-D reprojection error of `pred_3d_ttsr` is not larger than that of `pred_3d_psc2` in at least `90%` of frames.
- Bone-scale sanity: per-bone ratios `exp(s_b)` stay in `[0.5, 2.0]` for at least `95%` of bones.
- `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.

## 7. Required new files and files to modify

**New files:**

- `motionflow_mv/fusion/test_time_self_refinement_v55.py` — `TestTimeSelfRefinementV55` module.
- `configs/benchmark_v55_ttsr_smoke.yaml` — smoke config copied from v54 PSC-v2 smoke with v55 flags enabled.
- `scripts/run_v55_ttsr_smoke_local_4090.sh` — smoke launch script that warm-starts from the best available v54 checkpoint.
- `tests/test_test_time_self_refinement_v55.py` — unit tests for identity-at-init, gradient flow, reprojection sanity, and iterative stability.

**Files to modify:**

- `motionflow_mv/fusion/omniview_fusion_v5.py` — add constructor flag `use_v55_test_time_self_refinement`, instantiate `TestTimeSelfRefinementV55` when enabled, insert call after the v54 PSC-v2 block, and add `ttsr_loss` to `epi_loss`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — aggregate `loss_dict["v55_ttsr"]` with `v55_ttsr_loss_weight` and honor `v55_ttsr_warmup_epochs`.
- `scripts/launch_v33_a800_queue.py` — add A800 full-run entry `v55_test_time_self_refinement_on_v54`.

## Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_test_time_self_refinement` | bool | `False` | Master toggle |
| `v55_ttsr_hidden` | int | `64` | Correction-head hidden dimension |
| `v55_ttsr_n_layers` | int | `2` | Correction-head MLP depth |
| `v55_ttsr_num_steps` | int | `3` | Number of self-refinement iterations |
| `v55_ttsr_identity_init` | bool | `True` | Zero-initialize final correction layer and gate |
| `v55_ttsr_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init |
| `v55_ttsr_loss_weight` | float | `1.0` | Multiplier on total `L_ttsr` |
| `v55_ttsr_reproj_weight` | float | `0.1` | Weight of reprojection consistency term |
| `v55_ttsr_bone_weight` | float | `0.05` | Weight of bone-length consistency term |
| `v55_ttsr_floor_weight` | float | `0.01` | Weight of floor consistency term |
| `v55_ttsr_temporal_weight` | float | `0.01` | Weight of temporal smoothness term |
| `v55_ttsr_corr_weight` | float | `0.01` | Weight of correction L2 regularizer |
| `v55_ttsr_max_corr` | float | `0.1` | Max per-step correction magnitude during first epoch |
| `v55_ttsr_warmup_epochs` | int | `0` | Epochs before `ttsr_loss` contributes to total loss |

## Notes

- Do not implement any code; this proposal is for design review and agent assignment only.
- The module is intentionally kept small (`hidden=64`, `num_steps=3`) to preserve the low-risk identity-at-init property.
- If smoke shows that iterative refinement conflicts with v54 PSC-v2 losses, disable overlapping physical loss terms in v55 and rely on reprojection/correction regularization only.
- Keep the module optional: `OmniMultiViewFusionV5` must still load and run when `use_v55_test_time_self_refinement=False`.
