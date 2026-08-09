# v53 Learned View Selection Policy (LVSP)

## Motivation

v52 Uncertainty-Weighted Triangulation (UWT) learns per-view/per-joint precision
weights, but it still fuses evidence from every available view.  In practice, not
all views are equally useful for every joint: a camera may be occluded,
foreshortened, or poorly calibrated, and including it hurts triangulation.  v53
adds a **differentiable view-selection policy** on top of v52 that explicitly
chooses which views participate in the final weighted DLT.  The goal is to
improve sparse-view robustness, reduce the influence of outlier cameras, and
provide an interpretable selection mask that can be inspected at inference time.

## Architecture

The module is inserted after `UncertaintyWeightedTriangulationV52` in
`OmniMultiViewFusionV5`.  It receives the same 2-D keypoints, camera parameters,
and feature tokens as v52, plus the v52-predicted precision weights
`w^{(52)} \in [0,1]^{B,T,V,J}` and the current 3-D estimate
`X^{(0)} \in \mathbb{R}^{B,T,J,3}`.

```
features  (B,T,V,J,d)  ──┐
points_2d (B,T,V,J,2)  ──┤
K,R,t     (B,T,V,3,3/3)──┼──► LearnedViewSelectionPolicyV53 ──► selected weights  (B,T,V,J)
w^52      (B,T,V,J)    ──┤                              ──► refined 3-D pose X^(1)
X^(0)     (B,T,J,3)    ──┘
```

**Policy score network.**  Per-view/per-joint features are built from:

1.  **Feature statistics**: raw token `f_{t,v,j}`, its mean and std over views.
2.  **Geometry bias**: reprojection residual of `X^{(0)}` in each view and the
    v52 precision `w^{(52)}_{t,v,j}`.
3.  **Camera meta-bias**: ray angle and baseline-to-scene ratio (computed from
    `K,R,t`) encoded by a tiny MLP.

The score network is a -layer MLP with hidden size `v53_lvsp_hidden`:

```
s_{t,v,j} = Linear( ReLU( MLP( [f, mean(f), std(f), r_{t,v,j}, log r_{t,v,j}, w^{(52)}, \phi(K,R,t)] ) ) )
```

`r_{t,v,j}` is the reprojection residual norm of `X^{(0)}` in view `v`.

**Differentiable view selection.**  A Gumbel-softmax straight-through estimator
converts per-view logits `s_{t,v,j}` into a soft selection mask
`\alpha_{t,v,j} \in [0,1]`:

```
g_v ~ Gumbel(0, 1)
\alpha_{t,v,j} = sigmoid( (s_{t,v,j} + g_v) / \tau )   # \tau = v53_lvsp_gumbel_temperature
```

During the **forward pass** the soft mask is used; during the **backward pass**
the straight-through estimator treats `\alpha` as the hard argmax mask.  A
minimum-view constraint is enforced by keeping the top-`k` views per joint,
where `k = max(v53_lvsp_min_views, ceil(selection_rate * V))`, before adding
the Gumbel noise.

**Final triangulation weights.**  The selected view weights multiply the v52
precision weights:

```
w^{(53)}_{t,v,j} = w^{(52)}_{t,v,j} \cdot \alpha_{t,v,j}
```

A weighted DLT is run with `w^{(53)}` to obtain `X^{(1)}`.  A gated residual MLP
refines `X^{(1)}` relative to `X^{(0)}`:

```
X^{(1)} = X^{(0)} + g \cdot MLP_{res}(X^{(1)} - X^{(0)}),   g = sigmoid(v53_lvsp_identity_gate_init)
```

At initialization `g = 0`, so `X^{(1)} = X^{(0)}`.

## Inputs and Outputs

| Symbol | Shape | Description |
|--------|-------|-------------|
| `features` | `(B,T,V,J,d)` | multi-view feature tokens |
| `points_2d` | `(B,T,V,J,2)` | detected 2-D keypoints |
| `K` | `(B,T,V,3,3)` | intrinsics |
| `R` | `(B,T,V,3,3)` | rotations |
| `t` | `(B,T,V,3)` | translations |
| `pred_3d_v52` | `(B,T,J,3)` | 3-D pose from v52 UWT |
| `w52` | `(B,T,V,J)` | v52 precision weights |
| `view_mask` | `(B,T,V)` | optional bool mask for missing views |

Outputs:

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B,T,J,3)` | refined 3-D pose |
| `alpha` | `(B,T,V,J)` | soft view-selection mask |
| `lvsp_loss` | `()` | auxiliary loss |

## Config Flags

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `use_v53_learned_view_selection_policy` | `bool` | `False` | enable the module |
| `v53_lvsp_hidden` | `int` | `64` | MLP hidden dimension |
| `v53_lvsp_n_layers` | `int` | `2` | score MLP layers |
| `v53_lvsp_gumbel_temperature` | `float` | `0.5` | Gumbel-softmax temperature |
| `v53_lvsp_straight_through` | `bool` | `True` | use straight-through estimator |
| `v53_lvsp_min_views` | `int` | `2` | minimum views kept per joint |
| `v53_lvsp_max_views` | `int` | `V` | maximum views kept per joint |
| `v53_lvsp_sparsity_weight` | `float` | `0.01` | L1 penalty on `\alpha` |
| `v53_lvsp_stability_weight` | `float` | `0.005` | reward for well-conditioned DLT |
| `v53_lvsp_identity_gate_init` | `float` | `-6.0` | initial logit of residual gate (≈0 effect) |
| `v53_lvsp_use_v52_weights_as_prior` | `bool` | `True` | concatenate `w^{(52)}` into score features |
| `v53_lvsp_loss_weight` | `float` | `0.01` | multiplier of the auxiliary loss |
| `v53_lvsp_warmup_epochs` | `int` | `0` | epochs before the loss is added |

## Expected MPJPE Impact

- **Full 4-view setting**: small gain, ~0.2–0.8 mm, because v52 already weights
  views softly.
- **Sparse-view setting (2–3 views)**: larger gain, ~0.8–2.0 mm, by discarding
  the worst view and stabilizing DLT.
- **Cross-domain transfer**: improved robustness when camera layouts differ
  between training and validation.

## Risks

1.  **Gumbel noise instability** at very low temperatures can make selection
    non-differentiable or brittle.
2.  **Over-selection**: the policy may collapse to always selecting the same
    subset, losing the benefit of multi-view fusion.
3.  **Interaction with v52**: if v52 already down-weights bad views, v53 may
    duplicate its effect without additional gain.

## 5-Step Implementation Plan

1.  **Module skeleton** (`motionflow_mv/fusion/learned_view_selection_policy_v53.py`): define
    `LearnedViewSelectionPolicyV53` with the score MLP, Gumbel-sample helper,
    and identity-initialized residual gate.
2.  **Integration**: import and instantiate the module in
    `OmniMultiViewFusionV5.__init__` behind `use_v53_learned_view_selection_policy`;
    call it inside the `forward` after the v52 UWT block.
3.  **Loss wiring**: add the sparsity and DLT-stability auxiliary losses and
    combine them with `v53_lvsp_loss_weight`, respecting the warmup epoch flag.
4.  **Smoke config**: create `configs/benchmark_v53_lvsp_smoke.yaml` with
    `use_v52_uncertainty_weighted_triangulation: True` and
    `use_v53_learned_view_selection_policy: True`.
5.  **Validation**: run the smoke script on RTX 4090, verify identity-at-init
    (loading a v52 checkpoint with v53 enabled changes `val_MPJPE` by
    `< 0.1 mm`), then evaluate `MPJPE@2` and `MPJPE@3` on the variable-view
    benchmark.
