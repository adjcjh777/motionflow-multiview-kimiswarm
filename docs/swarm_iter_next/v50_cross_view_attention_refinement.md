# v50 Geometry-Reliability Cross-View Attention Refinement (CVar-GR)

## Module name

`CrossViewAttentionRefinementV50` — a residual cross-view attention block inserted into the v25/v45 geometry-fusion encoder before triangulation and the v46–v48 stack.

## Architecture description

The block refines per-view feature tokens with one transformer-style cross-view attention layer. Attention scores are biased by pairwise epipolar-line distances and ray-intersection depths, favoring geometrically consistent view pairs. They are further gated by per-view, per-joint reliability weights (reused from v37/v39 or the v50 Self-Evolution Feedback Head) to down-weight noisy or occluded views. A gated residual connection with near-zero initialization makes the block identity at startup, preventing regression of an already-strong baseline. Refined tokens then flow into v45 adaptive triangulation and downstream v46–v48 heads unchanged.

## New config flags and defaults

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v50_cross_view_attention_refinement` | bool | `False` | Master switch. |
| `v50_cvar_hidden` | int | `64` | Hidden dimension. |
| `v50_cvar_num_heads` | int | `4` | Number of attention heads. |
| `v50_cvar_num_layers` | int | `1` | Number of refinement layers. |
| `v50_cvar_geometry_bias` | str | `"epipolar_ray"` | `"none"`, `"epipolar"`, `"ray"`, or `"epipolar_ray"`. |
| `v50_cvar_use_reliability_gate` | bool | `True` | Modulate attention by per-view reliability. |
| `v50_cvar_attention_dropout` | float | `0.1` | Dropout on attention weights. |
| `v50_cvar_residual_scale` | float | `0.1` | Learned residual gate scale, initialized near zero. |
| `v50_cvar_identity_init` | bool | `True` | Identity initialization. |

## Loss term

Cross-view geometric consistency loss on the refined tokens:

```text
L_cvar = loss.v50_cvar_weight * (1 / VJ) Σ_v,j || Π_v(P_3D^j) - k_2D^{v,j} ||_2
```

| Setting | Default |
|---------|---------|
| `loss.v50_cvar_weight` | `0.01` |
| `loss.v50_cvar_clip` | `200.0` px |

The small weight and pixel clipping keep the module from overriding the triangulation loss during warm-up.

## Evaluation metric

Primary: `val_MPJPE@full` and `MPJPE@k` for `k = 2, 3, 4` via the canonical v49 protocol. Secondary: per-view attention entropy and Spearman correlation between refined attention weights and reprojection residuals (target `> 0.25`). Tertiary: 4090 smoke latency to confirm the extra layer is cheap.

## Expected MPJPE impact

From the v46-SVG smoke baseline `val_MPJPE@full = 32.97 mm`, CVar-GR should improve sparse-view robustness where geometric inconsistency hurts most: target `MPJPE@2 -2 to -3 mm`, `MPJPE@3 -1 to -2 mm`, and `MPJPE@full` within `0.5 mm` of v46. Gains on full views are expected to be modest (≈ -0.5 mm) because full-view triangulation already exploits geometry.

## Main risk / mitigations

| Risk | Mitigation |
|------|-----------|
| Geometry bias conflicts with learned attention, over-smoothing features. | Identity-at-init; learn the residual gate; ablate `v50_cvar_geometry_bias = "none"`. |
| Reliability gate collapses to uniform weights. | Initialize near-one, add small entropy regularizer, monitor Spearman. |
| Memory/latency breaks the A800 batch budget. | One layer, dim 64; profile on 4090 smoke first. |
| v37/v39/v50 SEFH reliability heads conflict. | Reuse the same reliability tensor; error on incompatible configs. |
| Full-view regression from overfitting to sparse subsets. | Small loss weight (`0.01`) and identity init preserve the baseline. |

## Next action

Create `motionflow_mv/fusion/cross_view_attention_refinement_v50.py` and `configs/benchmark_v50_cross_view_attention_refinement_smoke.yaml`, warm-start from the best v46 checkpoint, and validate that `MPJPE@full` stays within 1 mm of v46 while `MPJPE@2/3` improves.
