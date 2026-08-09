# v51 Geometry-Aware Attention Pooling

## Module overview

**`GeometryAwareAttentionPoolingV51`** (`motionflow_mv/fusion/geometry_aware_attention_pooling_v51.py`)

v51 replaces the per-joint mean/max view pooling that follows the v46 sparse-view reliability head with a geometry-aware attention pooling layer. Instead of collapsing the per-view pose tokens with a fixed reduction, the module lets each joint query all view tokens under both *feature* and *geometric* compatibility. The geometric key encodes epipolar distance, ray intersection quality, and camera-relative ray direction for every (view, joint) pair, so the model learns to up-weight views that are geometrically trustworthy and down-weight occluded or noisy views before triangulation. The module is identity-at-init by zero-initializing the attention gate, so enabling it on a warm v46/v50 checkpoint preserves the baseline.

## Architecture

Given per-view pose tokens `X ∈ R^(V×J×D)` from the v46 backbone, plus 2-D keypoints `P ∈ R^(V×J×2)` and calibrated cameras, we compute:

1. **Geometry keys**: for each (view `v`, joint `j`) pair, form a 7-D vector `g_vj = [epi_vj, ray_dir_vj · n_j, baseline_angle_vj, reproj_err_vj, cam_dist_vj, view_count_signal, reliability_vj]` where `reliability_vj` comes from the v46/v50 reliability head and `n_j` is the current estimate of the bone direction.
2. **Geometry-biased attention**: project `g_vj` into the token dimension with a learned MLP, add it as a bias to the scaled dot-product attention, and compute per-joint attention over views.
3. **Residual gate**: a learned residual gate `α = sigmoid(MLP(pool(X)))`, initialized near zero, blends the pooled token with the original mean-pooled token.
4. **Output**: the pooled token is fed back into the triangulation/aggregation path in `omniview_fusion_v5.py` exactly where the previous pooling occurred.

The module is implemented as a small transformer-style encoder with multi-head attention (`num_heads=4`, `hidden=64`) and two layers.

## New config flags and defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_geometry_aware_attention_pooling` | bool | `False` |
| `v51_gaap_hidden` | int | `64` |
| `v51_gaap_num_heads` | int | `4` |
| `v51_gaap_num_layers` | int | `2` |
| `v51_gaap_geometry_bias_type` | str | `"epipolar_ray_reproj"` |
| `v51_gaap_dropout` | float | `0.1` |
| `v51_gaap_identity_gate` | bool | `True` |
| `v51_gaap_min_keep_fraction` | float | `0.3` |
| `loss.v51_gaap_loss_weight` | float | `0.01` |
| `loss.v51_gaap_geometry_consistency_weight` | float | `0.005` |

## Loss term

```
L_gaap = λ * [ L_reproj_nll(pool) ]
L_geom = λ_geom * (1 / VJ) Σ_v,j w_vj * |g_vj - ḡ_j|^2
```

`L_reproj_nll` is the standard negative log-likelihood of the triangulated 3-D pose under the pooled features. `L_geom` regularizes the attention weights `w_vj` to respect the geometric ranking: views with smaller geometric residual should receive higher weight. The geometry consistency loss is only active when `loss.v51_gaap_geometry_consistency_weight > 0`.

## Evaluation metric

- `MPJPE@k` for `k = 2, 3, 4, full` using `experiments/eval_variable_views.py`.
- `Spearman(attention_weight_vj, 1 / reproj_err_vj)` > 0.30 to verify that geometry drives the pooling.
- Per-joint error improvement on distal joints (wrists/ankles), which benefit most from discarding bad views.

## Expected MPJPE impact

- `MPJPE@2`: −2 to −4 mm
- `MPJPE@3`: −1 to −2 mm
- `MPJPE@full`: −0.5 to −1 mm

The largest relative gain is expected for sparse views and for distal joints, where the current mean/max pooling is most likely to keep an noisy view.

## Main risk

**Risk**: The geometric bias can dominate the feature signal and wash out learned view relationships, causing a regression on full-view accuracy or instability in the first epochs.

**Mitigation**: Keep the residual gate zero-initialized and identity-at-init; freeze the base model for one epoch; clamp the geometric bias magnitude to a learned temperature; and start ablations with `loss.v51_gaap_loss_weight=0.001` before the default 0.01.

## Integration notes

- Insert after the v46 reliability head and before the final triangulation/aggregation in `motionflow_mv/fusion/omniview_fusion_v5.py`.
- Reuse the existing camera utilities for epipolar distance and ray-direction computation.
- Smoke config: `configs/benchmark_v51_geometry_aware_attention_pooling_smoke.yaml`; smoke script: `scripts/run_v51_gaap_smoke_local_4090.sh`.
- Ablate against the v50 SEFH baseline once both smokes are available.
