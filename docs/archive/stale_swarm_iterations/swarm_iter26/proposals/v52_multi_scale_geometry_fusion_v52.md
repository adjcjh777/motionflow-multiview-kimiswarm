# v52 Multi-Scale Geometry Fusion (`multi_scale_geometry_fusion_v52`)

## 1. Motivation

Current multi-view fusion in `OmniMultiViewFusionV5` operates at a single joint-level resolution: each view/joint token is refined by cross-view attention and then triangulated. Human pose, however, has natural structure at multiple scales—fine joints, coarse body parts, and the whole skeleton. When only 2–3 views are available or cameras are slightly miscalibrated, purely joint-level matching is brittle; coarser scales can provide robust geometric anchors that regularize fine-scale triangulation.

The paper pipeline is: multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized motionflow pipeline. **v52 lives in the multi-view fusion and calibration stage.** It explicitly fuses geometry across three scales (joint, part, body) before triangulation, using calibrated ray and epipolar constraints. The block is designed to be warm-startable/identity-at-init so it can be dropped into an already-trained v45/v46/v47/v48 model without destabilizing it.

## 2. Architecture

`MultiScaleGeometryFusionV52` is inserted **after the spatio-temporal (ST) transformer** and **before the covariance/weight heads** in `OmniMultiViewFusionV5.forward`. It takes the 5-D feature tensor

```
feat  : (B, T, V, J, d)
```

and produces a residual update of the same shape. At initialization the residual is zero, so `output == feat`.

### 2.1 Multi-scale token construction

Three scales are defined:

| Scale `s` | Name | Resolution | How it is built |
|---|---|---|---|
| 0 | Joint | `J` tokens | Identity: `feat_0 = feat` |
| 1 | Part | `J_p` tokens (e.g. 6) | Soft pooling: `feat_1[v, p] = Σ_j A_pj · feat[v, j] / Σ_j A_pj` |
| 2 | Body | 1 token | Global pooling: `feat_2[v] = Σ_j a_j · feat[v, j] / Σ_j a_j` |

`A ∈ R^(J_p × J)` is a learnable soft assignment matrix initialized to a dataset-specific hard grouping (head, torso, left/right arm, left/right leg). `a ∈ R^J` is a learnable weight vector initialized uniformly. Both use a temperature-controlled softmax so the assignment can sharpen during training but starts anatomically.

### 2.2 Per-scale cross-view attention with geometry bias

For each scale `s`, tokens are refined by a single multi-head self-attention over views:

```
Q_s = W_q^(s) feat_s,   K_s = W_k^(s) feat_s,   V_s = W_v^(s) feat_s
attn_s = softmax( (Q_s K_s^T)/sqrt(d_h) + γ_s · b_s^geo )
feat_s' = feat_s + MLP_s( Concat[ V_s, attn_s V_s ] )
```

`b_s^geo` is a multi-view geometry bias with two components:

1. **Ray-intersection affinity** (from `multiview_geometry_fusion_v25.py`):
   ```
   b_ray(i, j) = -(d_ij / σ_d) - ((1 - cos θ_ij) / σ_a)
   ```
   where `d_ij` is the shortest distance between rays from views `i` and `j`, and `θ_ij` is the angle between them.
2. **Epipolar distance** (from `epipolar_attention_bias`):
   ```
   b_epi(i, j) = -α · mean_j dist_epi(x_j^i, F_ij, x_j^j)
   ```
   pooled over the joints/parts at the current scale.

`γ_s` is a learnable scalar per scale **initialized to 0.0**, so the geometry bias is ignored at the start.

### 2.3 Scale mixing and identity-at-init

After up-sampling the part and body tokens back to joint-level, the three scale residuals are mixed by a small gated network:

```
δ = Σ_s g_s · (Upsample(feat_s') - feat)
```

The mixing logits are produced by an MLP whose final layer is initialized to zero, so `g_s = softmax(0) = 1/3` and, crucially, the final residual `δ` is zero at initialization. Therefore:

```
output = feat + λ · δ,   λ initialized to 0.0
```

makes the whole block an exact identity at init (warm-startable).

## 3. Inputs and outputs

**Inputs**
- `feat`: `(B, T, V, J, d)` tokens after ST transformer.
- `points_2d`: `(B, T, V, J, 2)` detected 2-D keypoints.
- `K, R, t`: `(B, T, V, 3, 3)`, `(B, T, V, 3, 3)`, `(B, T, V, 3)` calibrated cameras.
- `view_mask`: `(B, T, V)` bool/float mask.
- Optional `domain_id` for future cross-domain extension.

**Outputs**
- `feat_out`: `(B, T, V, J, d)` refined tokens, same shape as input.
- `msgf_loss`: scalar auxiliary geometry consistency loss (optional, can be zero).

## 4. Config flags

```yaml
use_v52_multi_scale_geometry_fusion: true
v52_msgf_scales: [0, 1, 2]                # which scales to use
v52_msgf_part_groups:                     # dataset-specific; example for H36M 17-joint
  head: [0, 1]
  torso: [2, 5, 8]
  left_arm: [3, 4]
  right_arm: [6, 7]
  left_leg: [9, 10, 11]
  right_leg: [12, 13, 14]
v52_msgf_n_heads: 4
v52_msgf_hidden: 64
v52_msgf_n_layers: 1
v52_msgf_dropout: 0.1
v52_msgf_residual_gate_init: 0.0          # zero -> identity at init
v52_msgf_geometry_bias_init: 0.0          # zero -> geometry bias ignored at init
v52_msgf_use_ray_bias: true
v52_msgf_use_epipolar_bias: true
v52_msgf_aux_loss_weight: 0.0
```

## 5. Expected MPJPE impact

- **Full-view (4+ views)**: small gain, ~0.3–0.8 mm, because the baseline already has enough geometry.
- **Sparse-view (2–3 views)**: larger gain, ~1.5–3.0 mm, because part/body scales provide stable cross-view anchors when joint-level cues are ambiguous.
- **Cross-domain / WebBridge**: modest improvement from coarser-scale regularization, ~0.5–1.5 mm.

## 6. Risks

1. **Part-group mismatch across datasets** (fixed H36M grouping harms MPI/WebBridge). Use a configurable `v52_msgf_part_groups` per dataset and a learnable soft assignment so the model can adapt.
2. **Geometry bias can be noisy** when cameras are miscalibrated. Initialize `γ_s = 0` and clamp epipolar/ray distances to soft values; let the network learn when to trust the bias.
3. **Compute/memory overhead** from three cross-view attention passes. Keep `n_layers=1`, downsample at part/body scales, and optionally apply stochastic depth.
4. **Interaction with v45/v46/v47/v48** could over-regularize. Start with `v52_msgf_aux_loss_weight=0` and gate the residual with a small learned scalar; only scale up after smoke tests.

## 7. Implementation plan

1. Create `motionflow_mv/fusion/multi_scale_geometry_fusion_v52.py` implementing `MultiScaleGeometryFusionV52` with scale pooling, per-scale geometry-biased cross-view attention, and zero-initialized residual mixer.
2. Wire the module into `OmniMultiViewFusionV5.__init__` and `forward`, placing it after the ST transformer and before the covariance/weight heads.
3. Add YAML config flags (template above) and a smoke config `configs/benchmark_v52_multi_scale_geometry_fusion_smoke.yaml`.
4. Validate warm-start: run a synthetic forward pass and assert `|output - feat| < 1e-5` before training.
5. Run smoke training on the local RTX 4090, comparing val_MPJPE to the v46/v47 baselines; iterate on grouping, scale set, and bias initialization.
