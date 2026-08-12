# v53 — Geometry-Aware Attention Pooling (GAAP)

## TL;DR

Add a lightweight, geometry-biased cross-view attention block on top of the v52 Uncertainty-Weighted Triangulation (UWT) output. It pools per-view feature tokens using pairwise camera-geometry embeddings (ray directions, epipolar relationships, baseline), then applies a gated residual pose update. The module is identity at initialization, so a v52 checkpoint loaded with v53 enabled reproduces the same `val_MPJPE` within numerical noise.

## Motivation

v52 UWT learns scalar precision weights `w_{v,j}` per view and joint, but it still treats each view independently when forming feature representations. Multi-view fusion can be strengthened by explicitly modelling *which* views should attend to each other based on their geometric configuration:

* Wide-baseline views have stronger triangulation power but larger reprojection uncertainty.
* Views with similar ray directions are redundant; down-weighting them reduces over-confidence.
* Occluded or dropped views can borrow evidence from geometrically consistent views through feature pooling.

v53 therefore introduces **Geometry-Aware Attention Pooling (GAAP)** as a refinement step that fuses the per-view ST-transformer tokens under geometric attention before the final pose output. It aligns with the paper story: *multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized MotionFlow pipeline*.

## Architecture

`GeometryAwareAttentionPoolingV53` is inserted **after** the v52 UWT refinement and **before** the final residual refinement head in `OmniMultiViewFusionV5`:

```
feat (B,T,V,J,d) ──> GeometryAwareAttentionPoolingV53 ──> refined feat + delta_pose
pred_3d_init (B,T,J,3)
K,R,t, view_mask
```

### 1. Geometry embedding

For every target view `i` and source view `j`, compute a geometry vector

```
g_{ij} = concat(
    log( || c_j - c_i ||_2 + eps ),          # baseline length
    <r_i, r_j>,                               # cosine of ray/angular separation
    reproj_{i,j},                             # residual of pred_3d_init in view j w.r.t. view i ray
    I[i==j]                                   # self-view indicator
)  in R^g
```

where `c_v = -R_v^T t_v` is the camera centre and `r_v` is the principal-axis / ray direction derived from `K_v, R_v`. `reproj_{i,j}` is the per-joint reprojection distance of `pred_3d_init` in view `j`, normalized by the median over views for stability.

The embedding is normalized with a running mean/std and clamped to `[-10, 10]` to handle corrupted cameras.

### 2. Geometry-biased cross-view attention

For each view `i` and joint `j`:

```
q_i = W_q f_i                                    # (d)
k_{ij} = W_k f_j + MLP_k(g_{ij})                # (d)
v_{ij} = W_v f_j + MLP_v(g_{ij})                # (d)

a_{ij} = (q_i^T k_{ij}) / sqrt(d) + m_{ij}      # scalar attention logit
m_{ij} = -1e9            if view j is masked
       = 0               otherwise

w_{ij} = softmax_j(a_{ij})                       # (V) attention weights
f'_i   = f_i + γ * W_o  sum_j w_{ij} v_{ij}     # gated residual update
```

`γ` is a learnable scalar **initialized to 0**. `W_o` is zero-initialized, so `f'_i = f_i` at init (identity). Dropout and LayerNorm follow the attention output.

### 3. Pose residual refinement

After updating tokens, pool across views and refine the pose:

```
h      = mean_i(f'_i)                            # (B,T,J,d)
delta  = MLP( concat(h, pred_3d_init) )          # (B,T,J,3)
pred   = pred_3d_init + α * delta
```

`α` is a learnable scalar **initialized to 0**, and the final layer of the MLP is zero-initialized. Therefore `pred = pred_3d_init` at initialization.

### 4. Auxiliary geometry-consistency loss

To prevent the attention from collapsing to a single view, we add a tiny auxiliary loss that encourages the attention distribution to match the v52 precision weights:

```
L_gaap = MSE( softmax_i over j(a_{ij}),  target_{ij} )
target_{ij} = w^{v52}_{ij} / sum_k w^{v52}_{ik}
```

This is weighted by `v53_gaap_loss_weight` and added only after the warmup epoch count.

## Inputs and Outputs

**Inputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `features` | `(B, T, V, J, d)` | Per-view ST-transformer tokens from `OmniMultiViewFusionV5`. |
| `points_2d` | `(B, T, V, J, 2)` | Input 2-D keypoints. |
| `K` | `(B, T, V, 3, 3)` | Intrinsics. |
| `R` | `(B, T, V, 3, 3)` | Camera rotations. |
| `t` | `(B, T, V, 3)` | Camera translations. |
| `pred_3d_init` | `(B, T, J, 3)` | 3-D pose estimate produced by v52 UWT. |
| `view_mask` | `(B, T, V)` | Boolean mask, `True` = valid view. |
| `v52_weights` | `(B, T, V, J)` | Optional v52 precision weights to guide auxiliary loss. |

**Outputs**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Refined 3-D pose. |
| `gaap_loss` | scalar | Auxiliary geometry-consistency loss. |
| `attn_weights` | `(B, T, V, V, J)` | Optional cross-view attention weights for diagnostics. |

## Config Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v53_geometry_aware_attention_pooling` | `bool` | `False` | Master toggle. |
| `v53_gaap_hidden` | `int` | `64` | Hidden dimension of geometry MLPs. |
| `v53_gaap_n_heads` | `int` | `4` | Attention heads. |
| `v53_gaap_n_layers` | `int` | `1` | Number of stacked GAAP layers. |
| `v53_gaap_use_epipolar_bias` | `bool` | `True` | Include epipolar-line distance in `g_{ij}`. |
| `v53_gaap_use_ray_geometry` | `bool` | `True` | Include ray-direction features in `g_{ij}`. |
| `v53_gaap_identity_init` | `bool` | `True` | Zero-initialize output projection and pose MLP. |
| `v53_gaap_residual_gate_init` | `float` | `0.0` | Initial value of update gates `γ` and `α`. |
| `v53_gaap_loss_weight` | `float` | `0.01` | Weight of `L_gaap`. |
| `v53_gaap_warmup_epochs` | `int` | `0` | Epochs before `L_gaap` is added. |
| `v53_gaap_dropout` | `float` | `0.1` | Dropout on attention output. |

## Expected MPJPE Impact

* **Local 4090 smoke (50 samples, 1 epoch):** 0.5–1.0 mm reduction over v52 baseline if the geometry bias helps redundant-view suppression; risk of ~1 mm degradation if the attention layer overfits the tiny smoke set.
* **A800 full run:** 0.3–0.8 mm reduction on `val_MPJPE`. The largest gains are expected on sparse-view settings (`MPJPE@2` and `MPJPE@3`) because GAAP can borrow evidence across views when one or two views are missing or down-weighted.
* **Warm-start verification:** Loading a v52 checkpoint with v53 enabled should change `val_MPJPE` by less than 0.05 mm; this is the go/no-go gate for the full run.

## Risks and Mitigations

See `docs/swarm_iter27/reports/agent_geometry_aware_attention_pooling_v53_risks.md` for the full risk register. The main concerns are: geometry embedding instability with noisy calibration, `O(V^2)` attention memory, overfitting on small smoke configs, interference with v48 domain-adapted features, and warm-start drift due to non-zero geometry MLP biases.

## 5-Step Implementation Plan

1. **Module stub** — Create `motionflow_mv/fusion/geometry_aware_attention_pooling_v53.py` with `GeometryAwareAttentionPoolingV53(nn.Module)`, implementing the geometry embedding, multi-head attention, gated token update, and pose residual MLP. Ensure all final projections are zero-initialized and gates start at 0.

2. **Integration into `OmniMultiViewFusionV5`** — In `motionflow_mv/fusion/omniview_fusion_v5.py`, add the config flags, instantiate the module when `use_v53_geometry_aware_attention_pooling=True`, and call it immediately after the v52 UWT block. Aggregate `gaap_loss` into `epi_loss` using `v53_gaap_loss_weight` and respect `v53_gaap_warmup_epochs`.

3. **Smoke config and script** — Add `configs/benchmark_v53_gaap_smoke.yaml` and `scripts/run_v53_gaap_smoke_local_4090.sh` that mirror the v52 smoke setup. Include the warm-start test: load the latest v52 checkpoint, enable v53, and verify `val_MPJPE` is unchanged.

4. **A800 queue entry** — Append a `v53_geometry_aware_attention_pooling_on_v52` entry to `scripts/launch_v33_a800_queue.py` behind the v52 run. Set `d=64` for the first full run to match the v45/v46/v47/v48 smoke convention before committing to `d=128`.

5. **Evaluation** — Run the smoke on RTX 4090. If `val_MPJPE@full` is within 0.1 mm of v52 and `MPJPE@2/3` improves, launch the A800 full run and update the status tables in `AGENTS.md`.
