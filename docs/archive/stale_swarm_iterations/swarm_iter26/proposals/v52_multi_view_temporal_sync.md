# v52 Multi-View Temporal Synchronization (MVTS)

## Proposal

**Title:** `multi_view_temporal_sync_v52` — learned cross-view temporal alignment for multi-camera human pose estimation  
**Author:** design-swarm  
**Tracking:** v52  
**Depends on:** v45-AGF, v46-SVG, v47-temporal, v49-Lite (optional)  

---

## 1. Motivation

The current MotionFlow-MultiView pipeline treats every camera view as temporally synchronized: a clip of length `T` is fed to each view with the same frame index. Real-world multi-view video rigs, however, suffer from sub-frame shutter offsets, variable frame timestamps, dropped frames, and rolling-shutter skew. These imperfections violate the implicit assumption that `(b, t, v, j)` corresponds to the same physical instant across views `v`. This misalignment propagates into the ST transformer and triangulation, especially hurting joints with fast motion (wrists/ankles) where even a 1-frame shift can produce large reprojection residuals.

Existing v47/v49 temporal modules refine the *output* pose sequence over time, and v51 CDSVR reasons about cross-domain view reliability. None of them explicitly model the **temporal relationship between views**. The v52 module proposes to insert a lightweight, warm-startable temporal-synchronization block *inside* the feature stream, before the ST transformer. It learns per-view temporal warp parameters and a cross-view temporal attention mechanism that aligns features in time, improving the "multi-view video -> human pose extraction -> multi-view fusion" stage of the paper story without changing data loaders.

---

## 2. Architecture

### 2.1 Position in `OmniMultiViewFusionV5`

Insert `multi_view_temporal_sync_v52` immediately **before** the spatio-temporal (ST) transformer, after all per-view feature enhancement modules (v33/v34/v36/v37/v48) and after the time/positional embeddings are added. At this point the tensor is:

```
feat: (B, T, V, J, d)
```

The module is residual and identity-initialized, so toggling it on with zero weights leaves the forward pass unchanged — a warm-start friendly addition.

### 2.2 Components

1. **Temporal-offset predictor**  
   A small MLP per view that predicts a continuous temporal offset (in units of frames) per `(view, joint)`:
   
   ```
   offset_vj = MLP_view( pool_T( feat[:, :, v, j, :] ) )  # (B, V, J)
   ```
   
   The per-view pooled feature has shape `(B, d)`; the MLP outputs `(B, V, J)` and is initialized to zero. Offsets are clamped to `[-max_shift, max_shift]` (e.g., `max_shift = 3` frames) and applied with differentiable temporal warping.

2. **Temporal-alignment warp**  
   For each `(b, v, j)` we sample the temporal feature curve at the predicted offsets using linear interpolation over the `T` frames:
   
   ```
   feat_warped[b, t, v, j, :] = interp( feat[b, t + offset_vj[b, v, j], v, j, :], t=0..T-1 )
   ```
   
   Shape preserved: `(B, T, V, J, d)`.

3. **Cross-view temporal attention**  
   After warping, we run a lightweight cross-view temporal attention that lets each `(t, v, j)` token attend to the same joint `j` at nearby temporal positions in *other* views:
   
   ```
   Q = W_q( feat_warped[t, v, j] )
   K = W_k( feat_warped[t, :, j] )
   V = W_v( feat_warped[t, :, j] )
   att = softmax( QK^T / sqrt(d_k) + mask_view )
   feat_sync[t, v, j] = W_o( att V )
   ```
   
   This is restricted to joint `j` and time `t` to keep memory `O(T V^2 J d)` instead of `O((T V)^2 J d)`.

4. **Residual fusion with learnable gate**  
   The final output is a gated residual so the module can fade in gradually:
   
   ```
   g = sigmoid( W_g( feat.mean(dim=1, keepdim=True) ) )  # (B, 1, V, J, d)
   out = (1 - g) * feat + g * feat_sync
   ```

### 2.3 Equations

Temporal offset predictor:

```
tau_vj = clamp( Linear_1( GELU( Linear_0( mean_t( feat_{b,:,v,j,:} ) ) ) ), -K, K )
```

Warped features (linear interpolation):

```
feat_warped[b,t,v,j,:] = (1-alpha) * feat[b, floor(t+tau), v, j, :]
                          + alpha    * feat[b, ceil(t+tau), v, j, :]
```

Cross-view temporal attention (single head shown for brevity; multi-head is used in practice):

```
A_{v'|v} = exp( q_v^T k_{v'} + m_{v'} ) / sum_u exp( q_v^T k_u + m_u )
feat_sync[b,t,v,j,:] = sum_v' A_{v'|v} * value( feat_warped[b,t,v',j,:] )
```

where `m_{v'}` masks out the current view (`v' = v`) and invalid views from `view_mask`.

---

## 3. Inputs and Outputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `feat` | `(B, T, V, J, d)` | Per-view, per-joint feature tokens from upstream fusion |
| `view_mask` | `(B, T, V)` | Boolean/0-1 mask for missing views |
| `temporal_mask` | `(B, T)` | Optional mask for padded frames within a clip |

**Outputs:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `feat_sync` | `(B, T, V, J, d)` | Temporally aligned and cross-view exchanged features |
| `offset_vj` | `(B, V, J)` | Predicted per-view, per-joint temporal offset in frames (diagnostic) |

---

## 4. Config Flags

Flag names follow the existing `use_vNN_*` convention:

```yaml
use_v52_multi_view_temporal_sync: false
v52_mvts_max_shift: 3.0
v52_mvts_d_model: 64
v52_mvts_n_heads: 4
v52_mvts_n_layers: 2
v52_mvts_use_view_self_excluded: true      # exclude a view from attending to itself
v52_mvts_offset_loss_weight: 0.01          # L2 regularizer on predicted offsets
v52_mvts_temporal_window: null             # if set, restrict attention to [-w, w] frames
v52_mvts_identity_init: true               # zero-init residual branch
```

---

## 5. Expected MPJPE Impact

- **Baseline:** v46-SVG/v49-Lite smoke reports ~30–63 mm on WebBridge smoke; A800 full runs are queued.
- **Expected improvement:** 2–5 mm on fast-motion sequences and datasets with known sub-frame camera skew (e.g., 3DPW, WebBridge dynamic clips). The gain is concentrated on distal joints, where temporal misalignment currently inflates error.
- **Neutral risk on static sequences:** identity-initialization and gating ensure no regression if offsets collapse to zero.

---

## 6. Risks

1. **Differentiable warping can introduce motion blur** when offsets are fractional. Mitigation: keep `max_shift` small (≤ 3 frames) and add an offset regularizer.
2. **Memory blow-up from `V x V` cross-view attention.** Mitigation: restrict to same-joint, same-time tokens only; no cross-time attention inside this block.
3. **Warm-start incompatibility with frozen upstreams.** Mitigation: identity-init plus optional `v52_mvts_identity_init` flag.

Full risk register is in `docs/swarm_iter26/reports/agent_multi_view_temporal_sync_risks.md`.

---

## 7. 5-Step Implementation Plan

1. **Module stub:** Create `motionflow_mv/fusion/multi_view_temporal_sync_v52.py` with the `TemporalOffsetPredictor`, `TemporalWarp`, and `CrossViewTemporalAttention` classes. Initialize all residual outputs to zero.
2. **Integration:** Wire `use_v52_multi_view_temporal_sync` into `OmniMultiViewFusionV5.__init__` and `forward`, inserting the call immediately before the ST transformer reshapes `feat` from `(B, T, V, J, d)` to `(B*J, T*V, d)`.
3. **Smoke config:** Add a new config `configs/benchmark_v52_mvts_smoke.yaml` that enables v52 on top of the v46-SVG baseline.
4. **Loss & regularization:** Add the offset regularizer to the training objective and expose `v52_mvts_offset_loss_weight` in the YAML.
5. **Evaluation:** Run the smoke test on RTX 4090; if `val_MPJPE` is < 75 mm and offsets show meaningful non-zero variance, add the full A800 queue entry in `scripts/launch_v33_a800_queue.py`.
