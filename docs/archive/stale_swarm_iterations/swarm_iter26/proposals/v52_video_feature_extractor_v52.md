# v52 Video Feature Extractor

## Proposal

**Title:** `video_feature_extractor_v52` — factorized video-level spatio-temporal feature extraction for multi-view 3D pose estimation
**Author:** design-swarm
**Tracking:** v52
**Depends on:** v25 geometry fusion, v45-AGF, v46-SVG (optional)

---

## 1. Motivation

The current MotionFlow-MultiView pipeline extracts per-frame, per-view features and feeds them into the spatio-temporal (ST) transformer. This mirrors the paper story "human pose extraction -> multi-view fusion" but omits an explicit **video feature extraction** stage: temporal or cross-view reasoning is only added after triangulation (v47/v49) or inside per-frame graph modules (v34/v36). Occluded joints, motion blur, and inconsistent 2D detections are therefore handled locally rather than with a coherent video-level representation.

The v52 module inserts a `VideoFeatureExtractorV52` block inside the fusion stream and builds a compact video-level feature by factorizing attention across time, views, and joints. It is residual and identity-initialized, leaving the v46/v49 baseline unchanged at init.

---

## 2. Architecture

### 2.1 Position in `OmniMultiViewFusionV5`

Insert the block in `OmniMultiViewFusionV5.forward` after the per-frame feature extractor and the optional per-view joint attention, but before the view positional embedding, camera conditioning, and the ST transformer. At this point the tensor is:

```
feat: (B, T, V, J, d)
```

### 2.2 Components

1. **Input projection.** Linear layer maps `feat` from `d` to `v52_video_feat_d_model` channels:
   ```
   z = W_in(feat)  # (B, T, V, J, d_model)
   ```

2. **Factorized attention branches.** Three parallel branches operate on different axes of the video tensor:

   - **Temporal branch:** For each `(v, j)`, multi-head self-attention over the `T` frames. Captures motion dynamics and smooths single-frame 2D outliers.
   - **Cross-view branch:** For each `(t, j)`, multi-head attention across the `V` views. Reinforces multi-view geometric consistency.
   - **Skeleton branch:** For each `(t, v)`, graph attention over the skeleton graph (`H36M_17_PARENTS` / `MPI_INF_3DHP_28_PARENTS`). Encodes kinematic priors.

   Each branch has its own transformer stack with layer-normalization and dropout.

3. **Adaptive branch fusion.** A lightweight gating network pools `z` across `(T, V, J)` to predict per-branch weights:
   ```
   g_t, g_v, g_j = softmax(MLG(pool(z)))  # (B, 3)
   fused = g_t * z_t + g_v * z_v + g_j * z_j  # (B, T, V, J, d_model)
   ```

4. **Residual output with warmup gate.** Project `fused` back to `d` channels and add it as a gated residual:
   ```
   out = feat + alpha * W_out(fused)
   ```
   `alpha` starts at `0.0` and linearly ramps to `1.0` over `v52_video_feat_warmup_steps`. This guarantees identity behavior at init.

### 2.3 Equations

Temporal branch (single head; multi-head in implementation):
```
Q_t = W_q^t z[:, :, v, j, :]
A_t = softmax(Q_t K_t^T / sqrt(d_k) + M_t)
z_t[:, :, v, j, :] = W_o^t (A_t V_t)
```

Cross-view branch:
```
Q_v = W_q^v z[:, t, :, j, :]
A_v = softmax(Q_v K_v^T / sqrt(d_k) + M_v)
z_v[:, t, :, j, :] = W_o^v (A_v V_v)
```

Skeleton branch (graph attention):
```
z_j[t, v, i, :] = sum_{k in N(i)} softmax_{k}( (W_q^j z_i)^T (W_k^j z_k) / sqrt(d_k) ) * W_v^j z_k
```

Residual output:
```
VideoFeatureExtractorV52(feat) = feat + alpha * Linear(W_out(fused))
```

---

## 3. Inputs and Outputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `feat` | `(B, T, V, J, d)` | Per-frame, per-view feature tokens from upstream fusion |
| `view_mask` | `(B, T, V)` | Boolean mask for missing views |
| `temporal_mask` | `(B, T)` | Optional mask for padded frames |

**Outputs:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `feat_video` | `(B, T, V, J, d)` | Video-level enhanced feature tokens, same shape as input |

---

## 4. Config Flags

Flag names follow the existing `use_vNN_*` convention:

```yaml
use_v52_video_feature_extractor: false
v52_video_feat_d_model: 64
v52_video_feat_n_heads: 4
v52_video_feat_temporal_layers: 2
v52_video_feat_crossview_layers: 2
v52_video_feat_joint_layers: 1
v52_video_feat_dropout: 0.1
v52_video_feat_use_branch_gating: true
v52_video_feat_identity_gate_init: true
v52_video_feat_warmup_steps: 1000
```

---

## 5. Expected MPJPE Impact

- **Baseline:** v46-SVG smoke reports ~30–63 mm on WebBridge; v25 A800 full runs report ~17 mm.
- **Expected improvement:** 1–4 mm on sequences with fast motion, occlusion, or noisy 2D detections. The factorized attention should reduce wrist/ankle jitter and improve temporal consistency before triangulation.
- **No-regression guarantee:** Identity-initialized projections and a zero-start warmup gate mean the module can be toggled on top of any v25/v46 checkpoint without warm-start regression.

---

## 6. Risks

See `docs/swarm_iter26/reports/agent_video_feature_extractor_v52_risks.md` for the full risk register.

---

## 7. 5-Step Implementation Plan

1. **Module stub:** Create `motionflow_mv/fusion/video_feature_extractor_v52.py` with `VideoFeatureExtractorV52(nn.Module)` containing `TemporalAttention`, `CrossViewAttention`, and `SkeletonGraphAttention` submodules. Zero-initialize all output projections and set `alpha=0.0`.

2. **Integration:** Wire `use_v52_video_feature_extractor` into `OmniMultiViewFusionV5.__init__` and `forward`, inserting the call after `_extract_frame_features` while `feat` has shape `(B, T, V, J, d)`.

3. **Smoke config:** Add `configs/benchmark_v52_video_feature_extractor_smoke.yaml` enabling v52 on top of the v46-SVG baseline.

4. **Warmup scheduling:** Add a scalar `alpha` parameter and a per-step ramp in the trainer; expose `v52_video_feat_warmup_steps` in YAML and CLI.

5. **Evaluation:** Run the RTX 4090 smoke; if `val_MPJPE` < 75 mm and branch weights are balanced, add the full A800 queue entry in `scripts/launch_v33_a800_queue.py`.
