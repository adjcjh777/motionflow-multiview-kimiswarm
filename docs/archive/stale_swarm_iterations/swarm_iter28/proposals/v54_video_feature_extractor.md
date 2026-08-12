# v54: Video Feature Extractor — Physics-Guided Spatiotemporal Feature Refinement

**Module name:** `video_feature_extractor_v54.py`  
**Integration point:** `OmniMultiViewFusionV5`, immediately before the v52 Uncertainty-Weighted Triangulation (UWT) head.  
**Depends on:** v25/v45 geometry fusion, v52 UWT; optional v53 Physical-Space Calibration (PSC).  
**Supersedes:** The v53 Video Feature Extractor proposal by adding multi-scale temporal convolution, geometry-biased cross-view attention, a skeleton graph mixer, and a physics-guided per-joint gate.

## 1. Motivation

v52 UWT triangulates from per-frame feature tokens, and v53 then calibrates the resulting pose against physical constraints. Both treat temporal, multi-view, and physical consistency as downstream corrections, which limits how much per-frame triangulation errors can be recovered.

The v54 Video Feature Extractor (VFE) refines feature tokens *before* triangulation, fusing evidence across time, views, and the skeleton. It uses the coarse v25/v45 pose plus physical signals (reprojection residual, bone-length deviation, and foot-to-floor distance) to decide which joints need the largest update, improving both full-view and sparse-view MPJPE.

## 2. Architecture

Notation: batch \(B\), time \(T\), views \(V\), joints \(J\), feature dim \(d\). Inputs are \(F \in \mathbb{R}^{B \times T \times V \times J \times d}\) and coarse pose \(X_{\text{init}} \in \mathbb{R}^{B \times T \times J \times 3}\).

### 2.1 Multi-Scale Causal Temporal Encoder

Reshape \(F\) to \((B V J) \times d \times T\) and apply \(L\) layers of multi-scale causal 1-D convolutions:

\[
H^{(l)} = H^{(l-1)} + W_{\text{proj}}^{(l)}\left[
\text{CausalConv}_{k=3}\bigl(\text{LN}(H^{(l-1)})\bigr) \;
\|\; \text{CausalConv}_{k=5}\bigl(\text{LN}(H^{(l-1)})\bigr)
\right]
\]

The concatenated features are projected back to \(d\) channels by a zero-initialized \(1\times 1\) convolution, so the residual branch is a no-op at initialization.

### 2.2 Geometry-Biased Cross-View Attention

Reshape to \((B T J) \times V \times d\) and apply scaled dot-product attention with a ray/epipolar bias:

\[
A_{ij} = \text{softmax}\left( \frac{Q_i K_j^T}{\sqrt{d_k}} + B^{\text{geo}}_{ij} \right),
\quad
H_v = H_t + \alpha \cdot A V
\]

Missing views are masked with `view_mask`.

### 2.3 Skeleton Graph Mixer

Reshape to \((B T V) \times J \times d\) and apply graph attention over the skeleton parent list:

\[
H_s[i] = H_v[i] + \alpha \sum_{k \in \mathcal{N}(i)} \text{softmax}_{k}\left( \frac{(W_q H_v[i])^T (W_k H_v[k])}{\sqrt{d_k}} \right) W_v H_v[k]
\]

Mixed skeletons fall back to a simple kinematic chain, so no data-loader changes are required.

### 2.4 Physics-Guided Per-Joint Gate

From the coarse pose and 2-D detections, compute per-joint cues:

- Reprojection residual: \(r_{j} = \frac{1}{V_j}\sum_{v \in \mathcal{V}_j} \|\Pi_v(X_j) - p^{(2D)}_{v j}\|_2\)
- Bone-length deviation: \(\delta^{\text{bone}}_{j} = |\|X_j - X_{\text{parent}(j)}\| - \bar{b}_j|\)
- Floor distance: \(\delta^{\text{floor}}_{j}\) for foot/ankle joints

These are fed to a tiny MLP: \(\gamma_j = \text{sigmoid}(\text{MLP}_{\text{physics}}(p_j))\). The final refined features are:

\[
F_{\text{video}} = F + \alpha \cdot \bigl(W_{\text{out}}(H_s) \odot \gamma_j\bigr)
\]

At initialization, the global gate \(\alpha = 0\) and all output projections are zero-initialized, so \(F_{\text{video}} = F\). This guarantees warm-start compatibility with any v52/v53 checkpoint.

## 3. Inputs and Outputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `features` | \((B, T, V, J, d)\) | Per-frame, per-view, per-joint feature tokens. |
| `pred_3d_init` | \((B, T, J, 3)\) | Coarse 3-D pose from v25/v45. |
| `points_2d` | \((B, T, V, J, 2)\) | 2-D keypoints for reprojection residual. |
| `K` / `R` / `t` | \((B, T, V, 3, 3)\) / \((B, T, V, 3, 3)\) / \((B, T, V, 3)\) | Camera intrinsics and extrinsics. |
| `view_mask` | \((B, T, V)\) | Boolean mask for missing views. |
| `features_video` | \((B, T, V, J, d)\) | Refined feature tokens for v52 UWT. |

## 4. Config Flags

```yaml
use_v54_video_feature_extractor: true
v54_vfe_d_model: 64
v54_vfe_n_heads: 4
v54_vfe_temporal_layers: 2
v54_vfe_temporal_kernels: [3, 5]
v54_vfe_crossview_layers: 1
v54_vfe_skeleton_layers: 1
v54_vfe_use_geometry_bias: true
v54_vfe_use_skeleton_graph: true
v54_vfe_use_physics_gate: true
v54_vfe_physics_hidden: 64
v54_vfe_dropout: 0.1
v54_vfe_gate_init: 0.0          # initial global gate α
v54_vfe_warmup_epochs: 0       # optional warmup before α leaves 0
```

## 5. Expected MPJPE Impact

- **Smoke:** within 0.5 mm of the v52/v53 baseline due to identity-at-init.
- **Full A800:** 0.8–2.0 mm reduction on WebBridge/H36M/MPI validation; larger gains on motion-blur and occlusion.
- **Sparse-view:** `MPJPE@2` may improve by 2–3 mm through better weighting of visible views.

## 6. Risks

See `docs/swarm_iter28/reports/agent_video_feature_extractor_risks.md` for the detailed risk register.

## 7. 5-Step Implementation Plan

1. **Create the module** `motionflow_mv/fusion/video_feature_extractor_v54.py` with `VideoFeatureExtractorV54(nn.Module)`; implement the causal encoder, geometry-biased cross-view attention, skeleton graph mixer, and physics gate. Zero-initialize every output projection and the global gate.
2. **Wire into `OmniMultiViewFusionV5`**: add `use_v54_video_feature_extractor` and hyperparameters to `__init__`, then call the module after the v25/v45 coarse pose is available and before the v52 UWT head.
3. **Add smoke config** `configs/benchmark_v54_video_feature_extractor_smoke.yaml` enabling v54 on top of the v52 baseline with conservative settings.
4. **Run the smoke test** on the RTX 4090: verify identity-at-init by loading a v52 checkpoint and confirming `val_MPJPE` matches within 0.1 mm; train one epoch and check for NaN/OOM and gate magnitude.
5. **Queue the A800 full run** in `scripts/launch_v33_a800_queue.py` (e.g., `v54_video_feature_extractor_on_v52`), compare `val_MPJPE`/`MPJPE@k` to the v52/v53 baselines, and update `docs/swarm_iter28/status.md`.
