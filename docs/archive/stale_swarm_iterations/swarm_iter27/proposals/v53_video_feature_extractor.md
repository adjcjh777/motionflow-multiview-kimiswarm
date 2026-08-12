# v53: Video Feature Extractor (VFE)

**Module name:** `video_feature_extractor_v53.py`  
**Integration point:** `OmniMultiViewFusionV5`, immediately before the v52 Uncertainty-Weighted Triangulation (UWT) head.  
**Depends on:** v52 UWT.

## 1. Motivation

The current v52 UWT head triangulates from per-frame, per-view feature tokens \(F \in \mathbb{R}^{B \times T \times V \times J \times d}\). These tokens are rich in single-frame appearance and geometry cues, but they discard the temporal continuity of the multi-view video. In contrast, v47 temporal aggregation and v49-Lite operate *after* triangulation on the 3-D poses, so any error introduced by per-frame triangulation cannot be fully recovered. The v53 Video Feature Extractor closes this gap by extracting causal spatiotemporal video features *before* triangulation and feeding them into v52, allowing the triangulation weights to exploit motion dynamics, temporal consistency, and multi-view spatiotemporal redundancy.

## 2. Architecture

v53 is a lightweight, causal, residual spatiotemporal feature extractor. It receives the same feature tokens that v52 consumes, reshapes them, applies a causal temporal convolution along the time axis, optionally mixes across the skeleton and views, and returns refined tokens with an identity-at-init residual gate.

### 2.1 Notation

- \(B\): batch size
- \(T\): temporal clip length
- \(V\): number of views
- \(J\): number of joints
- \(d\): feature dimension
- \(F \in \mathbb{R}^{B \times T \times V \times J \times d}\): input feature tokens

### 2.2 Temporal Causal Encoder

Reshape \(F\) to \((B \cdot V \cdot J, d, T)\) and apply \(L\) causal 1-D convolution layers with kernel size \(K\) and hidden dimension \(d_{\text{hidden}}\):

\[
H^{(0)} = \text{Permute}(F), \quad H^{(0)} \in \mathbb{R}^{(B V J) \times d \times T}
\]

\[
H^{(l)} = H^{(l-1)} + \text{CausalConv1D}\left(\text{LayerNorm}(H^{(l-1)})\right), \quad l = 1 \dots L
\]

Each `CausalConv1D` uses left padding \(K-1\) so the output at time \(t\) only depends on inputs \(\le t\), preserving online/causal inference. The hidden layer expands to \(d_{\text{hidden}} = 2d\) and projects back to \(d\) with a \(1\times 1\) convolution and GELU.

### 2.3 Spatiotemporal Mixer (optional)

After the temporal encoder, reshape to \((B \cdot T) \times V \times J \times d\). A lightweight factorized mixer is applied:

1. **Joint mixer:** for each view, apply a 2-layer MLP across joints with shared weights:
   \[
   M_{\text{joint}} = F_{\text{joint-MLP}}(H^{(L)}) \in \mathbb{R}^{(BT) \times V \times J \times d}
   \]

2. **View mixer:** for each joint, apply a 2-layer MLP across views:
   \[
   M_{\text{view}} = F_{\text{view-MLP}}(M_{\text{joint}}) \in \mathbb{R}^{(BT) \times V \times J \times d}
   \]

Both MLPs use LayerNorm, GELU, and dropout for regularization. The mixer is optional and gated by `v53_vfe_use_spatial_mixer`.

### 2.4 Identity-at-Init Output

The final refined feature tokens are computed as:

\[
F_{\text{video}} = F + g \cdot \text{MLP}_{\text{out}}(M_{\text{view}})
\]

where \(g\) is a scalar learnable gate initialized to \(0\), and the final output projection is zero-initialized. Therefore, at initialization:

\[
F_{\text{video}} = F
\]

This guarantees warm-start compatibility: loading a v52 checkpoint into a v53-enabled model and freezing the v53 gate reproduces the baseline.

### 2.5 Integration into v52 UWT

The v53 module is instantiated in `OmniMultiViewFusionV5.__init__` under the flag `use_v53_video_feature_extractor`. In `forward`, immediately before the v52 UWT call:

```python
feat_for_uwt = self.video_feature_extractor_v53(feat)  # (B, T, V, J, d)
```

`feat_for_uwt` replaces the original `feat` as the `features` argument to `UncertaintyWeightedTriangulationV52.forward`. All downstream losses and heads remain unchanged.

## 3. Inputs and Outputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input `features` | \((B, T, V, J, d)\) | Per-frame, per-view, per-joint feature tokens produced by upstream fusion. |
| Output `features_video` | \((B, T, V, J, d)\) | Temporally refined, identity-at-init feature tokens for v52 UWT. |

## 4. Config Flags

```yaml
use_v53_video_feature_extractor: true   # Master switch
v53_vfe_hidden: 128                       # Hidden dim of temporal conv (default 2*d)
v53_vfe_n_layers: 2                       # Number of causal conv layers
v53_vfe_kernel_size: 3                    # Kernel size for causal conv
v53_vfe_use_spatial_mixer: true           # Enable joint/view mixer
v53_vfe_mixer_dropout: 0.1                # Dropout in joint/view MLPs
v53_vfe_gate_init: 0.0                    # Initial residual gate (0.0 = identity)
```

## 5. Expected MPJPE Impact

- **Smoke (RTX 4090, 1 epoch, ~50 samples):** expect neutral-to-positive impact; the identity gate should keep the baseline at worst. Target `val_MPJPE` within 0.5 mm of v52 UWT smoke.
- **Full A800 run:** by exploiting temporal video context during triangulation, v53 is expected to reduce `val_MPJPE` by **0.5–1.5 mm** on WebBridge/H36M/MPI mixed validation, with larger gains on sequences with motion blur or partial occlusion.
- **Sparse-view settings:** the temporal prior should stabilize predictions when \(V\) is small (e.g., 2 views), improving `MPJPE@2` by up to 2 mm.

## 6. Risks

See `docs/swarm_iter27/reports/agent_video_feature_extractor_risks.md` for a detailed risk register. Top concerns include: (a) causal temporal conv overfitting on short clips, (b) added compute cost before the already-expensive v52 head, and (c) identity gate staying near zero if auxiliary losses dominate.

## 7. 5-Step Implementation Plan

1. **Create module** `motionflow_mv/fusion/video_feature_extractor_v53.py` with `VideoFeatureExtractorV53(nn.Module)`, implementing causal temporal conv, optional factorized joint/view mixer, and identity-at-init output gate.
2. **Wire into `OmniMultiViewFusionV5`**: add config flags in `__init__`, instantiate the module, and insert a single call to refine `feat` right before the v52 UWT forward pass.
3. **Add smoke config** `configs/benchmark_v53_video_feature_extractor_smoke.yaml` that enables v53 on top of the v52 baseline and sets `v53_vfe_hidden=128`, `v53_vfe_n_layers=2`, `v53_vfe_kernel_size=3`.
4. **Run smoke test** on the local RTX 4090: verify identity-at-init (loading a v52 checkpoint gives identical `val_MPJPE` to within 0.1 mm) and that one training epoch completes without NaN/OOM.
5. **Queue A800 full run** in `scripts/launch_v33_a800_queue.py` under the v53 entry; compare `val_MPJPE` and `MPJPE@k` against the v52 UWT baseline and update `docs/swarm_iter27/status.md`.
