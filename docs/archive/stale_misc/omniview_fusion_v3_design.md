# OmniMultiViewFusion v3 Design

**Target venues**: ICRA / CVPR 2027  
**Status**: Prototype (new files only, v2 untouched)  
**Location**:
- `motionflow_mv/fusion/omniview_fusion_v3.py`
- `experiments/train_omniview_fusion_v3_mpiinf3dhp.py`

## 1. Motivation

OmniMultiViewFusion v2 already unifies four strong ideas: visibility gating, graph-joint attention, uncertainty-weighted triangulation, and a spatio-temporal transformer. v3 targets two remaining weaknesses of that design:

1. **Flat fusion scale** – the ST transformer attends over a flat `T × V` grid and treats all joints at a single resolution.  Human motion and skeletons are inherently hierarchical (coarse limb groups vs. fine joints, fast vs. slow temporal dynamics).
2. **Geometry-agnostic attention** – the transformer has no direct access to calibrated camera geometry, so it must learn epipolar / rig consistency from data.

v3 adds (a) hierarchical multi-scale temporal/cross-view fusion and (b) camera-conditioned / epipolar-biased cross-view attention, while keeping every v2 capability intact.

## 2. Architecture

```
Input (B, T, V, J, 3)
        │
        ▼
Principal-point / focal correction (v2)
        │
        ▼
Per-frame ray-aware feature extraction (v2)
        │
        ▼
Optional dense joint-level self-attention (v2)
        │
        ▼
Graph-joint attention over (view, joint) skeleton graph (v2)
        │
        ▼
[NEW] Camera-parameter conditioning ─────┐
        │                                  │
        ▼                                  │
[NEW] Hierarchical multi-scale temporal/   │
      cross-view fusion                    │
        │                                  │
        ▼                                  │
Spatio-temporal (time + view) transformer  │
[NEW] with optional epipolar attention bias┘
        │
        ▼
Anisotropic covariance + visibility gating (v2)
        │
        ▼
Uncertainty-weighted DLT + adaptive Gauss-Newton (v2)
        │
        ▼
Residual 3D refinement (v2)
        │
        ▼
3D joints + fusion weights + visibility + covariance + epipolar loss
```

### 2.1 Hierarchical Multi-Scale Temporal / Cross-View Fusion

Module: `_HierarchicalMultiscaleFusion`  
Input / output shape: `(B, T, V, J, d)`

For each temporal/joint scale factor `s ∈ {1, 2, 4}` (default):

1. **Temporal pooling**: average-pool the time axis from `T` to `max(1, T // s)`.
2. **Joint pooling**: average-pool the joint axis from `J` to `max(1, J // s)`.
3. **Cross-view attention**: run a lightweight transformer encoder layer over the `V` views at the coarser `(T//s, J//s)` resolution.
4. **Upsample**: linearly upsample joints back to `J` and time back to `T`.

The multi-scale branch outputs are concatenated and projected back to `d` with a residual connection.  The module therefore:

- Captures **long-range temporal context** at coarse scales (e.g. limb motion over many frames).
- Captures **coarse skeleton grouping** at reduced joint resolution (e.g. torso vs. limbs).
- Preserves fine-grained detail through the full-resolution branch (`s = 1`) and the residual connection.

This differs from `CrossViewSpatialPyramid`, which is purely spatial (joint) and per-frame; v3 fuses both temporal and spatial scales end-to-end.

### 2.2 Camera-Conditioned / Epipolar-Biased Cross-View Attention

**Camera conditioning** (`_CameraConditioning`):  
Flatten `(K, R, t)` per view, encode them through a small MLP, and add the resulting per-view embedding to the feature tokens.  The camera embedding is broadcast across joints, giving the attention layers direct access to intrinsics and extrinsics without increasing token sequence length.

**Epipolar-biased transformer** (`_CameraConditionedEpipolarBias` + `EpipolarBiasedTransformerEncoderLayer`):  
From `(K, R, t)` and the observed 2-D points, compute per-frame pairwise epipolar distance between views using the existing `compute_per_frame_epipolar_bias`.  The distance is negated and scaled to form an additive attention bias, then lifted to a full `T×V` spatio-temporal mask using `build_temporal_bias_from_frames` so that only view pairs within the same timestep receive the bias.

The standard `nn.TransformerEncoderLayer` in the v2 ST block is replaced with `EpipolarBiasedTransformerEncoderLayer` when `use_epipolar_bias=True`.  The bias is added to the raw attention scores, encouraging the model to:

- Attend more strongly to geometrically consistent view pairs.
- Down-weight view pairs with large epipolar error (potentially due to mis-calibration or occlusion).

Both components are optional and independently toggled via flags.

## 3. Differences from v2

| Aspect | v2 | v3 |
|--------|----|----|
| Fusion scale | Single `(T, V)` ST transformer | Hierarchical `s = 1, 2, 4` temporal / joint / cross-view fusion |
| Camera geometry in attention | None | Camera conditioning + epipolar attention bias |
| Cross-view attention | Inside flat ST transformer | Explicit multi-scale cross-view blocks before ST transformer |
| Output arity | Same as v2 when `return_covariance=True` | Same as v2 |
| Constructor flags | `graph_num_layers`, `visibility_threshold`, ... | All v2 flags + `use_multiscale_fusion`, `use_camera_conditioning`, `use_epipolar_bias`, `multiscale_scales`, `camera_condition_dim`, `epipolar_temperature` |

v3 subclasses `OmniMultiViewFusionV2` so all v2 components (visibility head, graph-joint attention, Bayesian triangulation, residual refiner) are preserved without modification.

## 4. Expected Advantages

1. **Better long-range temporal reasoning** – coarse temporal scales allow the model to aggregate motion context over longer windows without increasing sequence length or computation in the full-resolution ST transformer.
2. **Scale-aware skeleton reasoning** – coarse joint scales encourage limb-level / body-part-level feature sharing before fine joint refinement.
3. **Geometry-regularized attention** – the epipolar bias provides an inductive bias for calibrated multi-view geometry, reducing the data required to learn view consistency.
4. **Calibration-aware features** – camera conditioning makes per-view tokens aware of viewpoint and intrinsics, which should improve view weighting and residual refinement under varying rigs.
5. **Modular ablation** – each new component can be toggled off with a single boolean flag, making it easy to measure marginal contribution.

## 5. Proposed Ablations

Recommended ablation table for a future paper:

| Run | Multi-scale fusion | Camera conditioning | Epipolar bias | Purpose |
|-----|-------------------|---------------------|---------------|---------|
| A (v3 full) | ✓ | ✓ | ✓ | Final model |
| B | ✗ | ✓ | ✓ | Measure impact of multi-scale fusion |
| C | ✓ | ✗ | ✓ | Measure impact of camera conditioning |
| D | ✓ | ✓ | ✗ | Measure impact of epipolar bias |
| E (v2 baseline) | ✗ | ✗ | ✗ | Reproduce v2 performance |

Additional ablations to consider:

- **Scale factors**: `{1}`, `{1, 2}`, `{1, 2, 4}`, `{1, 2, 4, 8}` to find the optimal multi-scale schedule.
- **Epipolar temperature**: sweep `{1, 5, 10, 20}` to control the strength of the geometry bias.
- **Camera condition dimension**: `{0, 16, 32, 64}` to test calibration feature capacity.
- **Joint graph vs. multi-scale fusion**: disable `graph_num_layers=0` while keeping multi-scale fusion to see if the new component replaces the need for explicit skeleton graphs.

## 6. Training & Evaluation Plan

- **Dataset**: MPI-INF-3DHP v14 multiview (`experiments/train_omniview_fusion_v3_mpiinf3dhp.py`).
- **Loss mix** (same as v2 trainer): 3D MPJPE, visibility BCE, uncertainty NLL, temporal consistency, bone-length regularisation, plus the model's own epipolar consistency auxiliary loss.
- **Warm-start**: v3 can load a v2 checkpoint with `strict=False` because all v2 parameter names are preserved.
- **Metrics**: MPJPE (mm), PA-MPJPE, per-joint errors, visibility accuracy, epipolar loss magnitude.

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Multi-scale pooling can blur fine joint detail | Residual connection + full-resolution `s=1` branch |
| Epipolar bias assumes perfect calibration | Camera conditioning allows the model to learn robustness; bias is soft, not a hard mask |
| Added compute / memory | Scales are small (1, 2, 4); only lightweight transformer layers used per scale |
| Warm-start compatibility | v3 keeps v2 parameter names and adds only new optional branches |

## 8. File Checklist

- `motionflow_mv/fusion/omniview_fusion_v3.py` – model implementation with `__main__` smoke test.
- `experiments/train_omniview_fusion_v3_mpiinf3dhp.py` – minimal training script adapted from the v2 trainer.
- `docs/omniview_fusion_v3_design.md` – this document.
