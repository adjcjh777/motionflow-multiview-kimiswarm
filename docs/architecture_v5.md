# MotionFlow-MultiView Architecture — OmniMultiViewFusion v5

**Target venues**: ICRA / CVPR 2027
**Status**: Architecture reference for current `main` (commit `5bdd076`)
**Last updated**: 2026-08-07

This document describes the `OmniMultiViewFusionV5` architecture, the data flow from raw multi-view 2-D keypoints to 3-D pose, the loss functions used during training, and the intended extension points for future iterations.

## 1. High-level architecture

`OmniMultiViewFusionV5` is the latest in the `OmniMultiViewFusion` family. It inherits from `OmniMultiViewFusionV4` and adds a small set of optional, independently togglable components. The design goal is to keep the proven v2/v3/v4 inference path intact while making the model permutation-invariant over views and robust to variable view counts.

```text
Input:  x (B, T, V, J, 3) = (u, v, confidence)  +  cameras or (K, R, t)
        |
        v
┌─────────────────────────────────────────────────────────────────┐
│ 1. Pre-processing                                               │
│    • Principal-point / focal correction                         │
│    • Optional rotation correction (SO(3) residual)              │
└─────────────────────────────────────────────────────────────────┘
        |
        v
┌─────────────────────────────────────────────────────────────────┐
│ 2. Feature extraction (per frame, per view, per joint)          │
│    • Ray-aware embedding from -D points + intrinsics/extrinsics │
│    • Optional dense joint-level self-attention                    │
│    • Graph-joint attention over skeleton topology               │
└─────────────────────────────────────────────────────────────────┘
        |
        v
┌─────────────────────────────────────────────────────────────────┐
│ 3. Geometry-aware conditioning & multi-scale fusion             │
│    • Camera conditioning (K, R, t)                              │
│    • Hierarchical multi-scale temporal/cross-view fusion      │
│    • Optional adaptive scale-selective multi-scale fusion     │
└─────────────────────────────────────────────────────────────────┘
        |
        v
┌─────────────────────────────────────────────────────────────────┐
│ 4. View representation                                          │
│    • Learned positional embedding (fixed view index)            │
│    • Optional camera-conditioned view embedding               │
│    • Optional set-transformer / Perceiver aggregator          │
│    • Optional domain embedding                                  │
└─────────────────────────────────────────────────────────────────
        |
        v
┌─────────────────────────────────────────────────────────────────┐
│ 5. Spatio-temporal (time × view) transformer                   │
│    • Epipolar-biased attention or additive view mask            │
│    • Time + view positional embeddings                          │
└─────────────────────────────────────────────────────────────────┘
        |
        v
┌─────────────────────────────────────────────────────────────────┐
│ 6. Per-view outputs                                            │
│    • Anisotropic 2-D covariance / precision                   │
│    • Visibility gating (context-aware or fallback)              │
│    • Triangulation weights                                      │
│    • Optional adaptive view selection mask                    │
└─────────────────────────────────────────────────────────────────┘
        |
        v
┌─────────────────────────────────────────────────────────────────┐
│ 7. Triangulation & refinement                                   │
│    • Weighted DLT (`triangulate_dlt_batched_lstsq`)             │
│    • Optional full-precision DLT with robust IRLS reweighting │
│    • Adaptive Gauss-Newton refinement                         │
│    • Residual refinement head (dense or skeleton-graph)       │
│    • Optional kinematic-chain final refiner                     │
└─────────────────────────────────────────────────────────────────┘
        |
        v
Output: (pred_3d, weights, visibility, covariance, epipolar_loss, ...)
```

Key files:

- `motionflow_mv/fusion/omniview_fusion_v5.py` — main model
- `motionflow_mv/fusion/omniview_fusion_v4.py` — parent class with v4 toggles
- `motionflow_mv/fusion/omniview_fusion_v3.py` — parent class with multi-scale / epipolar bias
- `motionflow_mv/fusion/omniview_fusion_v2.py` — base visibility / covariance / DLT path
- `motionflow_mv/fusion/fusion_module.py` — plugin interface and DLT baseline
- `motionflow_mv/training/trainer_v2.py` — generic `TrainerV2` with AMP, EMA, cosine LR
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — training entry point
- `motionflow_mv/eval/metrics.py` — evaluation metrics

## 2. Detailed data flow

### 2.1 Input representation

The model accepts either a list of `Camera` objects or explicit tensors `K, R, t`.

- `x`: `(B, T, V, J, 3)` where the last channel is `(u_pixel, v_pixel, confidence)`. A 4-D input `(B, V, J, 3)` is automatically unsqueezed to `T=1`.
- `K`: `(V, 3, 3)` or `(B, V, 3, 3)`
- `R`: `(V, 3, 3)` or `(B, V, 3, 3)`
- `t`: `(V, 3)` or `(B, V, 3)`
- `view_mask`: optional binary mask `(B, T, V)`, `(B, V)`, or `(B*T, V)` to drop missing views.

### 2.2 Camera pre-processing

The first learnable block is `PrincipalPointCorrection` (`motionflow_mv/fusion/principal_point_correction.py`):

```python
K_corrected, pp_delta, focal_scale = self.principal_point_correction(K=K, x=x_flat, weights=confidences)
```

It predicts a small principal-point offset and, when enabled, a focal-length scale. The corrected intrinsics are used for all downstream ray embedding and triangulation. Optionally, a `RotationCorrectionHead` predicts a bounded SO(3) residual per view.

### 2.3 Feature extraction

The per-frame feature extractor (`_extract_frame_features` from v2) embeds each `(u, v, confidence)` joint into a `d`-dimensional token using the corrected intrinsics and extrinsics. After that:

1. **Dense joint attention** (optional): per-view transformer layers over joints.
2. **Graph-joint attention** (`_apply_graph_joint_attention`): message passing over bone, symmetry, and self-loop edges defined in `motionflow_mv/fusion/graph_joint_relation.py`.
3. **Camera conditioning** (`_CameraConditioning`): an MLP encodes flattened `K, R, t` and adds the result to each token.
4. **Multi-scale fusion** (`_HierarchicalMultiscaleFusion`): processes tokens at temporal/joint scales `(1, 2, 4)` and fuses them back with a residual.

### 2.4 View representation and aggregation

v5 introduces three alternatives for handling variable views:

1. **Camera-conditioned view embedding** (`CameraConditionedViewEmbedding`): an MLP of `(K, R, t)` producing a view embedding that is permutation-invariant because it is computed from camera parameters, not view index.
2. **Set-transformer aggregator** (`VariableViewSetAggregator`): ISAB blocks that aggregate views before the spatio-temporal transformer.
3. **Perceiver aggregator** (`PerceiverViewAggregator`): a small Perceiver-style cross-attention over views.

The learned positional embedding `view_pos_embed` is still kept for fixed-view accuracy.

### 2.5 Spatio-temporal transformer

Tokens are arranged as `(B*J, T*V, d)`. Each transformer layer is either a standard `TransformerEncoderLayer` or an geometry-aware `EpipolarBiasedTransformerEncoderLayer`. When epipolar bias is enabled, the cross-view attention scores are biased by per-frame epipolar distances. A `view_mask` is converted into an `(B*J*n_heads, T*V, T*V)` additive mask so dropped views cannot attend or be attended.

### 2.6 Output heads

After the transformer, the model predicts:

- **Covariance**: anisotropic 2-D covariance per `(view, joint)` via Cholesky factor `L`.
- **Visibility**: per-view per-joint visibility probability in `[0, 1]`.
- **Weights**: per-view per-joint triangulation weights from a sigmoid head.
- **Adaptive view selection** (optional): a Gumbel-softmax mask and a `budget_loss`.

The final per-view weights are:

```python
weights = sigmoid(w_logits) * confidences * precision * visibility
```

### 2.7 Triangulation and refinement

1. **Weighted DLT**: `triangulate_dlt_batched_lstsq` solves a batched linear least-squares problem for the initial 3-D points.
2. **Full-precision DLT** (optional): uses the predicted `L` as a precision matrix and applies a robust reweighting + IRLS step for outlier rejection.
3. **Adaptive Gauss-Newton**: refines the DLT result with learned damping.
4. **Residual refinement**: `delta = residual_mlp([feat_pooled, pred_3d_gn])`, optionally a `SkeletonGraphResidualRefiner`.
5. **Kinematic-chain refiner** (optional): final skeleton-aware correction.

The model returns:

```python
(pred_3d, weights, visibility, L, epi_loss, [pp_delta, focal_scale])
```

## 3. Loss functions

The training script (`experiments/train_omniview_fusion_v5_webbridge_multi.py`) combines the following losses:

| Loss | Weight | Description |
|------|--------|-------------|
| 3-D MSE | 1.0 | `F.mse_loss(pred_3d, y)` — main supervision |
| Epipolar consistency | `epipolar_loss_weight` | Auxiliary geometry loss inside the model |
| Visibility BCE | `visibility_loss_weight` | Binary cross-entropy against `(confidence > 0)` |
| Procrustes MSE | `pa_loss_weight` | Procrustes-aligned 3-D MSE |
| Uncertainty NLL | `uncertainty_loss_weight` | 2-D reprojection negative log-likelihood under predicted `L` |
| Temporal consistency | `temporal_loss_weight` | Velocity + acceleration smoothness |
| Bone length | `bone_loss_weight` | MSE between predicted and GT bone vectors |
| Joint limit | `joint_limit_weight` | Hyper-extension penalty |
| Temporal bone length | `temporal_bone_weight` | Bone-length consistency over time |
| Attention entropy | `attention_entropy_weight` | Entropy regularisation on triangulation weights |
| Budget | `budget_loss_weight` | MSE between active view count and `target_k` |
| Reprojection | `reproj_loss_weight` | Robust 2-D reprojection error |
| Aleatoric reprojection | `aleatoric_reproj_loss_weight` | Aleatoric NLL variant of reprojection |
| Monotonic | `monotonic_loss_weight` | Subset-of-views error should not beat full-views |

The `TrainerV2` wrapper (`motionflow_mv/training/trainer_v2.py`) provides:

- cosine learning-rate schedule with linear warmup
- global L2 gradient clipping
- automatic mixed precision (AMP) with CPU-safe fallback
- exponential moving average (EMA) of model parameters

## 4. Evaluation metrics

Defined in `motionflow_mv/eval/metrics.py`:

- **MPJPE**: mean per-joint position error (mm)
- **PA-MPJPE**: Procrustes-aligned MPJPE
- **PCK / AUC**: percentage of correct keypoints and area-under-curve
- **Per-joint MPJPE**: per-joint breakdown
- **Velocity MPJPE**: first-order temporal consistency
- **Bone length error**: mean absolute bone-length error

## 5. Extension points

The v5 model is intentionally modular. New research directions can be inserted at the following points without rewriting the core forward pass:

### 5.1 Alternative view aggregators

The `use_camera_view_embedding`, `use_set_view_aggregator`, and `use_perceiver_aggregator` flags are orthogonal. A new permutation-invariant aggregator (e.g., cross-view transformer) only needs to implement a module with signature:

```python
out = aggregator(feat, view_mask=view_mask)  # (B, T, V, J, d) -> same
```

and be wired into `omniview_fusion_v5.py`.

### 5.2 New ST-transformer biases

`EpipolarBiasedTransformerEncoderLayer` accepts an additive `epipolar_bias` mask. Any module that returns a `(B*J*n_heads, T*V, T*V)` additive mask can be plugged in as an alternative geometry bias (e.g., motion-saliency, bone-length consistency, or scene-floor constraints).

### 5.3 Refinement heads

The residual refinement head is a drop-in replacement. The dense `residual_mlp` can be replaced by any module that consumes `(B*T, J, d+3)` and returns `(B*T, J, 3)`. The optional final `kinematic_refiner` operates purely on `(B*T, J, 3)`.

### 5.4 Robust triangulation backends

The `use_full_precision_dlt` path exposes a clean extension point: the precision matrix and reweighting strategy can be replaced by alternative M-estimators or learned outlier rejection without touching the rest of the model.

### 5.5 Domain / dataset embeddings

`use_domain_embedding` adds a learnable per-dataset embedding. This is useful when mixing H36M, MPI-INF-3DHP, and WebBridge in a single training run. The domain ID is passed directly to `forward` as `domain_id`.

### 5.6 Pipeline integration

For inference, `MultiViewFusionPlugin` (`motionflow_mv/pipeline_multiview_plugin.py`) wraps any registered `FusionModule`. New learned backends can be registered in `FUSION_REGISTRY` and used without changing the production pipeline.

## 6. Running a smoke test

The model ships with an built-in CPU smoke test:

```bash
python motionflow_mv/fusion/omniview_fusion_v5.py
```

A full training smoke test is available via:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py --smoke
```

## 7. Related documents

- `docs/omniview_fusion_v3_design.md`
- `docs/v4_architecture_design_proposal.md`
- `docs/design_omniview_fusion.md`
