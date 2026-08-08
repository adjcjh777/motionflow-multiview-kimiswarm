# v33: Ray-Conditioned Cross-View Attention

**Slug:** `ray_conditioned_attention`  
**Date:** 2026-08-08  
**Target stack:** `OmniMultiViewFusionV5` (WebBridge / H36M / MPI-INF-3DHP mixed training)  
**Baseline to beat:** v31/v32 full stack (`--use_hierarchical_multiview_v31 --v31_geometry_bias --v31_use_ray_attention` plus v32 combined flags), anchor 8.75 mm MPJPE / 4.95 mm PA-MPJPE on MPI-INF-3DHP S2/Seq1.

---

## 1. Problem statement and motivation

The v31 `HierarchicalViewEncoderV31` already injects ray geometry into the
hierarchical view encoder, but it does so in two limited ways:

1. **Geometry bias:** epipolar distance + ray-intersection quality are added as
   an *attention-score bias* in the cross-view blocks.
2. **Ray token embedding:** a small MLP projects per-joint ray
   (camera-centre, direction) features and adds them to the content tokens.

Both are additive. The attention *keys* and *queries* are still derived from
learned content features, so the model can ignore the geometry when it is
hard-to-fit (e.g. under occlusion, calibration noise, or variable view subsets).

**Idea for v33:** make the cross-view attention itself *ray-conditioned* by
letting the per-joint ray embeddings participate directly in the Q/K
computation. Views that are geometrically informative for a given joint should
therefore receive higher attention weights even before any content has been
fully learned, which should improve:

- outlier-view robustness,
- variable-view inference,
- cross-dataset transfer (camera geometry is an explicit, not learned, cue).

---

## 2. Proposed architecture changes

### 2.1 New module

Create `motionflow_mv/fusion/ray_conditioned_attention_v33.py`:

```text
RayConditionedCrossViewAttentionV33
    ├── __init__(d, n_heads, n_layers, dropout, residual_gate_init)
    ├── _build_ray_embedding(points_2d, K, R, t, view_mask) -> (B, T, V, J, d)
    └── forward(tokens, points_2d, K, R, t, view_mask=None) -> (B, T, V, J, d)
```

The module reuses the existing helpers
`motionflow_mv.fusion.multiview_geometry_fusion_v25.compute_rays` and
camera projection to obtain per-view, per-joint ray features.

### 2.2 Ray-conditioned attention

For tokens `X ∈ R^(B×T×V×J×d)` and ray embeddings `R ∈ R^(B×T×V×J×d)`:

```text
Q = W_q_content(X) + W_q_ray(R)
K = W_k_content(X) + W_k_ray(R)
V = W_v_content(X)

attn(Q, K, V) = softmax( (QK^T)/√d + λ·bias_ray ) V
out = X + σ(gate) · LayerNorm( MLP( attn(Q, K, V) ) )
```

- `bias_ray` is the same epipolar/ray-intersection bias already used by v31.
- The gate `σ(gate)` is initialized to near-zero so the block starts as an
  identity mapping and warms up safely.
- Values remain content-only, preserving the learned feature stream.

The module stacks `n_layers` of the block above; each block is residual and
pre-normalized, matching v30/v31 style.

### 2.3 Wiring into `OmniMultiViewFusionV5`

Add to `motionflow_mv/fusion/omniview_fusion_v5.py`:

```python
# New constructor flags
use_ray_conditioned_attention_v33: bool = False,
v33_n_heads: int = 4,
v33_n_layers: int = 2,
v33_dropout: float = 0.1,
v33_use_ray_bias: bool = True,
v33_residual_gate_init: float = -6.0,
```

In `__init__`, instantiate the module only when the flag is set, placed
**after** the v31 hierarchical encoder so it can refine the already
geometry-biased tokens:

```python
if self.use_ray_conditioned_attention_v33:
    self.ray_conditioned_attention_v33 = RayConditionedCrossViewAttentionV33(...)
```

In `forward`, after the v31 block (around the current line 846):

```python
if self.use_ray_conditioned_attention_v33:
    feat = feat + self.ray_conditioned_attention_v33(
        feat,
        points_2d=points_2d.view(B, T, V, J, 2),
        K=K_corrected.view(B, T, V, 3, 3),
        R=R.view(B, T, V, 3, 3),
        t=t.view(B, T, V, 3),
        view_mask=view_mask_flat.view(B, T, V),
    )
```

### 2.4 CLI / training-script flags

In `experiments/train_omniview_fusion_v5_webbridge_multi.py` and
`build_model_from_args`, add:

```bash
--use_ray_conditioned_attention_v33
--v33_n_heads 4
--v33_n_layers 2
--v33_dropout 0.1
--v33_use_ray_bias true
--v33_residual_gate_init -6.0
```

No new loss is required; the module is feature-only and is trained through the
existing 3-D MSE, reprojection, and auxiliary losses.

### 2.5 Ablation hooks

| Flag | Purpose |
|------|---------|
| `--use_ray_conditioned_attention_v33` | Master on/off switch. |
| `--v33_use_ray_bias` | Toggle the additive ray geometry bias (True vs. pure dot-product conditioning). |
| `--v33_n_layers` | 1 (light) vs. 2 (standard). |
| `--v33_n_heads` | 2, 4, or 8 heads. |

---

## 3. Data and preprocessing

No new dataset or preprocessing is needed. The module consumes the same tensors
as v31:

- `points_2d`: `(B, T, V, J, 2)` from the WebBridge/H36M/MPI mixed loader.
- `K, R, t`: camera intrinsics/extrinsics, already provided by the loader.
- `view_mask`: `(B, T, V)` float mask for padded/dropped views; the module must
  zero out masked views before computing ray embeddings and attention.

The existing `compute_rays` helper should be wrapped with a mask-safe call so
that dropped views do not contribute to the ray embedding or the attention
scores.

---

## 4. Training command

Example smoke run on the local RTX 4090 (mirrors the v32 ray-attention smoke
script):

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
    --v25_use_geometry_attention --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_hierarchical_multiview_v31 --v31_geometry_bias --v31_use_ray_attention \
    --use_ray_conditioned_attention_v33 --v33_n_heads 4 --v33_n_layers 2 \
    --v33_use_ray_bias true --v33_dropout 0.1 \
    --use_variable_view_training \
    --variable_view_min_views 2 --variable_view_max_views 14 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 \
    --variable_view_permute \
    --num_workers 4 --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 \
    --val_stride 10 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --early_stopping_patience 5 --early_stopping_min_delta 0.001 \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 3 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/omniview_fusion_v33_ray_conditioned_attention.pth
```

For a full A800 run, mirror `scripts/launch_v32_a800_queue.py` and add the v33
flags to `COMMON_FLAGS` or as an extra run entry.

---

## 5. Expected metrics and baseline to beat

Primary evaluation on the mixed WebBridge/H36M/MPI validation split in
`configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`.

| Metric | Baseline (v31/v32) | v33 target |
|--------|--------------------|------------|
| `val_MPJPE` (mixed) | best v32 combined checkpoint | ≥5% relative improvement |
| `val_MPJPE` MPI-INF-3DHP S2/Seq1 | 8.75 mm anchor | ≤8.3 mm |
| PA-MPJPE MPI-INF-3DHP S2/Seq1 | 4.95 mm anchor | ≤4.7 mm |
| View-drop robustness (k=2..14) | v31/v32 curve | lower mean and std. |
| Outlier-view robustness | v31/v32 curve | lower mean error. |

Secondary: inference latency on `experiments/benchmark_runtime.py` should be
within 15% of the v31/v32 stack; if larger, reduce `v33_n_layers` to 1.

---

## 6. Risks / unknowns

| Risk | Why | Mitigation |
|------|-----|------------|
| **Overfitting** | Ray embeddings add content-geometry interactions that may memorize the training rig. | Start with gate init `-6.0`, use dropout, and smoke with `v33_n_layers=1`. |
| **Memory/compute** | Computing ray embeddings per joint and extra Q/K projections raises cost. | Benchmark with `benchmark_runtime.py`; fall back to `v33_n_layers=1` or `d=32`. |
| **Variable-view masking bugs** | `compute_rays` must not leak information from masked views. | Add a unit test that feeds a zero-masked view and checks attention weights are near zero. |
| **Limited gain over v31 geometry bias** | v31 already uses ray bias; stronger conditioning may not be enough. | Ablation: compare `--v33_use_ray_bias true` vs. false, and vs. disabling v33 entirely. |
| **Calibration perturbation interaction** | Camera augmentation perturbs `K, R, t`; ray embeddings must remain stable. | Reuse the same `K_corrected` path already used by v31. |

---

## 7. Definition of done

- [ ] `motionflow_mv/fusion/ray_conditioned_attention_v33.py` created with forward/backward smoke test.
- [ ] Flags `--use_ray_conditioned_attention_v33` and ablation flags plumbed through `experiments/train_omniview_fusion_v5_webbridge_multi.py` and `build_model_from_args`.
- [ ] Smoke run completes with `val_MPJPE` lower than the corresponding v31-only smoke run.
- [ ] Variable-view and outlier-view robustness at least matches v32 combined.
- [ ] Full run queued on A800 or local RTX 4090 with a clear comparison checkpoint.
