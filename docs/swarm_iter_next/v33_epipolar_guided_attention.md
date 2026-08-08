# v33: Epipolar-guided Attention and Geometry Bias

**Slug:** `epipolar_guided_attention`  
**Date:** 2026-08-08  
**Target stack:** `OmniMultiViewFusionV5` (WebBridge / H36M / MPI-INF-3DHP mixed training)  
**Baseline to beat:** v32 ray-attention / combined stack (`--use_hierarchical_multiview_v31 --v31_geometry_bias --v31_use_ray_attention` plus v32 flags), anchor ~28–32 mm `val_MPJPE` on the mixed validation split.

---

## 1. Problem statement and motivation

`OmniMultiViewFusionV5` already exploits calibrated geometry in several places, but each existing mechanism has a clear limitation:

1. **v31 geometry bias** (`HierarchicalViewEncoderV31`) adds a hand-crafted bias from epipolar distance and ray-intersection quality to cross-view attention scores. It is **additive and learned only through a single scalar gate**, so the model can ignore or over-rule it when content features are confident but geometrically wrong.
2. **v18 deformable cross-view attention** uses epipolar distance to restrict which view pairs are allowed to attend, but the top-k/sparsity decision is **content-agnostic**: it never asks "given the current feature state, which geometrically consistent views are most informative?"
3. **v5 epipolar-biased ST transformer** injects a per-frame epipolar mask into the spatio-temporal transformer, but the mask is computed once from 2-D keypoints and cameras and is **not adapted to the evolving feature tokens**.

The missing piece is an *explicit, learned epipolar-guided attention* mechanism that:

- computes a geometry-derived attention bias from calibrated cameras,
- gates/modulates it with learned content features,
- can sharpen or suppress the bias via a small number of learnable parameters, and
- remains identity at initialization so it can be warm-started on top of v31/v32 checkpoints.

If this works, we expect better robustness to dropped/occluded views, outlier views, and cross-dataset transfer, because geometry would directly shape the attention field rather than merely nudging it.

---

## 2. Proposed architecture changes

### 2.1 New module

Create `motionflow_mv/fusion/epipolar_guided_attention_v33.py`:

```text
EpipolarGuidedAttentionV33
    ├── __init__(d, n_heads, n_layers, epipolar_temperature, top_k, gate_init, dropout)
    ├── _compute_geometry_bias(K, R, t, points_2d, view_mask) -> (B*T, V, V)
    ├── _content_gate(tokens) -> (B*T*J*n_heads, V, V)
    └── forward(tokens, points_2d, K, R, t, view_mask=None) -> (B, T, V, J, d)
```

The module:

1. Reuses `motionflow_mv.fusion.epipolar_attention_bias.compute_epipolar_distance` and the symmetrization already used by `epipolar_transformer_bias.py` to obtain per-joint, per-view-pair distances `dist ∈ (B*T, V, V, J)`.
2. Aggregates distances into an additive geometry bias:
   ```python
   geom_bias = -dist.mean(dim=-1) / temperature   # (B*T, V, V)
   ```
3. Splits `tokens` per joint and computes a **content-dependent gate**:
   ```python
   gate = sigmoid(MLP(tokens))                  # (B*T*J, V, V) per-head
   guided_bias = gate * geom_bias                 # broadcast over heads
   ```
4. Optionally applies a **learned top-k hardening** (differentiable straight-through or Gumbel-softmax) so the attention mask can become sparse across view pairs, similar to `DeformableCrossViewAttention` but conditioned on both geometry and content.
5. Returns a residual update of the same shape as `tokens`, zero-initialized so the block is an identity mapping at the start of training.

### 2.2 Integration into `OmniMultiViewFusionV5`

Add to `motionflow_mv/fusion/omniview_fusion_v5.py` constructor signature:

```python
use_epipolar_guided_attention_v33: bool = False,
v33_ega_n_layers: int = 1,
v33_ega_n_heads: int = 4,
v33_ega_temperature: float = 10.0,
v33_ega_top_k: int = 0,          # 0 = soft, >0 = straight-through top-k
v33_ega_gate_init: float = -6.0,
v33_ega_dropout: float = 0.1,
v33_ega_loss_weight: float = 0.0,
```

In `__init__`, instantiate the module next to the v31 hierarchical block (around line 489–522):

```python
if self.use_epipolar_guided_attention_v33:
    from motionflow_mv.fusion.epipolar_guided_attention_v33 import EpipolarGuidedAttentionV33
    self.epipolar_guided_attention_v33 = EpipolarGuidedAttentionV33(
        d=self.d,
        n_heads=self.n_heads,
        n_layers=self.v33_ega_n_layers,
        epipolar_temperature=self.v33_ega_temperature,
        top_k=self.v33_ega_top_k,
        gate_init=self.v33_ega_gate_init,
        dropout=self.v33_ega_dropout,
    )
else:
    self.epipolar_guided_attention_v33 = None
```

In `forward`, insert **after** the v31 hierarchical encoder and **before** the ST transformer (around the current line 846):

```python
if self.use_epipolar_guided_attention_v33 and self.epipolar_guided_attention_v33 is not None:
    feat = feat + self.epipolar_guided_attention_v33(
        feat,
        points_2d=points_2d.view(B, T, V, J, 2),
        K=K_corrected.view(B, T, V, 3, 3),
        R=R.view(B, T, V, 3, 3),
        t=t.view(B, T, V, 3),
        view_mask=view_mask_flat.view(B, T, V),
    )
```

If an auxiliary loss is added (e.g., a sparsity or entropy loss on the top-k mask), accumulate it into the existing `epi_loss` bucket near line 1133 of the training script, weighted by `v33_ega_loss_weight`.

### 2.3 CLI / training-script flags

Add to `experiments/train_omniview_fusion_v5_webbridge_multi.py` `parse_args` and forward through `build_model_from_args`:

```bash
--use_epipolar_guided_attention_v33
--v33_ega_n_layers 1
--v33_ega_n_heads 4
--v33_ega_temperature 10.0
--v33_ega_top_k 0
--v33_ega_gate_init -6.0
--v33_ega_dropout 0.1
--v33_ega_loss_weight 0.0
```

### 2.4 Ablation hooks

| Flag | Purpose |
|------|---------|
| `--use_epipolar_guided_attention_v33` | Master on/off switch. |
| `--v33_ega_temperature` | Scaling of the epipolar distance bias. |
| `--v33_ega_top_k` | 0 = soft bias; k > 0 = hard top-k view pairs (straight-through). |
| `--v33_ega_gate_init` | Initial value of the content gate (default -6.0 → near-identity). |
| `--v33_ega_n_layers` | 1 (light) vs. 2 (deeper content gating). |
| `--v33_ega_loss_weight` | Optional auxiliary loss on the attention sparsity/entropy. |

---

## 3. Data and preprocessing

No new dataset or preprocessing is required. The module consumes the same tensors already produced by the WebBridge/H36M/MPI mixed loader (`configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`):

- `points_2d`: `(B, T, V, J, 2)` image keypoints.
- `K, R, t`: calibrated camera intrinsics/extrinsics.
- `view_mask`: `(B, T, V)` float mask for padded/dropped views.

The implementation must zero out masked views in both `geom_bias` and the content gate so that dropped/padded views cannot attend or be attended to, mirroring the masking logic in `hierarchical_multiview_v31.py`.

---

## 4. Training command

Example smoke run on the local RTX 4090, piggy-backing on the v32 ray-attention stack:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
    --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment \
    --use_hierarchical_multiview_v31 --v31_geometry_bias --v31_use_ray_attention \
    --use_epipolar_guided_attention_v33 \
    --v33_ega_n_layers 1 --v33_ega_n_heads 4 \
    --v33_ega_temperature 10.0 --v33_ega_top_k 0 --v33_ega_gate_init -6.0 \
    --v33_ega_dropout 0.1 --v33_ega_loss_weight 0.0 \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --num_workers 4 --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --early_stopping_patience 5 --early_stopping_min_delta 0.001 \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 3 --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/v33_epipolar_guided_attention_smoke.pth
```

For a full A800 run, add a run entry in `scripts/launch_v32_a800_queue.py` (or a new v33 queue script):

```python
(
    "v33_epipolar_guided_attention",
    "--use_epipolar_guided_attention_v33 --v33_ega_n_layers 1 --v33_ega_temperature 10.0 --v33_ega_top_k 0",
    "omniview_fusion_v33_epipolar_guided_attention_a800",
),
```

---

## 5. Expected metrics and baseline to beat

Primary evaluation on the mixed WebBridge/H36M/MPI validation split (`configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`).

| Metric | Baseline (v32 ray-attention / combined) | v33 target |
|--------|-------------------------------------------|------------|
| `val_MPJPE` (mixed) | best v32 checkpoint | ≥ 3–5% relative improvement |
| `val_MPJPE` MPI-INF-3DHP S2/Seq1 | ~8.75 mm anchor | ≤ 8.3 mm |
| PA-MPJPE MPI-INF-3DHP S2/Seq1 | ~4.95 mm anchor | ≤ 4.7 mm |
| Variable-view robustness (k = 2..14) | v32 curve | lower mean and std. across k |
| Outlier-view robustness (1 injected outlier) | v32 curve | lower mean error |
| Calibration-perturbed val_MPJPE | v32 with `cam_aug_schedule` | not degraded |

Secondary: runtime on `experiments/benchmark_runtime.py` should stay within 15% of the v32 stack; if larger, reduce `--v33_ega_n_layers` to 1 or disable top-k hardening.

---

## 6. Risks / unknowns

| Risk | Why | Mitigation |
|------|-----|------------|
| **Limited gain over v31 geometry bias** | The v31 bias already uses epipolar + ray geometry; a learned gate may only reproduce it. | Ablation: disable v31 geometry bias while keeping v33, and vice versa; compare pure-geometry vs. content-gated variants. |
| **Overfitting to training rig geometry** | The content gate can learn dataset-specific attention patterns. | Zero-initialize output projection, use dropout, and smoke with `v33_ega_n_layers=1`. |
| **Top-k straight-through instability** | Hard top-k on view pairs can block gradients or create discontinuities. | Start with soft bias (`v33_ega_top_k=0`); add top-k only after smoke shows stable training. |
| **Variable-view masking bugs** | Masked views must not contribute to `geom_bias` or content gate; otherwise padding leaks information. | Unit test: feed a zero-masked view and assert attention weights for that view are near zero. |
| **Compute / memory overhead** | Per-joint, per-view-pair content gating is `O(V²·J)`; MPI has 14 views. | Benchmark with `benchmark_runtime.py`; fall back to `d=32` or single-layer if needed. |
| **Interaction with v18 deformable attention** | Both modules restrict cross-view attention; stacking them may over-constrain the model. | Ablation: run v33 with and without `--use_deformable_cross_view_attention_v18`. |

---

## 7. Definition of done

- [ ] `motionflow_mv/fusion/epipolar_guided_attention_v33.py` created with a forward/backward smoke test.
- [ ] Flags `--use_epipolar_guided_attention_v33` and ablation flags plumbed through `experiments/train_omniview_fusion_v5_webbridge_multi.py` and `build_model_from_args`.
- [ ] Masking unit test passes for padded/dropped views.
- [ ] Smoke run `val_MPJPE` is lower than the corresponding v32-only smoke run.
- [ ] Variable-view and outlier-view robustness at least matches v32.
- [ ] Full run queued on A800 or local RTX 4090 with a clear comparison checkpoint.
