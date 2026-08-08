# v33 Design Proposal: Sparse/Deformable Attention for Scalability

**Slug:** `sparse_deformable_attention`  
**Scope:** MotionFlow-MultiView v33 (next-iteration prototype)  
**Target downstream:** ICRA/CVPR 2027 multi-view pose pipeline  

## 1. Problem Statement and Motivation

The v31/v32 model family already has strong cross-view reasoning, but its attention blocks still scale quadratically with the number of views:

*   **v18 deformable cross-view attention** (`motionflow_mv/fusion/deformable_cross_view_attention.py`) is *optional* and only used at one point in the forward path (`omniview_fusion_v5.py`, lines 822–831). The rest of the encoder still pays attention to all `V` views.
*   **v31 hierarchical multi-view encoder** (`motionflow_mv/fusion/hierarchical_multiview_v31.py`) runs dense cross-view attention at joint, part, and body scales, i.e. `O(V²)` per head, per joint group. This becomes expensive when MPI-INF-3DHP uses 14 real views and the tensor is padded to 14.
*   **v32 variable-view training** (`--use_variable_view_training`, `--variable_view_max_views 14`) exposes the model to subsets of 2–14 views, but the attention *mechanism* itself is dense; only the masks change. Memory and FLOPs therefore remain dominated by the worst-case `V=14`.
*   Future multi-view datasets (panoramic rigs, moving cameras, dense arrays) may push `V` to 32 or more. Dense attention is no longer tenable.

**Goal for v33:** make cross-view attention *truly sparse and deformable* across all relevant blocks (hierarchical encoder and ST transformer), so the active attention cost is `O(V·k)` with `k << V`, while preserving or improving pose accuracy and robustness under variable-view inference.

## 2. Proposed Architecture Changes

### 2.1 New module: `SparseDeformableAttentionV33`

Location: `motionflow_mv/fusion/sparse_deformable_attention_v33.py`

A generalised sparse attention block that can replace any dense cross-view attention operation. It combines the v18 epipolar-aware sampler with a learned per-query sparsity pattern and a geometry gate.

```python
class SparseDeformableAttentionV33(nn.Module):
    """Sparse cross-view attention with learned, geometry-guided sampling.

    Inputs
    ------
    x : (B, T, V, J, d)
    K, R, t : cameras
    points_2d : (B, T, V, J, 2)
    view_mask : (B, T, V)

    Outputs
    -------
    out : (B, T, V, J, d)
    """
```

Key components:

| Component | Role |
|-----------|------|
| `geometry_sampler` | Computes per-query-key epipolar/ray distances and selects the top-`k` candidate key views, exactly like v18 but with a tunable `k`. |
| `content_sampler` | A lightweight content-based selector (1×1 conv over `d`) that refines the geometry proposal or overrides it for occluded views. |
| `deformable_offset_mlp` | Learns per-query *offset* logits over the sampled neighbourhood, giving the block a deformable receptive field (analogous to Deformable DETR). |
| `adaptive_k_gate` | A view-count gate that modulates `k` per sample based on pooled feature entropy, so hard samples keep more views and easy samples discard more. |

The module is designed as a *drop-in replacement* for `nn.MultiheadAttention` when the sequence dimension is views. It returns the same shape as the dense block and is identity-initialized (zero residual scale, zero output projection) so it can be warm-started from v31/v32 checkpoints.

### 2.2 Replace dense cross-view blocks with the sparse module

#### (a) v31 hierarchical encoder

In `motionflow_mv/fusion/hierarchical_multiview_v31.py`:

*   `_GeometryBiasedCrossViewAttentionBlock` currently calls `nn.MultiheadAttention` directly (lines 88–96, 117–133).
*   Add an constructor flag `use_sparse_deformable_v33: bool = False` and, when True, instantiate `SparseDeformableAttentionV33` for each cross-view attention layer instead of `nn.MultiheadAttention`.
*   The existing geometry bias (epipolar + ray intersection, lines 287–319) is *reused* to seed the sampler, so the sparse module does not need to recompute it.

#### (b) v18 integration point

In `motionflow_mv/fusion/omniview_fusion_v5.py`:

*   The current v18 block (lines 360–373, 822–831) uses a fixed `n_samples=max(2, n_views // 2)`. With v33, the v18 block can be replaced/augmented by `SparseDeformableAttentionV33` that supports a *dynamic* `k` per sample and per layer.

#### (c) ST transformer (optional)

In `motionflow_mv/fusion/omniview_fusion_v5.py`:

*   The spatio-temporal transformer attends over `T·V` tokens (line 856: `B*J, T*V, d`). When `T` and `V` grow, this becomes the memory bottleneck.
*   A second v33 block can optionally replace the dense view-to-view attention inside the ST transformer by reshaping to view tokens and applying sparse cross-view attention, keeping temporal attention dense. This is a lower-risk first extension after the hierarchical encoder is proven.

### 2.3 Model integration flags in `OmniMultiViewFusionV5`

Add to `motionflow_mv/fusion/omniview_fusion_v5.py` constructor signature:

```python
use_sparse_deformable_v33: bool = False,
v33_sparse_topk: int = 4,                    # fixed top-k if adaptive_k disabled
v33_sparse_use_adaptive_k: bool = False,       # vary k per sample
v33_sparse_min_k: int = 2,
v33_sparse_max_k: int = 8,
v33_sparse_apply_hierarchical: bool = True,  # replace v31 cross-view blocks
v33_sparse_apply_st: bool = False,           # replace ST view attention
v33_sparse_temperature: float = 10.0,        # epipolar sampler temperature
v33_sparse_dropout: float = 0.1,
```

When `use_sparse_deformable_v33` is True and `v33_sparse_apply_hierarchical` is True, pass the flag into `HierarchicalViewEncoderV31`. When `v33_sparse_apply_st` is True, instantiate a `SparseDeformableSTBlockV33` wrapper around the ST transformer layers.

## 3. Training Command / Ablation Flags

### Recommended smoke test (RTX 4090)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
  --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding \
  --use_deformable_cross_view_attention_v18 \
  --use_hierarchical_multiview_v31 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 \
  --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
  --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
  --use_sparse_deformable_v33 \
  --v33_sparse_topk 4 \
  --v33_sparse_use_adaptive_k \
  --v33_sparse_apply_hierarchical \
  --d 64 --residual_hidden 128 --n_st_layers 2 --graph_num_layers 1 \
  --epochs 5 --batch_size 8 --train_samples 1000 --val_stride 10 \
  --output outputs/omniview_fusion_v33_sparse_deformable_smoke.pth
```

### Ablation matrix

| Run | Flags | Purpose |
|-----|-------|---------|
| `v33_baseline` | v31 hierarchical + v18 deformable (current v32 best practice) | Baseline for comparison. |
| `v33_sparse_hierarchical` | baseline + `--use_sparse_deformable_v33 --v33_sparse_apply_hierarchical --v33_sparse_topk 4` | Replace dense hierarchical cross-view blocks with sparse/deformable ones. |
| `v33_sparse_adaptive_k` | v33_sparse_hierarchical + `--v33_sparse_use_adaptive_k` | Let each sample choose its own `k`. |
| `v33_sparse_st` | v33_sparse_hierarchical + `--v33_sparse_apply_st --v33_sparse_topk 4` | Also sparsify the ST transformer's view dimension. |
| `v33_sparse_small_k` | v33_sparse_hierarchical + `--v33_sparse_topk 2` | Measure accuracy/cost trade-off at very low sparsity. |

## 4. Expected Metrics and Baseline to Beat

### Primary metrics

| Metric | How computed | Target |
|--------|--------------|--------|
| `val_MPJPE` | `eval_metric()` in `train_omniview_fusion_v5_webbridge_multi.py` (mm) | **Match or beat v32 baseline** (recorded v32 smoke/full runs ~20–40 mm smoke; full A800 target < 28 mm). |
| `val_MPJPE` at fixed view counts | Evaluate with fixed `view_mask` for `k ∈ {2,4,8,14}` | Within 5% of the dense v32 baseline for `k ≥ 4`; equal or better at `k=2`. |
| `mean_attention_flops` | Profile the hierarchical encoder + ST transformer | ≥30% reduction vs. dense baseline at `V=14`, measured by `torch.utils.flop_counter` or custom hook. |
| `peak_gpu_memory_mb` | RTX 4090 smoke and A800 full | ≤90% of dense baseline at same batch size. |

### Robustness metrics

*   **Variable-view monotonicity:** With the same `--monotonic_loss_weight 0.1`, a subset of views should not outperform the full set. Sparse/deformable attention should make this more stable because it explicitly reasons about a small subset of views.
*   **Outlier-view robustness:** The existing `--outlier_view_prob 0.3` augmentation tests whether the model can down-weight corrupted views. Sparse sampling should *improve* this by limiting the influence of a single bad view to only the queries that sampled it.

### Baseline to beat

*   **v32 combined run** (`scripts/launch_v32_a800_queue.py` run `"v32_combined"`): domain-aware curriculum + trajectory consistency.
*   **v31 hierarchical encoder baseline** (`--use_hierarchical_multiview_v31 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1`).

## 5. Risks / Unknowns

1. **Geometry sampler quality.** If the epipolar/ray-based sampler discards informative views (e.g., extreme baseline views that are geometrically distant but content-rich), accuracy may degrade. Mitigation: keep a small soft-attention fallback branch (`content_sampler`) and a residual gate that is identity at init.
2. **Top-k straight-through instability.** v18 already offers a straight-through top-k option (`--deformable_attention_use_topk_st`), but it was not the production default because gradients can be noisy. Mitigation: start with soft sparse attention and anneal to harder top-k; use the existing `deformable_attention_use_topk_st` path for direct comparison.
3. **Adaptive `k` collapse.** The `adaptive_k_gate` might learn to always pick `min_k` to save computation. Mitigation: add an accuracy budget loss (existing `--monotonic_loss_weight`) and clamp `k` per domain (H36M real views = 4, MPI = 14).
4. **Variable-view mask interaction.** With `view_mask` zeroing padded views, the sampler must not select masked views. The existing v18 block already handles `view_mask` (lines 174–178), and v33 will inherit that logic.
5. **ST-transformer complexity.** Sparsifying the view dimension inside `T·V` attention is more invasive than sparsifying the hierarchical encoder. The hierarchical-only path is the safer first milestone; ST sparsity is a follow-up.
6. **Warm-start compatibility.** v33 introduces new sub-modules. Loading a v31/v32 checkpoint into a v33 model should use `strict=False` and zero-initialized residual gates so the new block starts as identity. This matches the existing warm-start convention in `train_omniview_fusion_v5_webbridge_multi.py` (`load_warm_start`).

## 6. Success Criteria (Go/No-Go)

*   Smoke test completes and `val_MPJPE` is within 5% of the v32 baseline on the same smoke config.
*   Hierarchical-only sparse attention shows ≥20% FLOP reduction at `V=14` without accuracy regression on the full validation set.
*   Variable-view robustness curve (MPJPE vs. `k ∈ {2,4,8,14}`) is flat or improved compared to the dense v31/v32 baseline.

## 7. Files Touched (if/when implemented)

*   `motionflow_mv/fusion/sparse_deformable_attention_v33.py` (new)
*   `motionflow_mv/fusion/hierarchical_multiview_v31.py` (add sparse flag + module wiring)
*   `motionflow_mv/fusion/omniview_fusion_v5.py` (add top-level flags + integration)
*   `experiments/train_omniview_fusion_v5_webbridge_multi.py` (add CLI flags in `parse_args()` and pass to `build_model_from_args()`)
*   `docs/swarm_iter_next/v33_sparse_deformable_attention.md` (this proposal)

---

*This proposal does not modify any existing source files; it only describes the intended v33 integration path.*
