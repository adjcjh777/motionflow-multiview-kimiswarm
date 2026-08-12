# v33 Direction: View-Joint Graph Network for Multi-View Pose Fusion

**Direction slug:** `view_joint_graph_network`  
**Title:** Graph neural network over views and joints  
**Target venue:** ICRA/CVPR 2027  
**Date:** 2026-08-08  
**Status:** Design proposal — ready for smoke implementation  

## 1. Problem statement and motivation

The current best model, `OmniMultiViewFusionV5`, fuses multi-view evidence through a stack of largely *separable* operations: camera-conditioned view embedding, set-transformer / Perceiver view aggregation, spatio-temporal (ST) transformers, and per-joint cross-view attention in `v25`/`v31`. While these blocks handle view-invariance and temporal consistency well, they do not explicitly model the **joint-relational structure** (skeleton, symmetry) **and** the **cross-view geometry** at the same time.

Recent iterations (v25 geometry fusion, v31 hierarchical encoder) operate attention over `(view, joint)` tokens, but they do so with dense or hand-grouped attention. A **graph neural network (GNN) over the `(view, joint)` product space** can:

1. **Encode skeleton constraints directly**: bone (parent→child), symmetry (left↔right), and self-loops per view.
2. **Encode cross-view consistency explicitly**: same-joint edges across views encourage multi-view agreement.
3. **Provide permutation-invariance over views** while preserving calibrated-camera geometry.
4. **Complement variable-view training**: a static sparse graph naturally handles missing views via masking and remains valid when `k < V`.

Prior work in the repo already built the pieces (`GraphJointRelation`, `CrossViewGraphAttention` in `motionflow_mv/fusion/prototypes/cross_view_graph_attention.py`, and `HierarchicalViewEncoderV31`), but none are wired as a first-class citizen in the v5 training pipeline. This proposal unifies them into a single **View-Joint Graph Network (VJGN)** block.

## 2. Proposed architecture changes

### 2.1 New module

`motionflow_mv/fusion/view_joint_graph_network_v33.py`

```text
ViewJointGraphNetworkV33
├── Graph construction (once per skeleton / view count)
│   └── Nodes: (view, joint)
│   └── Edge types:
│       0  bone         – parent↔child within each view
│       1  symmetry     – left↔right symmetric joints within each view
│       2  cross-view  – same joint across view pairs
│       3  self-loop    – residual identity edges
│       (optional) 4  epipolar/ray – geometry-aware edge weights from v25
├── Embedding layers
│   └── EdgeTypeEmbedding + per-edge scalar bias
├── N message-passing layers
│   └── Multi-head graph attention with edge-type modulated scores
└── LayerNorm + residual after each layer
```

Input: `(B, T, V, J, d)` per-view joint tokens.  
Output: `(B, T, V, J, d)` refined tokens (same shape, residual add).

The block is initialized to be **identity**: final output projection is zeroed so the network is a no-op at the start of training.

### 2.2 Optional geometry-aware edges

When `v33_use_geometry_edges=True`, the GNN also receives a geometry bias term derived from the same machinery used by `v25`/`v31`:

- **Epipolar distance**: lower distance ⇒ stronger edge weight.
- **Ray-intersection logit** (`ray_intersection_logit` in `multiview_geometry_fusion_v25.py`): rays that are closer to intersecting get a stronger edge weight.

This bias is added to the attention score per edge and is gated by a learnable scalar initialized near zero (following the v31 `geometry_gate` pattern).

### 2.3 Integration point in `OmniMultiViewFusionV5`

The VJGN block is inserted **after the per-view feature encoding but before the temporal transformer**, i.e., where tokens are `(B, T, V, J, d)`:

```python
# pseudo-forward inside OmniMultiViewFusionV5
view_joint_tokens = per_view_joint_features  # (B, T, V, J, d)
if self.use_view_joint_graph_network_v33:
    view_joint_tokens = self.view_joint_graph_network_v33(
        view_joint_tokens,
        points_2d=points_2d,
        K=K, R=R, t=t,
        view_mask=view_mask,
    )
# Continue to ST / joint transformers as before
```

It can also be inserted **after the v25/v31 geometry blocks** as a refinement stage. The smoke ablation should test both placements.

### 2.4 New CLI flags

Add to `experiments/train_omniview_fusion_v5_webbridge_multi.py`:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_view_joint_graph_network_v33` | bool | `False` | Enable the VJGN block |
| `--v33_gnn_num_layers` | int | `2` | Number of message-passing layers |
| `--v33_gnn_num_heads` | int | `4` | Attention heads per GNN layer |
| `--v33_gnn_dropout` | float | `0.1` | Dropout on attention weights |
| `--v33_use_geometry_edges` | bool | `False` | Add epipolar/ray edge bias |
| `--v33_gnn_placement` | str | `"before_st"` | `"before_st"` or `"after_geometry"` |
| `--v33_gnn_edge_hidden` | int | `32` | Hidden dim for edge-type MLP bias |

### 2.5 Data / preprocessing requirements

No new dataset is required. The existing WebBridge mixed loader already returns `(B, T, V, J, 3)` observations plus calibrated cameras. The VJGN block reuses the skeleton parent lists already defined in:

- `motionflow_mv/fusion/prototypes/cross_view_graph_attention.py` (`H36M_17_PARENTS`, `MPI_INF_3DHP_28_PARENTS`)
- `motionflow_mv/models/graph_joint_relation.py`

For variable-view training, masked-out views are handled by zeroing out their tokens before the GNN and by excluding them from cross-view edges (the graph is built for `n_views` but edge weights / attention masking respect `view_mask`).

## 3. Training command / ablation flags

### Smoke test (CPU, 1 epoch)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --use_view_joint_graph_network_v33 \
  --v33_gnn_num_layers 1 \
  --v33_gnn_num_heads 2 \
  --v33_use_geometry_edges
```

### Full WebBridge + H36M + MPI mixed run

```bash
python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
  --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding \
  --use_deformable_cross_view_attention_v18 \
  --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
  --v25_use_geometry_attention --v25_use_learned_depth_triangulation \
  --use_hierarchical_multiview_v31 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 \
  --use_view_joint_graph_network_v33 \
  --v33_gnn_num_layers 2 --v33_gnn_num_heads 4 --v33_gnn_dropout 0.1 \
  --v33_use_geometry_edges \
  --d 64 --residual_hidden 128 --n_st_layers 2 --n_joint_layers 1 --n_heads 4 \
  --clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 \
  --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 \
  --early_stopping_patience 5 --early_stopping_min_delta 0.001 \
  --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
  --outlier_view_prob 0.3 --outlier_view_max_views 1 \
  --output outputs/omniview_fusion_v33_view_joint_graph_network.pth
```

### Ablations to run

1. **VJGN only** (no geometry edges) → isolate skeleton/cross-view benefit.
2. **VJGN + geometry edges** → full proposed variant.
3. **Placement after v25 geometry fusion** vs. before ST transformer.
4. **Number of layers** (`1, 2, 3`) for cost/accuracy trade-off.

## 4. Expected metrics and baseline to beat

### Primary metric

- **val_MPJPE** on the mixed WebBridge/H36M/MPI validation set.

### Baselines

| Run | Expected target | Notes |
|-----|-----------------|-------|
| v32 combined (domain-aware + trajectory consistency) | ~20–25 mm | Current v32 best target from `launch_v32_a800_queue.py` |
| v31 geometry-attention | ~21–28 mm | A800 v31 top-5 baseline |
| v25 geometry fusion only | ~28–40 mm | Legacy baseline depending on config |

### Targets for v33

| Metric | Smoke | Full run target |
|--------|-------|-----------------|
| val_MPJPE | passes CPU smoke | ≤ v32 combined baseline |
| Variable-view k=2..14 | monotonic decreasing | no regression vs. full-view |
| Outlier-view robustness (outlier_view_prob=0.3) | — | ≤ 5% MPJPE degradation |
| PA-MPJPE | — | ≤ baseline + 1 mm |

Success is defined as **matching or beating the v32 combined baseline while improving the variable-view monotonicity curve**. A 1–2 mm improvement on clean MPJPE would be a strong signal that explicit graph structure helps.

## 5. Risks / unknowns

1. **Scalability with V=14, J=28**: the `(view, joint)` graph has `V·J` nodes and `O(V²·J)` cross-view edges. For MPI (`V=14`, `J=28`), that is ~392 nodes and ~10k edges per sample. While still sparse, memory and runtime must be benchmarked on the RTX 4090 smoke first.
2. **Variable-view masking**: cross-view edges involving a masked view need correct attention masking. The current prototype `CrossViewGraphAttention` builds a static edge list; dynamic masking must be added.
3. **Interaction with v31 hierarchical encoder**: both operate on `(view, joint)` tokens. Adding VJGN before or after the hierarchical encoder may be redundant unless they serve different roles (GNN for structure, transformer for global context). The ablation on placement is critical.
4. **Warm-start compatibility**: a new block introduces new parameters, so warm-starting from a v32 checkpoint requires the existing `load_warm_start` / `freeze_old_params` helpers to be updated (the v5 trainer already supports this, but the new prefix must be added to `freeze_old_params`).
5. **Geometry-edge overhead**: computing epipolar distance and ray-intersection logits at every layer adds cost. Caching per-frame geometry (as v31 does) is necessary for efficiency.
6. **Limited skeleton support**: the graph construction currently supports 17- and 28-joint skeletons only. If the project expands to other skeletons, the builder needs a fallback chain definition.

## 6. Implementation sketch

```python
# motionflow_mv/fusion/view_joint_graph_network_v33.py
class ViewJointGraphNetworkV33(nn.Module):
    def __init__(self, d, n_views, n_layers, n_heads, dropout, use_geometry_edges):
        ...

    def forward(self, tokens, points_2d, K, R, t, view_mask=None):
        # tokens: (B, T, V, J, d)
        # Build / cache edge_index, edge_type for (V, J)
        # Optional: compute geometry bias from cameras + 2D points
        # Run N graph-attention layers
        return refined_tokens  # (B, T, V, J, d)
```

Wiring in `motionflow_mv/fusion/omniview_fusion_v5.py`:

- Add constructor flag `use_view_joint_graph_network_v33` and instantiate the module.
- Pass `points_2d`, `K`, `R`, `t`, and `view_mask` into the block during `forward`.
- Update `build_model_from_args` in `experiments/train_omniview_fusion_v5_webbridge_multi.py` to forward the new CLI flags.

No existing source files need to be modified for this *design* document; the actual code change is left to the implementation phase.

## 7. Evaluation protocol

1. **Smoke**: CPU 1-epoch run with `--smoke`.
2. **RTX 4090 fast ablation**: 5–10 epoch runs on the mixed loader with `train_samples=500` to pick placement and layer count.
3. **A800 full run**: 20 epochs, `train_samples=1000`, compare against v32 combined baseline.
4. **Variable-view curve**: evaluate at `k = 2, 4, 6, 8, 10, 12, 14` and report MPJPE vs. number of views.
5. **Robustness matrix**: outlier-view injection, occlusion augmentation, and camera perturbation (reuse existing augmenters in the trainer).

## 8. Related files

- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `motionflow_mv/fusion/prototypes/cross_view_graph_attention.py`
- `motionflow_mv/models/graph_joint_relation.py`
- `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`
- `motionflow_mv/fusion/hierarchical_multiview_v31.py`
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`
- `scripts/launch_v32_a800_queue.py`
