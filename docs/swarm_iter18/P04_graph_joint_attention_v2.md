# P04 Graph-Joint Attention v2

**Branch:** `feat/swarm-iter18-omniview`  
**Author:** Kimi Code subagent  
**Date:** 2026-08-07  
**Status:** Implemented + CPU smoke test passed

## 1. Goal

Provide a publication-ready, skeleton-aware graph-joint attention block that can
replace the dense joint-level self-attention in the ray-attention fusion models.
The block operates over a sparse ``(view, joint)`` graph and combines
anatomical constraints (bone, symmetry) with cross-view consistency.

## 2. What changed

* New module: `motionflow_mv/fusion/graph_joint_attention_v2.py`
  * `GraphJointAttentionLayer` – single multi-head graph attention layer with:
    * per-edge-type embeddings added to source values,
    * per-edge-type per-head scalar bias for attention scores,
    * destination-normalised scatter softmax,
    * residual connection + layer norm,
    * optional point-wise FFN.
  * `GraphJointAttentionV2` – stack of layers with graph caching helpers.
  * `build_graph_joint_edge_index()` – builds the directed
    ``(view, joint)`` graph with four edge categories:
    * bone (parent ↔ child),
    * symmetry (left/right mirror pairs),
    * cross-view (same joint across views),
    * self-loop (identity).
  * Reuses existing skeleton definitions from
    `motionflow_mv/fusion/graph_joint_relation.py`.

* Smoke test: `experiments/prototypes/swarm_iter18/graph_joint_attention_v2_smoke.py`
  * Forward passes for MPI-INF-3DHP 28-joint and H36M 17-joint skeletons.
  * Gradient sanity checks.
  * Edge-count verification.
  * Variable view-count test (2, 3, 4 views).
  * Dropout-path test.

## 3. Verification

```bash
KMP_DUPLICATE_LIB_OK=TRUE python experiments/prototypes/swarm_iter18/graph_joint_attention_v2_smoke.py
```

All tests passed:

```
test_layer_forward_28j passed
test_layer_forward_17j passed
test_stack_forward_with_ffn passed
test_edge_counts passed
test_variable_view_count passed
test_dropout_path passed
All GraphJointAttentionV2 CPU smoke tests passed
```

## 4. Next steps

1. Plug `GraphJointAttentionV2` into the OmniMultiViewFusion v2 prototype
   (`docs/swarm_iter18/P02_omniview_arch.md`).
2. Benchmark against the current `GraphJointRelation` block on a small GPU run.
3. If clean, replace the graph block in the main fusion model for the
   full MPI-INF-3DHP benchmark.
