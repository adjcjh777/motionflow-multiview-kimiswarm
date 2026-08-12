# Direction 10: Graph neural networks for skeleton fusion

## Problem statement

The current best model (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`) uses dense self-attention over joints in the per-frame encoder.  This is agnostic to the kinematic skeleton: an elbow can attend to a knee just as easily as to the wrist, and the model must learn anatomical priors from data alone.  Replacing dense joint attention with a sparse graph that encodes parent-child bone edges, left-right symmetry, and same-joint cross-view edges should inject anatomy-aware inductive bias, reduce the number of spurious attention edges, and potentially improve distal-joint accuracy with fewer parameters.  The codebase already contains two `GraphJointRelation` implementations (`motionflow_mv/models/graph_joint_relation.py` and `motionflow_mv/fusion/graph_joint_relation.py`), but neither has been wired into the current best PP pipeline.

## Simplest concrete next experiment

Create a drop-in GNN variant of the best PP model by subclassing `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` and overriding its per-frame `_extract_frame_features` so that the `joint_attn` transformer layers are replaced by one or more `GraphJointRelation` layers over the (view, joint) skeleton graph.  Warm-start the rest of the network from the currently training PP curriculum checkpoint and run a short 20-epoch smoke on MPI-INF-3DHP to compare clean MPJPE/PA-MPJPE against the baseline.  If the smoke is promising, add 17-joint H36M and variable-view ablations.

## Files to touch / rough diff

### New files

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_graph_model.py`
  - New model class `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph`.
  - Subclasses the best PP model.
  - Overrides `_extract_frame_features` to replace `joint_attn` with `GraphJointRelation`.
  - Builds a static `(view, joint)` edge index from the skeleton parents/symmetry lists.
  - Supports 28-joint (MPI-INF-3DHP) and 17-joint (H36M) skeletons.
- `tests/test_graph_joint_relation_pp.py`
  - CPU-only forward-shape and gradient sanity tests for 17-joint and 28-joint inputs.
- `scripts/run_crossview_residual_principal_point_graph_wsl.sh`
  - GPU training launcher that warm-starts from the best PP checkpoint.

### Key sketch of the change

```python
# In _extract_frame_features:
# OLD:
# feat_j = feat_v.permute(0, 2, 1, 3).reshape(N * V, J, self.d)
# for layer in self.joint_attn:
#     feat_j = layer(feat_j)
# feat_j = feat_j.view(N, V, J, self.d)

# NEW:
feat_j = feat_v.permute(0, 2, 1, 3).reshape(N, V, J, self.d).contiguous()
for graph_layer in self.joint_graph:
    feat_j = graph_layer(feat_j, self.edge_index, self.edge_type)
feat_j = feat_j.view(N, V, J, self.d)
```

No existing experiment runner is modified; only new files are added, so currently running jobs are unaffected.

## Expected success metric

- CPU-only sanity: forward pass and backward gradients succeed for both 17-joint and 28-joint inputs.
- GPU smoke (when the RTX 4090 is free): MPI-INF-3DHP clean MPJPE ≤ 9.6 mm and PA-MPJPE ≤ 5.7 mm after 20 epochs of warm-start training.  A larger gain would be a clear improvement over the dense-attention baseline; if the smoke fails to reach within 3% of the baseline, the direction should be deprioritized.

## Resource requirement

- Model/code creation and sanity test: **CPU-only, completed in this step**.
- Full warm-start training: **GPU required**; the launcher is provided but not executed because the WSL RTX 4090 is currently training the cross-view PP curriculum and A800-D is read-only.

## Verification (CPU-only)

Run the CPU-only test that checks forward pass, output shapes, and gradient flow:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_graph_joint_relation_pp.py
```

Result:

```text
GraphJointRelation PP tests passed
```

All four test cases passed: 28-joint forward, 17-joint forward, single-frame 4D input, and shared-weight vs. separate-layer graph variants.

## Next action

Queue the launcher `scripts/run_crossview_residual_principal_point_graph_wsl.sh` after the currently running PP curriculum finishes.  Do not execute it now.

## Commit status

Files were added to the local branch (`multiview-residual-exploration`) and committed.  The working tree is clean for these files.  Push was not attempted per instructions; the commit remains local.

## Notes / follow-up

- The existing `GraphJointRelation` in `motionflow_mv/fusion/graph_joint_relation.py` uses an edge-conditioned attention message-passing scheme with three edge types (bone, symmetry, cross-view).  The variant in `motionflow_mv/models/graph_joint_relation.py` uses multi-head dot-product attention with edge-type bias; both are candidates for the ablation.  The current skeleton uses the former because it is already imported by the fusion package.
- Variable-view inference is not yet handled: the edge index is built for a fixed `n_views`.  For variable views, the graph must be rebuilt per-forward call or padded to `max_views`.
- The graph currently uses a full cross-view clique per joint.  If memory becomes a bottleneck at 14 views, cross-view edges can be pruned to a k-NN or dropped entirely, leaving only intra-view bone/symmetry edges.
