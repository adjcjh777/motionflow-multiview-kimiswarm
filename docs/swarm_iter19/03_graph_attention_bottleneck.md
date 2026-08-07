# 03 Graph-Attention Bottleneck — Swarm Iter 19

## Summary

This report examines whether the graph-joint attention block in `OmniMultiViewFusionV2` is a bottleneck for the ICRA/CVPR 2027 push. It compares the graph-enabled full run with the no-graph ablation currently training, inspects the implementation, and lists concrete next steps to resolve the question empirically.

## Current state

- `OmniMultiViewFusionV2` is implemented in `motionflow_mv/fusion/omniview_fusion_v2.py`. It inserts a graph-joint attention block after the per-frame encoder (line 281) and before the spatiotemporal transformer.
- The block is an `CrossViewGraphAttention` from `motionflow_mv/fusion/prototypes/cross_view_graph_attention.py` (imported at lines 40–46 of `omniview_fusion_v2.py`).
- The number of layers is controlled by `graph_num_layers`; setting it to `0` disables the graph layers but still instantiates the module with an zero-layer `ModuleList`.
- A no-graph ablation is running via `scripts/run_omniview_fusion_v2_full_wsl.sh` with `--graph_num_layers 0`; its checkpoint target is `outputs/omniview_fusion_v2_d128_no_graph.pth` and its log is `outputs/omniview_fusion_v2_d128_no_graph.log`.
- The graph-enabled d=128 run has 1,036,635 parameters (`outputs/omniview_fusion_v2_d128.log:2`); the no-graph run has 969,803 parameters (`outputs/omniview_fusion_v2_d128_no_graph.log:2`) — a 66,832-parameter difference.
- After the 5-epoch freeze phase, the graph-enabled run reached ~25 mm val MPJPE (`outputs/omniview_fusion_v2_d128.log:7–9`), while the no-graph run reached 44.37 mm (`outputs/omniview_fusion_v2_d128_no_graph.log:7–11`). The no-graph run has just unfreezing the full model and is still in progress.

## Key findings

1. **The model is using the older graph prototype, not the hardened v2.**
   - `omniview_fusion_v2.py` calls `CrossViewGraphAttention`, whose `_scatter_softmax` relies on the beta `index_reduce_` API and prints a warning on every forward pass (`motionflow_mv/fusion/prototypes/cross_view_graph_attention.py:54`).
   - A newer `GraphJointAttentionV2` already exists in `motionflow_mv/fusion/graph_joint_attention_v2.py` (lines 266–372) and adds an optional point-wise FFN and a more robust `scatter_reduce_`-based softmax (lines 108–160), but it is not wired into the model.

2. **Graph size scales quadratically with the number of views.**
   - For the H36M 17-joint skeleton: 448 directed edges at 4 views, 1,440 at 8 views, and 3,948 at 14 views.
   - For the MPI-INF-3DHP 28-joint skeleton: 744 directed edges at 4 views, 2,384 at 8 views, and 6,524 at 14 views.
   - Cross-view same-joint edges dominate these counts (`build_graph_joint_edge_index`, lines 94–101 of `graph_joint_attention_v2.py`), so cost grows faster than the rest of the model as `V` increases.

3. **The no-graph ablation is not a pure graph ablation because of the staged warm-start.**
   - `experiments/train_omniview_fusion_v2_mpiinf3dhp.py:505–510` freezes everything except `graph_joint_attention` and `visibility_head` during the warm-start phase.
   - With `graph_num_layers=0`, only the visibility head is trainable during the freeze, so the no-graph run loses an early training signal that the graph-enabled run receives. Its higher early MPJPE may reflect this, not a direct graph benefit.

4. **No timing or memory evidence has been collected yet.**
   - The logs do not record per-epoch wall time or peak GPU memory, so we cannot yet say whether graph attention is a throughput or memory bottleneck.

## Recommendations

1. **Replace `CrossViewGraphAttention` with `GraphJointAttentionV2`.** The newer module is already smoke-tested (`docs/swarm_iter18/P04_graph_joint_attention_v2.md`) and avoids the beta `index_reduce_` warning.
2. **Make `graph_num_layers=0` a true skip.** If `graph_num_layers == 0`, avoid instantiating the graph module entirely in `omniview_fusion_v2.py` so the no-graph ablation is genuinely graph-free and slightly cheaper to run.
3. **Add timing and memory logging to the trainer.** Record per-epoch wall time and `torch.cuda.max_memory_allocated()` in `OmniMultiViewTrainer` (or the training script) so future ablations can quantify the runtime cost of graph attention.
4. **Run a controlled d=48/10-epoch smoke ablation:** graph=0, graph=1 (old `CrossViewGraphAttention`), and graph=1 (new `GraphJointAttentionV2`). Compare final val MPJPE, training time per epoch, and peak memory on the same GPU.
5. **If graph remains expensive at high view counts**, prune cross-view edges for `V > 4` (e.g., only connect neighboring views, or drop cross-view edges entirely and keep only intra-view bone/symmetry edges).

## Open questions

- Does the old `CrossViewGraphAttention` actually slow down training on the RTX 4090, or is the spatiotemporal transformer the dominant cost?
- Is the early-convergence gap between graph and no-graph runs due to the graph itself or the fact that the no-graph run has only the visibility head trainable during the freeze phase?
- How does the quadratic growth of cross-view edges affect memory and throughput at 14 views, which is common in the WebBridge benchmark?
