# 07 Dense Joint Attention Reintegration

## Summary

`OmniMultiViewFusionV2` currently replaces the dense transformer-based joint-level
self-attention (`joint_attn`) used by earlier ray-attention models with a sparse,
skeleton-aware graph-joint attention block. A no-graph ablation
(`--graph_num_layers 0`) is training now to isolate the value of that graph block.
This subtask asks whether the **dense joint attention** should be reintegrated,
either as a drop-in replacement when graph attention is disabled or as an
additional capacity layer alongside the graph block. Reintegration matters
because the dense block was part of the strong 8.35 mm Bayesian Tri v2 ensemble
backbone; its complete removal may trade anatomical plausibility for raw
modeling capacity.

## Current state

- **Graph-only implementation.** `motionflow_mv/fusion/omniview_fusion_v2.py:137-146`
  instantiates `CrossViewGraphAttention` and inserts it between the per-frame
  encoder and the spatiotemporal transformer at lines 280-281.
- **Dense attention removed.** The model passes `n_joint_layers=0` to the
  Bayesian-tri parent, so the `joint_attn` module is never created. Warm-starting
  the no-graph run logs this explicitly:
  ```
  Warm-start: unexpected keys ignored: ['joint_attn.0.self_attn.in_proj_weight', ...]
  ```
  (`outputs/omniview_fusion_v2_d128_no_graph.log`).
- **No hybrid path.** There is no flag to enable both dense and graph attention,
  nor to re-enable dense attention when `graph_num_layers=0`.
- **No-graph run is in-flight.** The current log shows the staged freeze epoch
  finished (val_MPJPE ~44 mm during freeze, expected while encoder/ST-transformer
  are frozen) and the run has just unforozen for end-to-end training. No
  conclusion about graph-attention necessity can be drawn yet.
- **Graph block is well-tested.** `motionflow_mv/fusion/graph_joint_attention_v2.py`
  has CPU smoke tests for H36M 17-joint and MPI-INF-3DHP 28-joint skeletons, and
  `experiments/ablate_graph_joint.py` can ablate bone / symmetry / cross-view edges.

## Key findings

1. **Capacity gap when graph is disabled.** Setting `graph_num_layers=0` leaves a
   direct encoder -> ST-transformer path, removing all joint-level mixing before
   the temporal transformer. The 8.35 mm anchor relied on `joint_attn`
   (`nn.TransformerEncoderLayer`) for per-view joint interaction
   (`motionflow_mv/fusion/ray_attention_temporal_crossview_model.py:77-84` and
   `ray_attention_v2_model.py:48-52`). Removing both dense and graph joint
   attention may under-fit the skeleton topology.

2. **Graph and dense blocks are not mutually exclusive.** The graph block in
   `omniview_fusion_v2.py` is applied as
   ```python
   feat = self._apply_graph_joint_attention(feat, J)
   ```
   A dense `TransformerEncoderLayer` could be inserted either before or after
   this call, or gated by a learned combination weight.

3. **Edge-type bias in graph attention is the only skeleton prior.** The graph
   block encodes bone, symmetry, cross-view, and self-loop edges. If the dense
   block is reintroduced without these constraints, the model regains capacity
   but loses anatomical structure; a hybrid could retain both.

4. **Warm-start compatibility is already handled.** `load_warm_start` in
   `experiments/train_omniview_fusion_v2_mpiinf3dhp.py:492-502` tolerates missing
   and unexpected keys, so adding a `joint_attn` sub-module would not break the
   warm-start path from the Bayesian Tri v2 checkpoint.

5. **Training script supports the needed ablations.**
   `experiments/train_omniview_fusion_v2_mpiinf3dhp.py` accepts
   `--graph_num_layers`; adding a symmetric `--n_joint_layers` / `--use_dense_joint_attn`
   flag is the minimal code change.

## Recommendations

1. **Add a configurable dense-joint-attention branch to `OmniMultiViewFusionV2`.**
   Keep the default behavior unchanged, but allow three modes:
   - `graph_only` (current default)
   - `dense_only` (replaces graph block with one or more `TransformerEncoderLayer`s)
   - `hybrid` (runs dense attention, then graph attention, with a residual add)

2. **Wait for the no-graph run to finish before committing GPU time.** The run
   is at the unfreeze boundary; its final MPJPE will tell us whether removing the
   graph block alone is catastrophic. Use that result to decide between
   reintegration or abandonment of the graph-only path.

3. **If reintegration is tested, run a small d=48/10-epoch ablation comparing:**
   - `graph_num_layers=1`
   - `graph_num_layers=0, n_joint_layers=1`
   - `graph_num_layers=1, n_joint_layers=1` (hybrid)
   Use the same warm-start and freeze schedule as the no-graph run to keep the
   comparison fair.

4. **Preserve edge-type bias when hybridizing.** If both blocks are active, feed
   the graph block's per-edge-type embeddings into the residual branch so the
   dense block does not drown the skeleton prior.

5. **Update the decision matrix gate.** The `next_iteration_decision_matrix.md`
   should explicitly list "dense joint attention reintegrated" as a conditional
   action for the 8.35-9.0 mm and >9.0 mm rows, rather than treating the graph
   block as the only tunable component.

## Open questions

- Does reintegrating dense joint attention improve MPJPE when graph attention is
  disabled, and does it reach within 5% of the graph-enabled run?
- Does a hybrid (dense + graph) outperform either single block, or does it
  over-smooth joint features and hurt accuracy?
- How does the reintegrated dense block affect variable-view inference and
  robustness under calibration perturbations relative to the graph-only model?
- What is the GPU/memory cost of the hybrid path compared with graph-only and
  dense-only?
