# Design Graph Joint Relation for Skeleton-Aware Multi-View Fusion

## 1. Technical report

The current best model, `RayAttentionFusionModelTemporal` (`motionflow_mv/fusion/ray_attention_temporal_model.py`), fuses multi-view 2D keypoints with ray-aware view attention, a dense joint-level transformer, a temporal transformer, and a residual refinement head. Its `joint_attn` treats all joints equally and ignores the skeleton topology. We propose replacing that dense joint self-attention with a sparse **Graph Joint Relation (GJR)** module whose edges encode anatomical structure.

**Key design decisions**

1. **Sparse skeleton graph vs. dense attention.** Nodes are per-view, per-joint features after view-level attention. Edges are:
   - *Bone (parent–child) edges* within each view.
   - *Left/right symmetry edges* between mirrored limbs.
   - *Cross-view same-joint edges* so multi-view evidence and skeleton structure propagate jointly.
2. **Edge-conditioned message passing.** Each edge type uses its own projection, and messages are gated by a learned scalar so the network can down-weight anatomical links when observations are unreliable.
3. **Drop-in replacement.** GJR sits in the same slot as the current `joint_attn`, so the ray embedding, temporal transformer, weighted DLT, and residual head remain unchanged.
4. **Skeleton-aware losses.** Reuse `experiments/train_utils.py` for supervised bone-length and symmetry terms.

**Equations**

For node $i$ at layer $l$ with neighbor set $\mathcal{N}(i)$,

$$
\begin{aligned}
m_{j\to i} &= \alpha_{ij}\, W_{e_{ij}} h_j^l, \
\alpha_{ij} &= \sigma\left( \frac{(W_q h_i^l + W_k h_j^l)^{\top} w}{\sqrt{d}} \right), \
h_i^{l+1} &= \mathrm{LayerNorm}\left( h_i^l + \sum_{j\in\mathcal{N}(i)} m_{j\to i} \right).
\end{aligned}
$$

The combined training objective is

$$
\mathcal{L} = \mathcal{L}_{3D} + \lambda_b \mathcal{L}_{\text{bone}} + \lambda_s \mathcal{L}_{\text{sym}} + \lambda_r \mathcal{L}_{\text{reproj}},
$$

where $\mathcal{L}_{3D}$ is the per-joint 3D MSE, $\mathcal{L}_{\text{bone}}$ is the L1 bone-length error, and $\mathcal{L}_{\text{sym}}$ matches mirrored limb lengths.

**Expected impact**

- Occluded joints borrow evidence from connected joints, reducing isolated outliers.
- Bone-length and symmetry priors make the residual head anatomically plausible.
- It provides a clean ablation: dense joint attention (current) vs. skeleton-graph attention (proposed).

**Relevant files**
- `motionflow_mv/fusion/ray_attention_temporal_model.py` (lines 99–107: dense `joint_attn`).
- `motionflow_mv/fusion/ray_attention_v2_model.py` (joint-level attention).
- `experiments/train_utils.py` (bone-length / symmetry losses).
- `docs/swarm_iter3/graph_neural_networks_skeleton.md` (prior GNN survey).
- `docs/swarm_iter7/bone_length_skeleton_consistency_loss.md` (skeleton-loss rationale).

## 2. Implementation plan

1. `motionflow_mv/fusion/graph_joint_relation.py` — implement `GraphJointRelation` module with edge-list construction and edge-conditioned message passing.
2. `motionflow_mv/fusion/ray_attention_temporal_model_graph.py` — new `RayAttentionFusionModelTemporalGraph` that replaces `joint_attn` with `GraphJointRelation` and adds skeleton-loss hooks.
3. `motionflow_mv/fusion/ray_attention_temporal_graph_module.py` — `FusionModule` wrapper for the new model.
4. `motionflow_mv/fusion/__init__.py` — register the new plugin.
5. `experiments/train_ray_attention_temporal_graph_mpiinf3dhp.py` — training script (copy of `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py`) that wires in `bone_length_loss` and `bone_symmetry_loss` with CLI flags `--bone_weight` / `--sym_weight`.
6. `tests/test_graph_joint_relation.py` — forward/backward sanity check on synthetic 4-view, 17-joint data.
7. Smoke run: 2 epochs on `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz` and compare to the existing 11.17 mm baseline.

## 3. Prototype script

See `docs/swarm_iter_next/design_graph_joint_relation/graph_joint_relation_demo.py` for a runnable minimal `GraphJointRelation` and a synthetic forward/backward check.
