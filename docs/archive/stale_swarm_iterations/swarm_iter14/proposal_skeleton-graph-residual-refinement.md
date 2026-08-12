# Proposal: Skeleton-Graph Residual Refinement

**Author:** Iter14 design swarm — agent task "Skeleton-graph residual refinement: propagate pose corrections along bone graph"  
**Date:** 2026-08-06  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  
**Related prior work in repo:** Graph-joint-relation PP model (`motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_graph_model.py`), per-joint residual MLP in `ray_attention_temporal_crossview_residual_principal_point_model.py:62-68`.

---

## 1. Problem

The anchor model refines the triangulated 3-D pose with a dense per-joint MLP that predicts each joint's correction independently, so anatomically impossible configurations (e.g., over-stretched limbs, asymmetric left/right arms) are not explicitly penalized and errors at one joint cannot be corrected by its neighbors along the bone graph.

## 2. Hypothesis

Replacing the per-joint residual MLP with a small skeleton-graph message-passing module that propagates pose corrections along bone and symmetry edges will enforce local anatomical consistency, reduce distal-joint errors, and preserve or improve the 9.32 mm anchor with only a modest parameter increase.

## 3. Method

### 3.1 Architecture changes

Create a new residual refiner module that operates on the joint graph only (no view dimension) and is a drop-in replacement for `self.residual_mlp` in the existing PP-graph model.

**New file:** `motionflow_mv/fusion/skeleton_graph_residual_refiner.py`

- Implement `SkeletonGraphResidualRefiner(nn.Module)`:
  - **Input:** `(B*T, J, d+3)` — concatenation of the pooled spatio-temporal feature `(B*T, J, d)` and the raw triangulated 3-D pose `(B*T, J, 3)`, exactly as the current `residual_mlp` receives.
  - **Graph construction:** reuse `motionflow_mv.fusion.graph_joint_relation.build_edge_index` with `n_views=1`, using `H36M_17_PARENTS/SYMMETRY` for 17 joints and `MPI_INF_3DHP_28_PARENTS/SYMMETRY` for 28 joints (raise `NotImplementedError` for other layouts).
  - **Message passing:** two lightweight graph-attention layers. Each node aggregates messages from bone/symmetry/self-loop neighbors via edge-type-aware attention (reusing the same dot-product + MLP attention pattern from `GraphJointRelation` but without the view dimension).
  - **Output:** `(B*T, J, 3)` residual correction.
  - Keep the hidden dimension at `residual_hidden` (default 128) and the layer count at 2; no expansion of model capacity.

**New file:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_graph_skeleton_residual_model.py`

- Subclass `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph`.
- In `__init__`, after the parent init, replace `self.residual_mlp` with a `SkeletonGraphResidualRefiner` instance. No other forward logic is changed; the base `forward` still calls `delta = self.residual_mlp(residual_input)`.

### 3.2 Loss / data changes

No loss or data changes are required for the smoke test; keep the standard MPJPE objective.

For the full run, optionally add the existing skeleton consistency term as an auxiliary loss:

- `motionflow_mv/losses/bone_length.py` can be applied to the final refined pose with a small weight (e.g., `bone_weight=0.01`).
- This is **not** required for the smoke and should only be enabled if the smoke shows bone-length drift.

### 3.3 Files to create or modify

- **Create:** `motionflow_mv/fusion/skeleton_graph_residual_refiner.py`
- **Create:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_graph_skeleton_residual_model.py`
- **Create:** `experiments/train_skeleton_graph_residual_pp_smoke_mpiinf3dhp.py` (copy of `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`, fixed to the new model)
- **Create:** `tests/test_skeleton_graph_residual_refiner.py` (CPU sanity: forward + backward for 17-joint and 28-joint skeletons)
- **Modify:** `experiments/eval_full_metrics.py`
  - Add to `MODEL_CLASSES`:
    ```python
    "crossview_residual_pp_graph_skeleton": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraphSkeletonResidual,
    ```
  - Add the corresponding `elif` branch in `build_model(...)` mirroring the `crossview_residual_pp` path.
- **Modify:** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
  - Add `"graph_skeleton_residual"` to the `model_type` choices.
  - Add an `elif args.model_type == "graph_skeleton_residual":` block that instantiates the new model class with the same PP kwargs.

## 4. Smoke-Test Plan

Run a 5-epoch smoke on a small MPI-INF-3DHP split, matching the settings used for the factorized ST+PP smoke so the numbers are comparable.

| Setting | Value |
|---|---|
| Train | `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz` (500 random clips) |
| Val | `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz` |
| Clip length | 13 |
| Batch size | 4 |
| Model dims | `d=32`, `residual_hidden=64`, `n_st_layers=2`, `graph_num_layers=1` (graph attention in encoder), skeleton-graph residual layers = 2 |
| Optimizer | Adam, lr=1e-3 |
| Loss | MPJPE only (no bone-length term in smoke) |
| Epochs | 5 |

**Pass/fail criteria:**

- **Pass:** training completes with no NaNs / crashes and val MPJPE ≤ 60 mm.
- **Pass:** the skeleton-graph refiner produces finite gradients and the edge index builds correctly for both 17- and 28-joint layouts.
- **Pass:** val MPJPE is within 5 mm of the non-graph PP-graph baseline under identical smoke settings (if available).
- **Fail:** val MPJPE > 80 mm, any NaN/Inf, or edge-index construction fails for a supported skeleton.

## 5. Evaluation Plan

If the smoke passes, evaluate with the standard harness:

- **Clean metrics:**
  - `experiments/eval_full_metrics.py --model crossview_residual_pp_graph_skeleton --checkpoint <smoke_checkpoint> --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`
  - Report MPJPE, PA-MPJPE, PCK@50/100/150, AUC.
- **Comparison:** run the same script on the 9.32 mm anchor checkpoint (`crossview_residual_pp`) on the identical validation split and compare.
- **Per-joint / bone-length diagnostics:** add a one-off script `experiments/analyze_skeleton_graph_residual.py` that reports mean bone-length error and per-joint MPJPE; pass if distal joints (wrists/ankles) improve relative to the anchor without degrading root joints.
- **Robustness (CPU, 20-clip smoke):** run `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` for `joint_dropout_0.2` and `view_dropout_0.2`; pass if degradation relative to clean is ≤ 10 percentage points worse than the anchor.

## 6. Estimated GPU/CPU Cost on RTX 4090

| Phase | Hardware | Time |
|---|---|---|
| Smoke (5 epochs, 500 samples) | RTX 4090 | ~15–25 min |
| CPU sanity tests (`tests/test_skeleton_graph_residual_refiner.py`) | CPU | < 1 min |
| Full training (20–50 epochs, full split) | RTX 4090 / A800-D | ~4–8 h on RTX 4090 |
| Clean eval + robustness matrix | RTX 4090 or CPU | ~10–20 min |

The skeleton-graph refiner adds only ~5–10k parameters and a sparse graph operation, so memory and throughput are essentially identical to the PP-graph baseline.

## 7. Risks & Fallback

| Risk | Likelihood | Mitigation / Fallback |
|---|---|---|
| Graph residual overfits the small smoke sample and underperforms the dense MLP. | Medium | If smoke MPJPE is > 5 mm worse than the non-graph baseline, fall back to the dense residual MLP and keep only the encoder graph attention. |
| Skeleton edge index mismatches an unsupported joint layout (e.g., 16-joint). | Low | Raise a clear `NotImplementedError` and default to the parent PP model class for unsupported skeletons. |
| Graph message passing is unstable at high learning rate. | Low | Add residual connections and layer norm (inherited from the `GraphJointRelation` style); if NaNs appear, reduce lr to 3e-4. |
| Bone-length auxiliary loss conflicts with 3-D MPJPE in full run. | Low | Keep `bone_weight` at 0.01 or disable it; if it degrades MPJPE, remove the auxiliary term entirely. |
| No clean improvement over 9.32 mm anchor. | Medium-High | The graph residual is a low-risk inductive bias; if it matches the anchor within 0.3 mm it can be kept as the default graph variant. If it consistently loses, abandon and keep the existing per-joint residual MLP. |

---

## Summary

Replace the dense per-joint residual MLP in the PP-graph model with a `SkeletonGraphResidualRefiner` that propagates pose corrections along bone and symmetry edges. The change is a single-module swap, requires no new data, and is validated with a 5-epoch smoke on MPI-INF-3DHP before any full run.
