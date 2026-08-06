# Proposal: Hierarchical View → Temporal → Skeleton-Joint Attention

**Author:** iter15 design swarm  
**Date:** 2026-08-06  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP S2/Seq1 clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  

---

## 1. Hypothesis

A **coarse-to-fine hierarchical attention backbone**—camera-group view attention, then temporal attention, then skeleton-graph joint attention—will aggregate multi-view evidence more robustly than a flat spatio-temporal transformer and improve both absolute accuracy and cross-view robustness, while remaining a minimal, ablatable change on top of the current anchor.

---

## 2. Related Existing Files / Modules

| File / Module | Role in this proposal |
|---------------|----------------------|
| `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py` | **Anchor model** (9.32 mm). We subclass it and replace only the flat `(time, view)` transformer block. |
| `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py` | Parent residual + DLT pipeline that we keep unchanged. |
| `motionflow_mv/fusion/principal_point_correction.py` | Learned intrinsic correction; reused unchanged. |
| `motionflow_mv/fusion/graph_joint_relation.py` | `GraphJointRelation` / `build_edge_index` for skeleton-aware (view, joint) message passing; used in the new joint-attention stage. |
| `motionflow_mv/losses/reprojection.py` | Existing reprojection loss; used as an auxiliary signal. |

---

## 3. Proposed Code Changes

### 3.1 New model file

**Create:** `motionflow_mv/fusion/ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model.py`

- New class:
  ```python
  class RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint(
      RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
  ):
  ```
- New constructor arguments (defaults):
  ```python
  n_view_groups: int = 2,
  n_view_layers: int = 2,
  n_temporal_layers: int = 2,
  n_joint_graph_layers: int = 1,
  use_skeleton_graph: bool = True,
  ```
- New private helper:
  ```python
  class _HierarchicalViewTemporalJointBlock(nn.Module)
  ```
  Implements the three-stage hierarchy:
  1. **Hierarchical view attention**: contiguous camera groups → within-group self-attention → cross-group token exchange → broadcast residual.
  2. **Temporal attention**: standard self-attention over time for every `(view, joint)` token.
  3. **Skeleton-graph joint attention**: `GraphJointRelation` over bone / symmetry / cross-view edges.

### 3.2 Loss change (optional ablation)

**Create:** `motionflow_mv/losses/cross_view_group_consistency.py`

- Propose a `cross_view_group_consistency_loss(pred_3d_group_a, pred_3d_group_b, mask)` that triangulates per-camera-group 3D poses and penalizes their 3-D disagreement.
- This is **not yet implemented**; it is reserved as a follow-up if the backbone smoke shows promise, so the first smoke can be trained with the existing MPJPE + confidence-weighted triangulation loss only.

### 3.3 Files that remain untouched

- All existing experiment scripts, configs, and the anchor model remain unchanged.
- No modifications to `motionflow_mv/fusion/__init__.py` are required for the smoke; registration of a FusionModule wrapper can be added after the ablation proves viable.

---

## 4. Training / Smoke Plan

Run a **5-epoch smoke** on MPI-INF-3DHP S2/Seq1 to validate that the new hierarchy trains stably and does not underperform the anchor at the same sample budget.

| Setting | Value |
|---------|-------|
| Dataset | MPI-INF-3DHP S2/Seq1 |
| Samples | 500 (matching prior smoke protocol) |
| Epochs | 5 |
| Batch size | 2 |
| Model dims | `d=32`, `residual_hidden=64`, `n_view_groups=2`, `n_view_layers=2`, `n_temporal_layers=2`, `n_joint_graph_layers=1` |
| Loss | Existing MPJPE + confidence-weighted triangulation; optionally add reprojection loss at 0.1 weight |
| Training script | Adapted from `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` by importing the new model class |

**Estimated runtime on RTX 4090:** ~20–30 min for the 5-epoch smoke.

---

## 5. Success Metrics

| Metric | Target |
|--------|--------|
| Smoke val MPJPE | ≤ 60 mm and no NaNs / crashes |
| Smoke val MPJPE vs. anchor smoke | Within 5 % of the anchor smoke at the same sample budget |
| Full-run clean MPJPE (MPI-INF-3DHP S2/Seq1) | ≤ 9.1 mm (improve or match the 9.32 mm anchor within measurement noise) |
| Full-run PA-MPJPE | ≤ 5.3 mm or better |
| Cross-view robustness | ≥ 3 % relative improvement on a worst-50 %-view-dropout subset |
| Cross-dataset sanity | Stable inference on H36M / Shelf / Campus without retraining; no regression vs. anchor on A800-D validation |

---

## 6. Risk and Fallback

| Risk | Mitigation / Fallback |
|------|----------------------|
| The hierarchy adds memory/compute and the RTX 4090 cannot fit the same batch size. | Reduce `n_view_layers` / `n_temporal_layers` to 1 or set `use_skeleton_graph=False` to isolate the view stage. |
| Camera grouping by contiguous index is suboptimal for non-linear rig layouts. | Make grouping camera-angle aware (e.g., sort by camera azimuth before splitting) or learn soft view grouping. |
| Graph joint attention overfits the small smoke set. | Drop `n_joint_graph_layers` to 0 (equivalent to the anchor) and rely on the hierarchical view/temporal stages only. |
| No accuracy gain on full data. | Abandon the new backbone; the change is a single subclass and is fully reversible. |

---

## 7. Deliverables

- **Design document:** `docs/swarm_iter15/proposal_hierarchical-view-temporal-joint-attention.md`
- **Runnable skeleton:** `motionflow_mv/fusion/ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model.py` (syntax-checked with `py_compile`)
