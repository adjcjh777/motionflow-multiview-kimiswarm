# Proposal: Multi-Person Multi-View Association Graph for 3D Pose Fusion

**Author:** Iter15 design swarm — agent task "Multi-person multi-view association graph"  
**Date:** 2026-08-06  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  
**Related prior work in repo:** skeleton-graph attention (`motionflow_mv/fusion/graph_joint_relation.py`), graph-joint PP model (`motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_graph_model.py`), principal-point correction (`motionflow_mv/fusion/principal_point_correction.py`).

---

## 1. Problem

The current 9.32 mm anchor fuses a *single* person observed by multiple calibrated views. In real multi-view video, multiple people appear simultaneously and the same person must be associated across views before triangulation. Without an explicit association/reasoning step, per-person triangulation is either impossible or requires an separate, error-prone matching stage.

## 2. Hypothesis

A lightweight `(view, person, joint)` association graph inserted between the spatio-temporal transformer and the triangulation head can jointly fuse multi-view video of multiple people, resolve cross-view identity ambiguities, and preserve or improve the single-person anchor accuracy while adding a natural multi-person capability suitable for ICRA/CVPR 2027.

## 3. Method

### 3.1 Architecture changes

Create two new files. The first is a reusable graph module:

**New file:** `motionflow_mv/fusion/multiperson_association_graph.py`

- `MultiPersonAssociationGraph(nn.Module)`
  - **Input:** `(N, V, P, J, d)` — spatio-temporal tokens for each person `p ∈ [0, P-1]`.
  - **Graph nodes:** one node per `(view, person, joint)`.
  - **Edges:**
    - Type 0 (skeleton): bone and symmetry edges inside each `(view, person)` pair, reusing `H36M_17_*` or `MPI_INF_3DHP_28_*` from `graph_joint_relation.py`.
    - Type 1 (cross-view): same-person same-joint edges across views, strengthening multi-view fusion.
    - Type 2 (cross-person): same-view same-joint edges across people, modelling occlusion/identity constraints.
  - **Message passing:** two edge-conditioned graph-attention layers with type-specific projections and a sigmoid attention gate, matching the style of `GraphJointRelation`.
  - **Output:** `(N, V, P, J, d)` refined tokens.

The second file is a drop-in model that subclasses the anchor:

**New file:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_multiperson_assoc_model.py`

- `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointMultiPersonAssoc`
  - Subclasses `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
  - Accepts either the existing single-person input `(B, T, V, J, 3)` or a new multi-person input `(B, T, V, P, J, 3)`.
  - For `P > 1`, the forward pass:
    1. Reshapes input to `(B*P, T, V, J, 3)` and repeats cameras accordingly.
    2. Runs the anchor's per-person backbone up to the spatio-temporal transformer, producing tokens `(B*P*T, V, J, d)`.
    3. Reshapes to `(B*T, V, P, J, d)` and applies `MultiPersonAssociationGraph`.
    4. Reshapes back and runs the existing weight head, DLT triangulation, and residual MLP independently per person.
    5. Returns `pred_3d: (B, T, P, J, 3)` and `weights: (B, T, V, P, J)`.
  - For `P = 1`, the association graph is disabled and the model degenerates to the anchor.

### 3.2 Loss / data changes

No new loss is required for the smoke test; the existing per-joint 3D MSE is applied independently to each person.

For the full multi-person run, optionally add an **association consistency loss** (`motionflow_mv/losses/multiperson_association_loss.py`):

```python
class MultiPersonAssociationLoss(nn.Module):
    def __init__(self, distinctiveness_weight: float = 0.01):
        super().__init__()
        self.distinctiveness_weight = distinctiveness_weight

    def forward(self, pred_3d: torch.Tensor, gt_3d: torch.Tensor):
        # pred_3d / gt_3d: (B, T, P, J, 3)
        mse = F.mse_loss(pred_3d, gt_3d)
        # Encourage different people to occupy different 3-D centres.
        centres = pred_3d.mean(dim=-2)  # (B, T, P, 3)
        diffs = centres.unsqueeze(3) - centres.unsqueeze(2)  # (B, T, P, P, 3)
        # Penalize small pairwise distances (only for distinct people).
        eye = torch.eye(centres.shape[2], device=centres.device, dtype=torch.bool)
        dists = diffs[~eye].norm(dim=-1)
        distinct = torch.relu(1.0 - dists).mean()
        return mse + self.distinctiveness_weight * distinct
```

Training data: existing single-person `.npz` files are stacked offline into synthetic multi-person clips by translating two non-overlapping subjects to different world coordinates, producing `(T, V, P, J, 3)` inputs with `P=2`.

### 3.3 Files to create or modify

- **Create:** `motionflow_mv/fusion/multiperson_association_graph.py`
- **Create:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_multiperson_assoc_model.py`
- **Create (optional):** `motionflow_mv/losses/multiperson_association_loss.py`
- **Create (smoke):** `experiments/train_multiperson_assoc_pp_smoke_mpiinf3dhp.py` by extending `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
  - Stack two single-person clips into a `P=2` sample on the fly.
  - Add argument `--n_persons` (default `2`).
- **Modify:** `experiments/eval_full_metrics.py`
  - Add the new model class to `MODEL_CLASSES` and `build_model(...)`.

## 4. Smoke-Test Plan

Run a 5-epoch smoke on a small MPI-INF-3DHP split. Because true multi-person ground truth is not available, the smoke validates both the `P=1` backward-compatibility path and a synthetic `P=2` path.

| Setting | Value |
|---|---|
| Train | `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz` (500 random clips) |
| Val | `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz` |
| Clip length | 13 |
| Batch size | 4 |
| Model dims | `d=32`, `residual_hidden=64`, `n_st_layers=2`, `assoc_num_layers=2`, `n_persons=2` |
| Optimizer | Adam, lr=1e-3 |
| Loss | MPJPE only (no distinctiveness term in smoke) |
| Epochs | 5 |

**Pass/fail criteria:**

- **Pass:** `P=1` path runs and produces finite gradients; val MPJPE ≤ 60 mm.
- **Pass:** `P=2` synthetic path runs without NaNs/Inf and outputs shapes `(B, T, P, J, 3)` and `(B, T, V, P, J)`.
- **Pass:** the association-graph edge index builds correctly for both 17- and 28-joint skeletons.
- **Fail:** any NaN/Inf, output-shape mismatch, or edge-index construction failure.

## 5. Evaluation Plan

If the smoke passes, evaluate as follows:

1. **Single-person backward compatibility:**
   - Run `experiments/eval_full_metrics.py --model multiperson_assoc_pp --n_persons 1 ...` on the standard S2/Seq1 split.
   - Target: clean MPJPE within 0.5 mm of the 9.32 mm anchor.

2. **Multi-person synthetic benchmark:**
   - Generate 200 synthetic `P=2` validation clips by combining two non-overlapping subjects.
   - Report per-person MPJPE and association accuracy (percentage of correctly matched person identities across views).
   - Target: association accuracy ≥ 90 % and per-person MPJPE within 15 % of the single-person anchor.

3. **Robustness:**
   - Apply `view_dropout_0.2` and `joint_dropout_0.2` to one person while keeping the other clean.
   - Target: per-person MPJPE degradation ≤ 10 percentage points worse than the anchor under the same single-person corruption.

## 6. Estimated GPU/CPU Cost on RTX 4090

| Phase | Hardware | Time |
|---|---|---|
| Smoke (5 epochs, 500 samples, P=2) | RTX 4090 | ~20–35 min |
| CPU sanity (instantiate + forward shapes) | CPU | < 1 min |
| Full single-person eval | RTX 4090 | ~10 min |
| Full multi-person training (20 epochs, synthetic 2-person blend) | RTX 4090 | ~3–5 h |

The graph adds `O(P²·V²·J)` edges but only two message-passing layers, so for small `P` (≤4) the overhead is modest compared with the ST transformer.

## 7. Risks & Fallback

| Risk | Mitigation / Fallback |
|------|-----------------------|
| Association graph hurts single-person (`P=1`) accuracy. | Make the graph bypassable; when `n_persons=1` the model is identical to the anchor. |
| Synthetic multi-person data is too simplistic. | Limit the smoke to `P=1`; postpone multi-person training until Shelf/Campus multi-person data or CMU Panoptic is integrated. |
| Edge index becomes large for `P > 4`. | Cap `n_persons` in the smoke at 2–4; for larger crowds, switch to a coarse-to-fine graph that first clusters people and then refines. |
| Cross-person edges cause identity collapse (all poses identical). | Add the distinctiveness loss; if it still collapses, remove cross-person edges and keep only cross-view + skeleton edges. |
| No real multi-person ground truth available. | Fall back to leaving the code as a reusable module and demonstrate single-person backward compatibility only. |

---

## Summary

Extend the 9.32 mm anchor with a `(view, person, joint)` association graph that refines spatio-temporal features across people and views before triangulation. The change is a single-module, two-file addition, keeps `P=1` behavior identical to the anchor, and is validated first with a 5-epoch smoke on synthetic two-person clips from MPI-INF-3DHP.
