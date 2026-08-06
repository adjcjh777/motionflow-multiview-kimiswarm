# Proposal: Unified Multi-Dataset Canonical Skeleton Prior

**Author:** Iter15 design swarm — agent task "unified-multidataset-canonical-skeleton"  
**Date:** 2026-08-06  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  
**Related prior work in repo:** mixed-dataset loader (`motionflow_mv/data/mixed_dataset.py`), graph-joint-relation (`motionflow_mv/fusion/graph_joint_relation.py`), skeleton-graph residual refiner (`motionflow_mv/fusion/skeleton_graph_residual_refiner.py`), bone-length loss (`motionflow_mv/losses/bone_length.py`).

---

## 1. Problem

The anchor model is trained and evaluated on a single dataset (MPI-INF-3DHP, 17 joints / 4 views).  Human3.6M and AIST++ use the same 17-joint layout, while MPI-INF-3DHP natively uses 28 joints.  The current per-joint residual MLP has no notion of a *common canonical skeleton*, so when the model is exposed to multiple datasets it must implicitly re-learn bone-length and symmetry priors from scratch for each source, which limits cross-dataset generalization and makes the predictions less anatomically grounded.

## 2. Hypothesis

Replacing the dense per-joint residual MLP in the anchor model with a **dataset-conditional graph residual refiner** that predicts corrections on a shared skeleton graph, regularized by a cross-dataset **canonical bone-length consistency loss**, will align predictions with a unified anatomical frame, improve calibration/alignment robustness, and preserve or improve the 9.32 mm anchor while providing a stronger multi-dataset training signal.

## 3. Method

### 3.1 Architecture changes

Create a new residual refiner module that operates on the joint graph and is conditioned on a learnable per-dataset embedding.  It is a drop-in replacement for `self.residual_mlp` in the PP model.

**New file:** `motionflow_mv/fusion/canonical_skeleton_residual_refiner.py`

- Implement `CanonicalSkeletonResidualRefiner(nn.Module)`:
  - **Input:** `(B*T, J, d+3)` — same concatenation of pooled spatio-temporal feature and raw triangulated 3-D pose used by the current `residual_mlp`.
  - **Dataset embedding:** `nn.Embedding(num_datasets, dataset_embed_dim)`; the embedding is broadcast over joints and concatenated to the input.
  - **Graph construction:** reuse `motionflow_mv.fusion.graph_joint_relation.build_edge_index` with `n_views=1`, supporting the existing 17-joint (H36M/AIST++) and 28-joint (MPI-INF-3DHP) skeletons.
  - **Message passing:** two edge-conditioned graph-attention layers (`GraphJointRelation`) over bone, symmetry, and self-loop edges.
  - **Canonical prior:** a learnable per-dataset pose offset `canonical_offset` is added to the predicted residual, giving each dataset a soft mean pose prior in the shared canonical space.
  - **Output:** `(B*T, J, 3)` residual correction.

**New file:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_canonical_skeleton_model.py`

- Subclass `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
- In `__init__`, after the parent init, allocate `self.residual_refiner = CanonicalSkeletonResidualRefiner(...)`.
- Override `forward` to accept an optional `dataset_ids` argument.  When present, expand over the temporal dimension and pass the per-sample dataset ids into the residual refiner; otherwise treat all samples as dataset 0.
- All other logic (principal-point correction, spatio-temporal transformer, triangulation) is unchanged from the anchor.

**New file:** `motionflow_mv/losses/canonical_skeleton_loss.py`

- `canonical_skeleton_loss(pred, target, parents, mask=None)` — MSE between the bone lengths of the predicted and ground-truth 3-D poses, using the canonical parent list.  Supports a per-joint validity mask for mixed-dataset padding.
- `canonical_bone_length_regularizer(pred, canonical_lengths, parents, mask=None)` — optional regularizer toward a fixed canonical bone-length prior (useful when a subset of the data lacks ground-truth 3-D labels).

### 3.2 Loss / data changes

No changes to the existing data loader are required for the smoke test; the model can consume the standard single-dataset format.

For the full multi-dataset run, wire the new loss in the trainer:

- Add a `--canonical_loss_weight` argument (default `0.01`).
- After the -D MPJPE loss, compute `canonical_skeleton_loss(pred_3d, gt_3d, H36M_17_PARENTS)` and add it with the configured weight.
- (Optional) When training on the mixed loader (`motionflow_mv/data/mixed_dataset.py`), pass `dataset_ids` into the model forward so the residual refiner can modulate by dataset identity.

### 3.3 Files to create or modify

- **Create:** `motionflow_mv/fusion/canonical_skeleton_residual_refiner.py`
- **Create:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_canonical_skeleton_model.py`
- **Create:** `motionflow_mv/losses/canonical_skeleton_loss.py`
- **Create (smoke trainer):** `experiments/train_canonical_skeleton_pp_smoke_mpiinf3dhp.py`
  - Thin wrapper around the existing PP trainer, using the new model class and adding the auxiliary canonical-skeleton loss.
- **Modify:** `experiments/eval_full_metrics.py`
  - Add to `MODEL_CLASSES`:
    ```python
    "crossview_residual_pp_canonical_skeleton": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCanonicalSkeleton,
    ```
  - Add the corresponding `elif` branch in `build_model(...)` mirroring the `crossview_residual_pp` path.
- **Modify:** `motionflow_mv/losses/__init__.py`
  - Export `canonical_skeleton_loss`.

## 4. Smoke-Test Plan

Run a 5-epoch smoke on a small MPI-INF-3DHP split, matching the settings used for the factorized ST+PP smoke so the numbers are comparable.

| Setting | Value |
|---|---|
| Train | `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz` (500 random clips) |
| Val | `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz` |
| Clip length | 13 |
| Batch size | 4 |
| Model dims | `j=17`, `d=32`, `residual_hidden=64`, `n_st_layers=2`, `dataset_embed_dim=16`, `graph_num_layers=2` |
| Optimizer | Adam, lr=1e-3 |
| Loss | MPJPE + canonical bone-length loss (`weight=0.01`) |
| Epochs | 5 |

**Pass/fail criteria:**

- **Pass:** training completes with no NaNs / crashes and val MPJPE ≤ 60 mm.
- **Pass:** the canonical refiner produces finite gradients for both 17- and 28-joint skeletons in CPU sanity tests.
- **Pass:** val MPJPE is within 5 mm of the non-canonical PP baseline under identical smoke settings.
- **Fail:** val MPJPE > 80 mm, any NaN/Inf, or graph edge-index construction fails.

## 5. Evaluation Plan

If the smoke passes, evaluate with the standard harness:

- **Clean metrics:**
  - `experiments/eval_full_metrics.py --model crossview_residual_pp_canonical_skeleton --checkpoint <smoke_checkpoint> --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`
  - Report MPJPE, PA-MPJPE, PCK@50/100/150, AUC and compare to the 9.32 mm anchor.
- **Cross-dataset smoke:** run the same 5-epoch smoke on a two-dataset mix (H36M + AIST++ 17-joint) and report val MPJPE on a held-out Human3.6M split; pass if it matches the single-dataset smoke within 5 mm.
- **Per-bone diagnostics:** add a one-off script `experiments/analyze_canonical_skeleton_bone_lengths.py` that reports mean bone-length error and per-joint MPJPE; pass if bone-length consistency improves relative to the anchor.
- **Robustness (CPU, 20-clip smoke):** run `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` for `cxcy_3px` and `view_dropout_0.2`; pass if degradation relative to clean is ≤ 10 percentage points worse than the anchor.

## 6. Estimated GPU/CPU Cost on RTX 4090

| Phase | Hardware | Time |
|---|---|---|
| Smoke (5 epochs, 500 samples) | RTX 4090 | ~15–25 min |
| CPU sanity tests (`tests/test_canonical_skeleton_residual_refiner.py`) | CPU | < 1 min |
| Full training (20–50 epochs, full split) | RTX 4090 / A800-D | ~4–8 h on RTX 4090 |
| Clean eval + robustness matrix | RTX 4090 or CPU | ~10–20 min |

The graph refiner adds only ~10–20k parameters and a sparse graph operation, so memory and throughput are essentially identical to the PP baseline.

## 7. Risks & Fallback

| Risk | Likelihood | Mitigation / Fallback |
|---|---|---|
| The per-dataset embedding overfits to camera/skeleton differences rather than dataset identity. | Medium | Reduce `dataset_embed_dim` to 8 or remove the embedding entirely; keep the skeleton graph as the only new component. |
| Canonical bone-length loss conflicts with the 3-D MPJPE objective and hurts accuracy. | Medium | Lower the loss weight to `0.001` or disable it; keep only the graph residual refiner. |
| Mixed 17- / 28-joint batches require a canonical skeleton map that is not yet implemented. | Low | For the smoke, run single-skeleton batches only; defer full 28→17 mapping to a follow-up `motionflow_mv/data/skeleton_maps.py` PR. |
| Graph message passing is unstable at the chosen learning rate. | Low | Add residual connections and layer norm (already inherited from `GraphJointRelation`); if NaNs appear, reduce lr to 3e-4. |
| No clean improvement over the 9.32 mm anchor. | Medium-High | If the model matches the anchor within 0.3 mm, keep it as the canonical-skeleton variant. If it consistently loses, abandon and keep the existing per-joint residual MLP. |

---

## Summary

Replace the dense per-joint residual MLP in the principal-point anchor with a `CanonicalSkeletonResidualRefiner` that predicts pose corrections on a shared skeleton graph and a per-dataset embedding, and add a matching `canonical_skeleton_loss` to regularize bone lengths.  The change is a single-module swap in the residual path, requires no new data for the smoke, and is validated with a 5-epoch smoke on MPI-INF-3DHP before any full run.
