# Proposal: Kinematic-Chain Graph Convolutional Refiner

## One-sentence hypothesis

Adding an final, skeleton-aware **kinematic-chain graph convolutional refiner** that operates directly on the triangulated 3-D skeleton—and training it with a small kinematic-chain regularization loss—will improve anatomical plausibility and cross-view robustness while keeping the change minimally invasive and ablatable.

## Related existing files / modules

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py` — current iter14 anchor (clean MPJPE 9.32 mm on MPI-INF-3DHP S2/Seq1).
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py` — base residual refinement model.
- `motionflow_mv/fusion/graph_joint_relation.py` — skeleton definitions and `build_edge_index` helper.
- `motionflow_mv/fusion/skeleton_graph_residual_refiner.py` — existing feature-space skeleton-graph residual refiner.
- `motionflow_mv/losses/bone_length.py` — existing bone-length loss used for regularization.
- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` — closest training script to clone for the smoke run.

## Proposed code changes

### New files

1. `motionflow_mv/fusion/kinematic_chain_graph_refiner.py`
   - `KinematicChainGraphRefiner` — edge-conditioned graph convolutions on `(B, J, 3)` skeletons using bone/symmetry/self-loop edges.
   - `KinematicChainGraphRefinerTemporal` — thin wrapper that applies the refiner per-frame to `(B, T, J, 3)` outputs.

2. `motionflow_mv/losses/kinematic_chain.py`
   - `bone_length_consistency_loss(...)` — robust bone-direction + relative bone-length term.
   - `symmetry_plane_loss(...)` — encourages left/right symmetry around the pelvis plane.
   - `kinematic_chain_loss(...)` — combined regularizer with scalar weights.

3. `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_kinematic_chain_model.py`
   - Class: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointKinematicChain`
   - Inherits from the anchor `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
   - Signature change: adds `kc_hidden_dim: int = 64` and `kc_num_layers: int = 2`.
   - Forward: calls the anchor forward, then passes the 3-D skeleton through `KinematicChainGraphRefinerTemporal` before returning it.

### Modified files

- None for the smoke skeleton. A real experiment would add a new training script (e.g. `experiments/train_ray_attention_temporal_crossview_residual_principal_point_kinematic_chain_mpiinf3dhp.py`) and a config in `configs/`, but these are not created here to keep the change reversible.

## Training / smoke plan

1. Clone `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` into a new smoke script.
2. Replace the model constructor with:
   ```python
   model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointKinematicChain(
       j=j, d=64, n_views=n_views, n_heads=4,
       n_joint_layers=1, n_st_layers=2,
       residual_hidden=128, principal_point_hidden=64,
       principal_point_max_offset=20.0,
       kc_hidden_dim=64, kc_num_layers=2,
   ).to(device)
   ```
3. Loss:
   - Primary: `F.mse_loss(pred, target)` (or existing MPJPE loss).
   - Auxiliary: `0.001 * kinematic_chain_loss(pred, target, parents, symmetry_pairs)`.
4. Smoke dataset: 1 subject of H36M or a small MPI-INF-3DHP split, clip length 9–13, batch size 4–8.
5. Run ≤5 epochs on the RTX 4090. Estimated runtime: **15–45 minutes** for 5 epochs depending on clip length and dataset size.
6. Compare clean MPJPE against the anchor on the same validation split.

## Success metrics

- **Primary**: clean MPJPE ≤ 9.10 mm on MPI-INF-3DHP S2/Seq1 (improvement over 9.32 mm anchor).
- **Robustness axis**: MPJPE under 1-view dropout should degrade by ≤ 15% relative to the anchor.
- **Anatomical plausibility**: mean per-frame bone-length variance on validation should decrease.
- **Convergence**: smoke run reaches a training loss < 1.2× the anchor’s smoke loss within 5 epochs.

## Risk and fallback

- **Risk 1 — Graph refiner over-smooths joints and raises MPJPE.**
  - *Fallback*: Reduce `kc_num_layers` to 1 or `kc_hidden_dim` to 32, or freeze the anchor weights and train only the new refiner for the first few epochs.
- **Risk 2 — Kinematic-chain loss dominates and distorts scale.**
  - *Fallback*: Reduce the loss weight to 1e-4 or remove the symmetry term, keeping only the bone-direction term.
- **Risk 3 — Training is unstable with the new module.**
  - *Fallback*: Initialize the refiner’s output projection to near zero (residual init) so the model starts from the anchor solution and learns only small corrections.
- **Risk 4 — Not publishable enough.**
  - *Fallback*: Frame the refiner as a post-processing block with explicit bone-length/angle constraints and ablate against the existing `SkeletonGraphResidualRefiner` to demonstrate that operating in output 3-D space is superior to feature-space refinement.
