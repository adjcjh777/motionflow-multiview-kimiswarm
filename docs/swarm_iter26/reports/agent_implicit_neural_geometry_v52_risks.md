# Agent v52: Implicit Neural Geometry — Risk Report

**Owner:** design-swarm agent v52  
**Module:** `implicit_neural_geometry_v52`  
**Tracking issue:** #190  
**Date:** 2026-08-09  

## 1. Risk: Canonical frame construction is brittle for non-H36M skeletons

**Description:** v52 builds a per-joint local frame from the kinematic parent and grandparent. If the dataset uses a different skeleton (MPI-INF-3DHP 28 joints, AIST++ 24 joints, 3DPW 24 joints), the parent list, root definition, or bone orientation conventions may differ. A wrong frame maps the implicit field to inconsistent canonical coordinates, producing garbage residuals and NaNs during the first training steps.

**Evidence:** Previous modules (`v28_physical_space_alignment_v28.py`, `v40_skeleton_aware_physical_loss.py`) already maintain per-skeleton parent lists (`H36M_17_PARENTS`, `MPI_INF_3DHP_28_PARENTS`). The v52 design assumes a single `parents` argument but does not yet specify how to derive it from `j`.

**Mitigation:**
- Accept `parents` as a constructor argument and pass it explicitly from `OmniMultiViewFusionV5` based on `self.j`.
- Implement a fallback for unsupported `j` that uses the pelvis-to-hip vector and a world-up axis instead of full kinematic tree.
- Unit-test canonical-frame construction on H36M-17, MPI-28, and 3DPW-24 stubs.

## 2. Risk: The implicit field overfits to the training poses and becomes an identity map

**Description:** Because the residual gate is initialized to zero and the surface head is zero-initialized, the easiest path for the network is to keep producing zero residuals and zero surface energy everywhere. If the auxiliary loss weights are too low, the module remains an expensive no-op. If they are too high, it may overfit to training-set pose statistics and hurt generalization to novel poses or datasets.

**Evidence:** Warm-startable residual modules in this codebase (v33 HMSP, v34 VJGN, v36 UGIGR, v37 SCVR) all explicitly zero-initialize residual branches and rely on small loss weights to nudge them away from identity. v52 faces the same cold-start problem but with a more expressive MLP field.

**Mitigation:**
- Begin smoke tests with `v52_ing_surface_loss_weight=0.01` and `v52_ing_bone_loss_weight=0.001`; do not increase until the smoke shows non-zero residuals.
- Monitor the mean absolute residual `mean(|Δp|)` and the surface energy mean `|s|`; both should move away from zero within the first 100 steps.
- Add an explicit identity-loss penalty in early epochs (optional) to keep the module from drifting too far from the warm v51 checkpoint before it has learned a meaningful field.

## 3. Risk: Inner-loop camera/pose refinement is unstable and slow

**Description:** The optional Gauss-Newton refinement step differentiates through the reprojection of the refined pose with respect to the predicted 3-D coordinates. This adds a second-order-style backward graph on top of an already deep model and can explode memory or create NaN gradients when camera geometry is near-degenerate (e.g., 3DPW actual mode with `V=1`).

**Evidence:** v25 geometry fusion and v21 neural bundle adjustment already showed that camera/pose joint optimization is sensitive to initial pose quality and can diverge if damping is not conservative. The v52 inner loop does not even include explicit damping, so it is high-risk.

**Mitigation:**
- Keep `v52_ing_use_inner_loop_refinement=False` for the first smoke and full run; ship v52 as a residual-only module initially.
- When enabling the inner loop, cap `v52_ing_inner_loop_steps=1` and clamp coordinate updates to `±5` mm per step.
- Use `torch.no_grad()` for the inner loop at inference, so it never contributes to the training graph.
- Add a guard: skip the inner loop when `n_visible_views < 2` to avoid degenerate single-view optimization in early training.

## 4. Risk: Interaction with v50 SEFH and v51 CDSVR produces conflicting losses

**Description:** v50 SEFH already predicts per-view reliability and per-joint log-variance from reprojection, temporal, and epipolar terms. v51 CDSVR refines these with cross-domain attention. v52 adds a geometry energy and residual correction on top of the same signals. The three modules could fight each other: SEFH/CDSVR down-weights a noisy view, but v52 tries to force the pose onto a learned manifold that conflicts with the reduced evidence.

**Evidence:** In v43 adaptive per-node residual, adding a second residual gate on top of v36 UGIGR required careful tuning to avoid over-correction. v52 adds a third refinement layer (ING field + residual + optional inner loop), increasing the chance of compounding corrections.

**Mitigation:**
- Make v52 consume, not replace, SEFH/CDSVR outputs: use `R_sefh` to weight the reprojection term and `Λ_sefh` to scale the bone-length consistency loss, so all three heads agree on which joints/views are trustworthy.
- Add a global gate `v52_ing_residual_gate_init=0.0` and let it learn to remain near identity if v51 already solves the example.
- Run an ablation with `use_v50_self_evolution_feedback_head=False` vs. `True` to quantify interaction.

## 5. Risk: Runtime and memory overhead break the A800 batch budget

**Description:** The positional encoding expands canonical coordinates from 3 to `2 * positional_encoding_dim` dimensions, the implicit-field MLP has `n_layers=3` with `hidden=128`, and the optional inner loop runs multiple forward passes. For `B=8`, `T=9`, `V=4`, `J=17`, this could add >10 % memory and latency. Combined with v50/v51, the model may OOM on the A800 smoke or reduce the maximum clip length.

**Evidence:** v47 temporal aggregation and v50 SEFH already increased memory enough that A800 runs were limited to `clip_len=9` or `batch_size=8`. v52’s additional MLP and inner-loop overhead may push runs over the GPU memory threshold.

**Mitigation:**
- Default `v52_ing_hidden=64` (not 128) for the first full run; use 128 only for the scaled d=128 experiment.
- Disable the inner loop by default; it is the dominant memory/latency source.
- Cache the canonical-frame rotation matrices across time steps to avoid recomputing them for every backward pass.
- Run the smoke at `clip_len=3` and `B=4` first; only scale to full `clip_len=9` after confirming GPU headroom.
