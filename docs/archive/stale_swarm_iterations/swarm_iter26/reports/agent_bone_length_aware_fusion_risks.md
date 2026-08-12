# v52 Bone-Length-Aware Fusion — Risk Register

**Module:** `bone_length_aware_fusion_v52`  
**Proposal:** `docs/swarm_iter26/proposals/v52_bone_length_aware_fusion.md`  
**Author:** design-swarm agent  
**Date:** 2026-08-09  

## 1. Warm-start / identity-at-init fails to preserve baseline accuracy

**Risk:** Even with zero-initialized output layers, the auxiliary bone-length loss or the bone transformer residual connection may perturb the trained baseline when `v52_bone_residual_gate > 0`, causing an sudden MPJPE regression on the first smoke run.

**Mitigation:**
* Initialize the length-correction head `MLP_length` and the skeleton GNN final projection to zero weights and zero biases.
* Start all smoke and first full runs with `v52_bone_residual_gate=0.0` and increase it only after the auxiliary loss has converged.
* Add a unit test that asserts `module(pred_3d, ...).pred_3d == pred_3d` when `residual_gate=0.0`.

## 2. Learned bone-length prior collapses to a single canonical skeleton

**Risk:** The learnable mean `μ_j` and variance `σ_j^2` may converge to dataset-specific averages (e.g. H36M adult sizes), hurting generalization on children, extreme poses, or cross-domain subjects where absolute bone lengths differ.

**Mitigation:**
* Parameterize the prior as **bone-length ratios** relative to a reference bone (e.g. torso length) instead of absolute lengths, so the model is scale invariant.
* Add a per-subject latent bone-length embedding learned from the clip-level features, conditioned on the pooled feature vector.
* Clamp the loss to bones with high visibility and use the log-variance term to let uncertain bones adapt.

## 3. Conflict with existing physical-space modules (v28/v40, v47/v49-Lite)

**Risk:** v28/v40 already enforce bone-length, floor, and symmetry priors. Adding v52 may duplicate penalties or, worse, create opposing gradients when v52 shortens bones while v28 penalizes the same joint displacement.

**Mitigation:**
* Place v52 **before** v28/v40 so that v28/v40 operate on the already bone-consistent pose rather than fighting it.
* Make the v52 auxiliary loss weight (`v52_bone_loss_weight`) tunable and start at `0.0` or a very small value (`1e-3`) during the first epoch.
* If v28 `bone_temporal_weight` is active, disable it during v52 smoke tests to isolate the effect.

## 4. Skeleton graph mismatch across datasets (H36M 17 joints vs. MPI 28 joints)

**Risk:** `OmniMultiViewFusionV5` supports both 17-joint and 28-joint skeletons. Hard-coding the parent list inside the v52 module will break when the model is trained on mixed datasets.

**Mitigation:**
* Always pass the runtime `parents` list to `BoneLengthAwareFusionV52.forward`.
* Build the adjacency / incidence matrix on-the-fly in the skeleton GNN from `parents`.
* Unit-test with both 17-joint and 28-joint skeletons in the smoke tests.

## 5. Sparse-view degradation and occlusion handling

**Risk:** When only 2–3 views are visible, many bone endpoints are occluded or have low visibility. The bone-length transformer may then attend mostly to invalid/uncertain bones and amplify noise.

**Mitigation:**
* Use `v52_bone_min_visible_views` to mask out invalid bones before the transformer.
* Weight the transformer attention by per-bone visibility confidence derived from the v46/v51 reliability or the raw `visibility` tensor.
* Gate the final pose correction by a per-bone confidence predicted by the network, initialized so low-confidence bones contribute nothing at startup.
