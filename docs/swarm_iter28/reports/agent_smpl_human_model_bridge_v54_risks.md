# v54 SMPL Human Model Bridge — Risk Report

This document lists the main technical risks for the proposed `smpl_human_model_bridge_v54` module and concrete mitigations for each.

## Risk 1: SMPL body model not available or incompatible skeleton

**Description:** The module assumes access to a pre-trained SMPL model and a joint regressor that maps the SMPL mesh to the same `J` joints used by the MotionFlow backbone. If the model file is missing, the joint regressor mismatches, or the target skeleton uses a different joint set (e.g., MPI-INF-3DHP 28 joints vs H36M 17 joints), the bridge cannot produce valid `J_smpl` and may crash or output corrupted poses.

**Mitigation:**
- Provide a fallback **learned surrogate body model** that is enabled when `v54_shmb_smpl_model_path` is missing or `v54_shmb_use_pretrained_smpl=False`.
- Validate the joint-regressor shape at import time and raise a clear error if `(J, 6890)` does not match the expected number of SMPL vertices.
- Use per-joint selection masks so mismatched joints can be ignored during the SMPL-to-backbone mapping.

## Risk 2: Identity-at-init fails and v53 checkpoints regress

**Description:** If the final residual MLP or the gate are not initialized correctly, enabling SHMB on a trained v53 checkpoint could change the output pose by more than the planned `< 0.1 mm`, breaking warm-start compatibility.

**Mitigation:**
- Zero-initialize the final linear layer of `MLP_delta`.
- Initialize the residual gate logit to `-6.0` so `σ(gate) ≈ 0.0025` at start.
- Add a smoke test that loads a v53 checkpoint, enables SHMB, and asserts `||pred_v54 - pred_v53||_∞ < 1e-4 mm` on a single batch.

## Risk 3: SMPL parameter prediction collapses to trivial pose

**Description:** Predicting pose parameters from 3-D joints can collapse to a near-constant, low-variance pose (e.g., T-pose) if the auxiliary losses dominate or the body prior is too strong. This would make `J_smpl` uninformative and waste capacity.

**Mitigation:**
- Start with a small shape regularization (`v54_shmb_shape_reg_weight=0.001`) and disable the pose prior for the first epoch (`v54_shmb_warmup_epochs=1`).
- Use a pose prior only on the **root-relative** rotations, not the global root translation, so the network still learns scene-specific placement.
- Monitor the standard deviation of `theta` across a validation batch; if it drops below `0.05` rad, halve `v54_shmb_pose_prior_weight`.

## Risk 4: Extra parameters and differentiable body model slow training / cause OOM

**Description:** Loading the full SMPL model (`6890` vertices, `24` joints) and running it inside the training loop adds GPU memory and compute. On the RTX 4090 smoke run this may cause OOM, especially when combined with v47 temporal aggregation or v50 SEFH.

**Mitigation:**
- Cache the SMPL `V` mesh and joint regressor as read-only buffers on the GPU; do not compute per-vertex losses.
- Use the **surrogate model** for smoke runs by default; enable the full SMPL model only on A800 with `v54_shmb_use_pretrained_smpl=True`.
- Add a gradient checkpointing toggle (`v54_shmb_checkpoint_gradients`) for the body-model forward pass.

## Risk 5: Conflict with existing physical-space losses (v28, v31, v40, v53)

**Description:** SHMB introduces a body-model prior that overlaps with v53 floor/bone calibration and v40 skeleton-aware physical loss. Stacking all of them may over-constrain the pose and hurt accuracy.

**Mitigation:**
- Place SHMB **after v53 PSC** and before the final residual MLP; document the ordering clearly.
- When SHMB is enabled, recommend zeroing or reducing `v53_psc_bone_weight` and `v40_bone_loss_weight` because the body model already encodes a skeleton prior.
- Run an ablation matrix on smoke configs: (v53 only), (v53+SHMB), (v53+SHMB+v40), and (v53+SHMB+v28).
