# v54 Differentiable Bundle Adjustment — Risk Register

## R1: Camera correction overfits to training-set calibration

**Risk:** The camera correction head may learn dataset-specific intrinsics or extrinsics (e.g., a constant focal-length shift for H36M) rather than correcting genuine capture drift. This improves training-set MPJPE but degrades cross-dataset generalization and fails in the paper's multi-view fusion and calibration story.

**Mitigation:**
- Constrain corrections to be small: clamp focal-scale \(\exp(\Delta f)\) to \([0.95, 1.05]\), principal-point offsets to \([-20, +20]\) px, rotation angles to \([-2°, +2°]\), and translation to \([-50, +50]\) mm.
- Add a strong L2 regularizer `v54_dba_camera_reg_weight=0.1` on all camera deltas.
- Zero-initialize the final correction MLP layer so corrections are exactly zero at init.

## R2: Joint camera/pose optimization diverges

**Risk:** Optimizing 3-D pose and camera parameters simultaneously is a classic chicken-and-egg problem. If the pose residual and camera correction amplify each other, the training loss can diverge or produce implausible poses.

**Mitigation:**
- Keep the pose residual gated with `v54_dba_residual_gate_init=-6.0` (\(\sigma(-6) \approx 0.0025\)) so the pose starts as identity and only gradually absorbs corrections.
- Apply `v54_dba_warmup_epochs=1` so the DBA loss is disabled during the first epoch while the rest of the network stabilizes.
- Use a Huber robustifier (`v54_dba_huber_delta=5.0`) to down-weight outlier 2-D keypoints that could pull the optimization off track.

## R3: Reprojection loss dominates and overrides physical-space calibration

**Risk:** Because the reprojection term is directly tied to 2-D observations, it can dominate the smaller physical-space losses from v53 and push the pose back toward an uncalibrated, geometrically consistent but physically implausible state.

**Mitigation:**
- Scale the DBA loss weight conservatively (`v54_dba_loss_weight=0.01`) so the reprojection term acts as a refinement rather than a replacement.
- Feed v53 physical hints (`h_floor`, `h_bone`) into the pose residual MLP, preserving the floor and bone-scale constraints.
- Compare `v52+v53` vs `v52+v53+v54`; if v54 degrades v53's physical metrics, lower `v54_dba_loss_weight` or disable `v54_dba_use_psc_hints`.

## R4: Memory and runtime overhead from per-view reprojection Jacobians

**Risk:** Computing the robust reprojection loss for every view and joint adds extra forward-pass operations and gradients, increasing peak GPU memory and slowing each training step.

**Mitigation:**
- Avoid explicit Jacobian factorization; use only the scalar reprojection residual loss, which is cheaper than a full Gauss-Newton step.
- Default to `v54_dba_hidden=64` and `v54_dba_n_layers=2` for the correction MLPs.
- If profiling shows a bottleneck, subsample the views used for the DBA loss to the top-k most reliable views according to `uwt_weights`.

## R5: Warm-start failure when loading v53 checkpoint

**Risk:** If the pose or camera correction initialization is not truly identity, enabling v54 on a trained v53 checkpoint could shift `val_MPJPE@full` by more than the allowed 0.1 mm, breaking the module's compatibility contract.

**Mitigation:**
- Zero-initialize the final layer of every correction MLP and set all correction parameter registers to zero.
- Initialize the residual gate to `v54_dba_residual_gate_init=-6.0`.
- Add a unit test that instantiates `OmniMultiViewFusionV5(use_v53=True, use_v54=True)` with a random input and asserts `||output_v54 - output_v53|| < 1e-4` before any training.
