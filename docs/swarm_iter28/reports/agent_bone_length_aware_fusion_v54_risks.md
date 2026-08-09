# v54 Bone-Length-Aware Fusion — Risk Register

## R1: Redundancy / conflict with v53 Physical-Space Calibration bone-scale loss

**Risk:** v53 already enforces bone-length and floor constraints. Adding v54 may create competing gradients or double-counting, causing instability or degraded MPJPE.

**Mitigation:**
- Initialize v54 as identity (`v54_blaf_identity_init=True`, `v54_blaf_residual_gate_init=-6.0`).
- Start with `v54_blaf_loss_weight=0.0` for the first few hundred steps and ramp up gradually.
- Ablate `v53 only` vs `v53 + v54` on the same checkpoint to isolate the gain.

## R2: Canonical bone-length overfitting

**Risk:** A learned global/sequence bone-length profile may overfit to average poses and hurt extreme articulations (e.g., crouching, jumping).

**Mitigation:**
- Predict a *length offset* `Δℓ*` rather than an absolute replacement, preserving the current pose geometry at init.
- Use the residual gate `g_j` to let the network bypass the correction when uncertain.
- Add a small regularizer on the magnitude of `Δℓ*` (e.g., `1e-4 · ||Δℓ*||²`).

## R3: Sparse-view joint visibility breaks bone computation

**Risk:** When a parent or child joint is missing, bone length/direction cannot be computed, producing NaN or masked-out bones that remove useful constraints.

**Mitigation:**
- Use a `bone_mask` to mark bones with at least one missing joint as invalid.
- For partially visible bones, fall back to identity (`refined_3d = pred_3d`).
- Ensure the loss is computed only over visible bones and normalized by visible count.

## R4: Memory / compute overhead from per-bone MLP

**Risk:** Even a small MLP on every bone can increase latency and peak memory, especially with long temporal windows and many joints.

**Mitigation:**
- Default to `v54_blaf_hidden=64` and `v54_blaf_n_layers=2`.
- Share MLP weights across symmetric left/right bones to reduce parameters.
- Fuse the bone-length computation across the time dimension with grouped 1D convolutions if profiling shows a bottleneck.

## R5: Warm-start failure when loading v53 checkpoint

**Risk:** If the gate/MLP initialization is not truly identity, enabling v54 on a trained v53 checkpoint could shift `val_MPJPE` by more than the allowed 0.1 mm, violating the module's compatibility contract.

**Mitigation:**
- Set the final gate linear layer weights to zero and bias to `-6.0` at init.
- Set the canonical offset MLP output layer to zero so `Δℓ* = 0` at init.
- Add an unit test that instantiates `OmniMultiViewFusionV5(use_v53=True, use_v54=True)` with a random input and asserts `||output_v54 - output_v53|| < 1e-4` before any training.
