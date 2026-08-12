# v52 Test-Time Self-Refinement — Risk Report

## 1. Regression / perturbation of a trained baseline

**Risk:** Even with zero-initialized output heads, the module introduces new parameters and gradients.  If the correction head or gate receives a poor gradient signal, it can initially nudge the pretrained pose in the wrong direction during the first epochs, raising val_MPJPE.

**Mitigation:**
* Default `v52_ttsr_identity_init=True` zero-initializes both the correction `ΔP` and the gate `g`, so the forward pass is exactly identity at start.
* Start training with `v52_ttsr_loss_weight=0.0` for a warm-up epoch, then linearly ramp it to the target value.
* Freeze all backbone parameters for the first epoch after enabling the module so only the new head adapts.

## 2. Over-smoothing of fast/athletic motion

**Risk:** The bone-length and temporal-smoothness features can dominate on dynamic actions, causing the network to collapse high-velocity joints toward static averages and increase MPJPE on fast motions (e.g., jumping, kicking).

**Mitigation:**
* Keep `v52_ttsr_temporal_weight` and `v52_ttsr_bone_weight` zero by default; only add them after the supervised 3D loss has stabilised.
* Clamp the magnitude of `ΔP` to a small neighbourhood (e.g. `±200 mm`) so refinements remain local corrections.
* Gate the correction with `g = 2·sigmoid(g) - 1` so each joint can choose to ignore the update, preventing blanket smoothing.

## 3. Latency increase at inference

**Risk:** Running the transformer over `J` joints, plus the iterative refinement loop (`v52_ttsr_num_iter > 1`), adds non-negligible compute per frame.  On A800 this is small, but on embedded / demo targets it may drop throughput below 30 FPS.

**Mitigation:**
* Default `v52_ttsr_num_iter=1` and keep the transformer to `2` layers / `64` hidden units, keeping the module under ~5% of total forward cost.
* Profile with `v52_ttsr_num_iter=0` (bypass) to measure exact overhead before enabling by default.
* Provide a runtime flag `use_v52_test_time_self_refinement` so the module can be disabled without retraining when low latency is required.

## 4. Camera-noise amplification via reprojection residuals

**Risk:** The reprojection residual `e_vt,j` depends on the camera parameters.  If `K`, `R`, or `t` are noisy or only approximately calibrated, the refinement network may overfit to the wrong residual pattern and degrade cross-dataset generalization.

**Mitigation:**
* Detach the camera tensors from the feature computation path so gradients flow through the pose correction only, not through noisy extrinsics.
* Use a Huber loss on `e_vt,j` with a fixed threshold (e.g. `50 px`) to ignore outlier projections.
* Clamp the per-joint residual contribution with `clamp_max=1.0` after normalization so one bad view cannot dominate the token.

## 5. Negative interaction with v50/v51 modules

**Risk:** v50 SEFH and v51 CDSVR already produce per-view reliability and per-joint uncertainty.  If v52 refines the pose *after* these modules, the downstream losses that supervise v50/v51 may receive conflicting gradients, undoing the reliability calibration learned by v51.

**Mitigation:**
* Apply v52 on the base pose *before* v50/v51, then feed the refined pose into v50/v51.  This makes v50/v51 operate on a cleaner input without changing their loss targets.
* If v52 must be placed after v50/v51, detach the pose before v52 and add a small auxiliary loss `MPJPE(P*, P_v50_out)` so the reliability head still sees a consistent target.
* Gate the v52 correction by the v51 per-joint uncertainty: `P* = P0 + (1 - σ) · g ⊙ ΔP`, reducing the correction where v51 is already confident.
