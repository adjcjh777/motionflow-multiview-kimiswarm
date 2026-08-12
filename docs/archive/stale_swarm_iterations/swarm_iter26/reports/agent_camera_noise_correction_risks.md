# v52 Camera Noise Correction — Risks and Mitigations

## Risk 1: Camera corrections collapse to a degenerate geometry

* **Description.**  A small update to `K, R, t` can push one or more cameras into a configuration where rays no longer intersect, producing `NaN` or extremely large triangulation errors.  Because the correction is learned, it may exploit the triangulation objective to reduce training MPJPE in a geometrically implausible way (e.g., by pushing all cameras to the same viewpoint).
* **Likelihood.**  Medium during the first epochs, when the gate is still small but gradients can already drift.
* **Mitigation.**  Clamp all correction terms explicitly (focal, principal point, rotation, translation).  Add a differentiable sanity check in `forward` that resets any camera whose reprojection error after correction exceeds a threshold to its uncorrected value.  Initialize the final projection to zero and start with `v52_cnc_residual_gate_init = -6` so corrections are negligible until the rest of the network has warmed up.

## Risk 2: Conflict with existing v21 neural bundle adjustment and v45 adaptive geometry fusion

* **Description.**  v21 already refines pose and cameras, and v45 learns per-view triangulation weights.  Running all three simultaneously may create redundant or competing optimization paths: v52 corrects cameras, v21 re-refines them, and v45 down-weights views, so the model can become unstable or overfit to the training camera setup.
* **Likelihood.**  Medium when stacked naïvely.
* **Mitigation.**  Make `use_v52_camera_noise_correction` mutually exclusive with `use_neural_bundle_adjustment_v21` for the first smoke run, then run an ablation that replaces v21 with v52.  Keep v45 enabled because adaptive weights complement the corrected cameras, but gate v52 updates by the v45 reliability score so unreliable views receive smaller corrections.

## Risk 3: Rotation parameterization becomes non-differentiable or unstable

* **Description.**  The `so(3)` exponential map is needed to keep `R_corrected` a valid rotation matrix.  If implemented with `cv2.Rodrigues` inside the forward pass, the backward pass may be discontinuous or break in mixed precision.  Naïve axis-angle updates can also wrap around 2π or produce sign ambiguities.
* **Likelihood.**  High if not handled carefully.
* **Mitigation.**  Use a small-angle first-order approximation `exp([ξ]) ≈ I + [ξ]` clamped to a few degrees, which is sufficient for calibration noise and avoids the Rodrigues singularity.  Alternatively, represent the correction as a unit quaternion predicted from `ξ/||ξ||` and `||ξ||`, then compose via quaternion multiplication.  Unit-test the gradient through the correction module on GPU.

## Risk 4: Auxiliary reprojection loss overfits to training camera noise patterns

* **Description.**  If `v52_cnc_loss_weight > 0`, the auxiliary loss uses the corrected cameras to reproject the ground-truth 3D pose.  On clean datasets this can drive the correction head to memorize per-sequence camera biases that do not generalize, or it can amplify noise when the 2D detections are already accurate.
* **Likelihood.**  Medium on small datasets.
* **Mitigation.**  Default `v52_cnc_loss_weight = 0.0` and only enable it after the main MPJPE loss has plateaued.  Apply the loss only on the triangulated pose, not on the 3D ground truth, so the correction is tied to the actual multi-view consistency objective.  Add an L2 regularizer on the correction magnitudes.

## Risk 5: Temporal smoothing leaks future information or breaks causality

* **Description.**  `v52_cnc_use_temporal_context = True` adds a 1-D Conv1D over the clip.  A non-causal kernel would use future frames to correct the current camera, which is acceptable offline but violates the real-time / causal motionflow pipeline assumed in v49-Lite and downstream deployment.
* **Likelihood.**  Medium if the temporal block is copied from v47 without enforcing causality.
* **Mitigation.**  Use a causal Conv1D with left padding only, or a lightweight GRU with hidden state propagated forward in time.  Add a smoke test with `T = 1` to ensure the module still works frame-wise.
