# Agent Risk Report: v54 Camera Noise Correction

## 1. Over-correction and cross-dataset calibration collapse

**Risk.** The correction heads may overfit to the training camera rig (e.g., H36M/WebBridge) and learn dataset-specific biases rather than true noise. When evaluated on a different domain (3DPW, MPI-INF-3DHP, or a new capture), the learned corrections can push cameras/keypoints in the wrong direction and raise MPJPE.

**Mitigation.** Enforce strict bounds on every correction (`v54_cnc_max_*`), zero-initialize the final correction layers, and initialize the residual gate to a very small value (`-6.0`). Keep `v54_cnc_loss_weight` modest (≤ 0.1) and use `v54_cnc_warmup_epochs` so the module activates only after v52/v53 have stabilized. Run a domain-transfer ablation where v54 is trained on one dataset and evaluated on another.

## 2. Compounding errors through v52/v53

**Risk.** v54 rewrites the inputs to v52 UWT. If the predicted corrections are wrong, v52's per-view precision weights will be computed from corrupted observations, amplifying the error before v53 ever runs.

**Mitigation.** Gate the correction with the sigmoid-initialized residual gate and start with a tiny gate value. Add a bypass test in the smoke script: load a v52/v53 checkpoint with `use_v54_camera_noise_correction=true` and confirm the delta outputs are near zero and MPJPE changes by < 0.1 mm. Only then enable training.

## 3. Rotation parameterization singularities

**Risk.** Representing camera rotation corrections as Euler angles or a single axis-angle vector can hit singularities when the correction magnitude is large or when compositions stack. This can make training unstable and gradients unreliable.

**Mitigation.** Predict the rotation correction as a small axis-angle vector clamped to a few degrees, or use a 6-D rotation representation for the incremental rotation. Keep corrections small enough that the `SO(3)` exponential map is well-behaved, and validate that `R_corr` stays a proper rotation matrix.

## 4. Differentiable re-triangulation cost and graph complexity

**Risk.** Camera correction changes `K`, `R`, and `t`, so any downstream triangulation must be fully differentiable. Re-triangulating inside the correction block with DLT adds another matrix-inversion path; with noisy corrections, the DLT can become ill-conditioned and produce NaN/Inf gradients.

**Mitigation.** Reuse the existing `weighted_dlt_triangulate` utility with its built-in damping. Clamp the corrected cameras to stay within bounded perturbations. Keep v54 as a pure pre-correction module (only modifies inputs) and do not add iterative bundle adjustment inside the module.

## 5. Gains depend on camera quality and may vanish on clean data

**Risk.** If the input calibration is already very accurate, v54 may introduce extra parameters and regularization without improving MPJPE, or even slightly degrade it by adding noise to the 2-D observations.

**Mitigation.** Make the module fully gated and bypassable via `use_v54_camera_noise_correction=false`. Compare on both clean (H36M) and noisy/real-world datasets. If H36M shows no gain while 3DPW/MPI shows gain, market v54 as a robustness module and keep it optional in the paper ablation table.
