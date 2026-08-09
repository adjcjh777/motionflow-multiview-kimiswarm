# v52 Physical-Space Calibration — Risk Register

## Risk 1: Identity-at-init is not actually identity

**Description**  
If the camera rotation head or pose residual head is not zero-initialised, enabling v52 on a trained v51 checkpoint will immediately perturb the already-good pose/cameras and can cause a large jump in `val_MPJPE` at epoch 0.

**Mitigation**
- Zero-initialise the final linear layers of both `PSCCameraHead` and `PSCPoseHead`.
- Initialise the scalar gate `gate = nn.Parameter(torch.tensor(0.0))` and apply it as `λ = torch.tanh(gate)`.
- Add a smoke test that loads a v51 checkpoint, enables v52, and asserts that the first forward pass returns the input pose/cameras within `1e-4`.

## Risk 2: Camera refinement overfits to training-set rig geometry

**Description**  
The camera correction head learns from `pred_3d` and reprojection residuals. If the training data has a fixed small set of camera rigs, the module may memorise rig-specific corrections rather than learning a general physical-space prior. This will hurt cross-dataset generalisation (e.g., H36M → 3DPW).

**Mitigation**
- Default `v52_psc_refine_intrinsics=False`; only refine extrinsics, which are less tied to sensor-specific calibration.
- Clamp updates tightly (`max_rotation_deg=1.0`, `max_translation_m=0.05`) so the module cannot remap the entire scene.
- Train with domain labels and add a domain-adversarial loss (reuse v48 domain generalisation) on the camera-correction latent to discourage dataset-specific corrections.

## Risk 3: Gradient instability through SO(3) parameterisation

**Description**  
The rotation refinement uses `so3_exp(Δaxis)` and composes it with the input rotation matrix. When the gate grows, small numerical errors in the exponential map or `torch.autograd.grad` through repeated camera reprojection can produce NaN/Inf gradients, especially on mixed-precision training.

**Mitigation**
- Use the existing `motionflow_mv.calibration.perturb.so3_exp` implementation, which is already used in `camera_refinement_v26.py`.
- Compute the camera update once per frame and detach it from the pose refinement branch where possible, breaking the expensive second-order path.
- Add `torch.autograd.set_detect_anomaly(True)` in smoke tests and clip gradients globally (`max_norm=1.0`) during the first epoch of the full run.

## Risk 4: Physical losses dominate and suppress pose accuracy

**Description**  
Bone-length and floor terms can pull the pose toward an anatomically plausible but metrically wrong configuration if their weights are too high. On benchmark data with accurate ground-truth calibration, this can actually increase `MPJPE`.

**Mitigation**
- Default `v52_psc_bone_weight=0.01` and `v52_psc_floor_weight=0.01`, which is an order of magnitude smaller than the reprojection weight.
- Implement `v52_psc_warmup_epochs` to linearly ramp the physical terms from 0 to the target weight, giving the pose branch time to stabilise.
- Gate the pose residual through the same `λ = tanh(gate)` used for cameras, keeping corrections small until the network has learned reliable physical cues.

## Risk 5: Interaction with v26 camera refinement and v28 physical-space alignment

**Description**  
v26 already refines cameras via gradient descent on reprojection, and v28 applies a learned physical-space residual. Running v52 alongside them may create redundant or contradictory corrections: v26 pushes cameras to fit 2-D, v52 pushes cameras to fit physical constraints, and v28 then re-adjusts the pose.

**Mitigation**
- Treat v52 as the *higher-level* module: disable v26 when v52 is enabled (`assert not (use_v26 and use_v52)` or at least document the interaction).
- Run v52 **before** v28 so that v28 operates on a camera-corrected, physically consistent pose rather than fighting the original calibration.
- Add an ablation flag `v52_psc_refine_cameras_only` that updates cameras but leaves the pose to v28, letting the team compare the two strategies quickly.
