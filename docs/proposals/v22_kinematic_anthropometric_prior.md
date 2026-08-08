# V22: Kinematic Anthropometric Prior (KAP)

**Task identifier:** `design_v22_kinematic_anthropometric_prior`  
**Tracking issue:** #88  

## Motivation

The current best pipeline stacks v18 deformable cross-view attention, v19
temporal Perceiver refinement, and v21 neural bundle adjustment inside
`OmniMultiViewFusionV5`. Each of these modules improves geometric or temporal
reasoning, but none of them inject an explicit, learned anthropometric prior:

* Joints are still estimated independently once triangulation is finished.
* Bone-length constraints are only present when the trainer adds an auxiliary
  bone-length MSE loss against ground truth.
* There is no parametric regularizer that can fire during inference to keep
  implausible poses in check.

The SMPL-based v22 proposal (`docs/proposals/v22_smpl_prior.md`) was rejected
because the project has no SMPL/SMPL-X model files and no license to ship them.
This proposal replaces that external parametric body with a **lightweight, fully
learned Kinematic Anthropometric Prior** that:

* lives entirely in the existing codebase,
* uses the existing H36M 17-joint (or MPI-INF-3DHP 28-joint) parent list,
* is small enough to train on an A800 alongside other runs,
* and is gated by a single boolean flag inside `OmniMultiViewFusionV5`.

## Design principles

1. **No external assets.** No SMPL, no MANO, no pre-trained body models.
   Everything is a learnable PyTorch module.
2. **Per-frame, not per-clip.** The module operates on a single frame of 3D
   pose at a time, so it adds almost no memory and plays nicely with the
   existing temporal Perceiver (v19).
3. **Additive and warm-startable.** At init the residual branch outputs ~zero
   and the confidence gate is ~1, so a v21/v19/v18 checkpoint loads without
   disruption.
4. **Two loss terms only.**
   * Learned Gaussian bone-length negative log-likelihood.
   * Optional soft joint-angle limit penalty.
5. **Stack after v21, before v19.** The module refines the 3D skeleton after
   camera/pose refinement (v21) and before the temporal Perceiver (v19).

## Module

**File:** `motionflow_mv/fusion/kinematic_anthropometric_prior_v22.py`

```text
KinematicAnthropometricPrior(
    j: int = 17,
    d: int = 64,
    hidden: int = 64,
    residual_hidden: int = 128,
    use_angle_limit: bool = True,
    max_flexion_deg: float = 160.0,
    max_delta: float = 0.10,
)
```

### Inputs

* `feat_pooled`: `(B*T, J, d)` — the same per-joint pooled feature that already
  feeds the residual MLP in `omniview_fusion_v5.py`.
* `pred_3d`: `(B*T, J, 3)` — the current 3D pose estimate (typically the output
  of the residual/diffusion or v21 bundle-adjustment stage).

### Outputs

* `pred_3d_refined`: `(B*T, J, 3)` — the pose after the learned kinematic
  correction.
* `kap_loss`: scalar — the KAP loss to be added to the training objective.

### Components

1. **Learned bone-length prior**
   * Build bone edges from the existing parent list (`H36M_17_PARENTS` or
     `MPI_INF_3DHP_28_PARENTS`).
   * For each bone `e = (child, parent)` compute the 3D bone vector and length:
     ```
     b_e = pred_3d[child] - pred_3d[parent]
     l_e = ||b_e||_2
     ```
   * Learn per-bone parameters:
     * `bone_mu_e` — mean bone length.
     * `bone_logvar_e` — log-variance of bone length.
   * Negative log-likelihood:
     ```
     L_bone = mean_e 0.5 * ((l_e - mu_e)^2 / exp(logvar_e) + logvar_e)
     ```

2. **Optional soft joint-angle limit penalty**
   * Reuse `motionflow_mv.losses.kinematic_v15.joint_limit_loss`.
   * For any interior joint, penalize an interior angle larger than
     `max_flexion_deg`.
   * This is a soft constraint; it can be disabled with
     `use_angle_limit=False`.

3. **Kinematic prior features**
   * Convert the per-bone NLL into a per-joint feature by scattering the bone
     NLL to its child joint and to its parent joint.
   * For joints without a parent/child bone, pad with zero.
   * Run a tiny MLP to produce a 1-D (or 2-D) per-joint kinematic feature.

4. **Residual refinement branch**
   * Input: concatenate `[feat_pooled, pred_3d, kinematic_feature]` along the
     last dim.
   * Two-layer MLP → output 4 scalars per joint:
     * 3-D correction `delta_raw`.
     * 1-D confidence logit.
   * Gating:
     ```
     delta = tanh(delta_raw) * max_delta
     conf  = sigmoid(logit)
     pred_3d_refined = pred_3d + conf * delta
     ```
   * The final linear layer for `delta_raw` is initialized to near zero and the
     confidence bias is initialized to ~2 so the module starts as an identity
     map.

### Total KAP loss

```
L_kap = L_bone + w_angle * L_angle
```

where `L_angle` is the optional joint-limit loss and `w_angle` is a small scalar
(default 0.1).

## Integration with `OmniMultiViewFusionV5`

### 1. Add a new toggle

In `motionflow_mv/fusion/omniview_fusion_v5.py`:

```python
use_kinematic_anthropometric_prior_v22: bool = False,
kap_loss_weight: float = 0.01,
```

Instantiate the module inside `__init__`:

```python
self.use_kinematic_anthropometric_prior_v22 = use_kinematic_anthropometric_prior_v22
self.kap_loss_weight = kap_loss_weight
if self.use_kinematic_anthropometric_prior_v22:
    self.kinematic_anthropometric_prior_v22 = KinematicAnthropometricPrior(
        j=self.j,
        d=self.d,
        ...
    )
```

### 2. Hook into the forward pass

After the existing residual/diffusion refinement (and after v21 neural bundle
adjustment, if enabled), but before the optional final kinematic-chain refiner:

```python
# After residual/diffusion and v21, pred_3d is (B*T, J, 3)
if (
    self.use_kinematic_anthropometric_prior_v22
    and self.kinematic_anthropometric_prior_v22 is not None
):
    pred_3d, kap_loss = self.kinematic_anthropometric_prior_v22(
        feat_pooled, pred_3d
    )
    epi_loss = epi_loss + self.kap_loss_weight * kap_loss
```

Then continue with the existing optional `use_kinematic_refiner`, reshape to
`(B, T, J, 3)`, and the v19 temporal Perceiver.

### 3. Trainer wiring

In `experiments/train_omniview_fusion_v5_webbridge_multi.py`:

* Add CLI arguments:
  * `--use_kinematic_anthropometric_prior_v22`
  * `--kap_loss_weight`
  * `--kap_angle_limit_weight`
* Pass them to `build_model_from_args`.
* Optionally log `kap_loss` separately if the model returns it; otherwise fold
  it into `epi_loss` as shown above.

No other loss terms need to change. Existing `bone_loss_weight`,
`joint_limit_weight`, etc. remain independent and can be kept at their current
values.

## Training considerations

* **Initialization of `bone_mu`:** if ground-truth 3D statistics are available
  offline, initialize each `bone_mu_e` to the mean training-set length of that
  bone. Initialize `bone_logvar_e` to `log(0.05^2)`. If no statistics are
  pre-computed, initialize `bone_mu_e ≈ 0.25` m and let them learn.
* **Residual branch warm-start:** zero-initialize the final `delta_raw` layer
  and set the confidence bias to `+2.0` so the module starts as the identity.
* **Loss weight schedule:** start with `kap_loss_weight = 0.01` and
  `w_angle = 0.1`. Increase only if validation MPJPE improves.
* **A800 footprint:** the module is a few thousand parameters, has no
  attention over views/time, and no additional SMPL memory. It can run
  alongside existing experiments.
* **Stacking order:** place KAP **after** v21 so that bone lengths are computed
  on the camera-corrected pose, and **before** v19 so the temporal Perceiver
  sees the anthropometrically refined pose.

## Expected risks and mitigations

| Risk | Mitigation |
|------|------------|
| Bone-length prior overfits to the dataset scale. | Use per-bone log-variance; keep the weight small. |
| Angle limits hurt extreme poses (sports, gymnastics). | Keep the penalty soft and the weight small; allow disabling with `use_angle_limit=False`. |
| Conflicts with v21 camera updates. | Apply KAP after v21, so bone lengths reflect the corrected geometry. |
| Ground-truth bone lengths are not available at test time. | The prior is fully learned; no GT is needed at inference. |

## Test coverage

Add `tests/test_kinematic_anthropometric_prior_v22.py` covering:

* Forward shape `(B*T, J, 3) → (B*T, J, 3)` and scalar `kap_loss`.
* Gradient flow through `feat_pooled` and `pred_3d`.
* The module is ~identity at init: `||pred_refined - pred||` is small.
* Confidence is in `(0, 1)`.
* Works for both `J=17` and `J=28`.
* `use_angle_limit=True/False`.

Also add a toggle-on case to `tests/test_omniview_fusion_v5.py` to ensure the
flag integrates cleanly with the full v5 forward pass.

Run:

```bash
pytest tests/test_kinematic_anthropometric_prior_v22.py tests/test_omniview_fusion_v5.py -q
```

## Future work

* Learn a shared latent across bones (e.g. one scalar “body scale”) instead of
  independent per-bone means.
* Add a temporal consistency term on `bone_mu` via the v19 latent set.
* Use the learned prior for outlier rejection: joints with very high per-bone
  NLL can be masked before triangulation.

## References

* `motionflow_mv/fusion/omniview_fusion_v5.py`
* `motionflow_mv/fusion/neural_bundle_adjustment_v21.py`
* `motionflow_mv/fusion/temporal_perceiver_v19.py`
* `motionflow_mv/fusion/deformable_cross_view_attention.py`
* `motionflow_mv/losses/kinematic_v15.py`
* `motionflow_mv/fusion/graph_joint_relation.py`
