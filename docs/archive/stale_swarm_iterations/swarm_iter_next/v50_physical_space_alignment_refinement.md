# Physical-Space Alignment Refiner v50 (PSAR-v50)

## Architecture

`PhysicalSpaceAlignmentRefinerV50` is a lightweight, residual-guided post-triangulation refiner that enforces skeleton and floor consistency on the 3-D pose produced by the v25/v45 geometry-fusion stage. It takes the initial triangulated joints `J x 3`, per-joint reprojection residuals `R_j`, and the per-view reliability weights from v37/v46, and feeds them into a small per-joint MLP (2 layers, 64 hidden). The network predicts a bounded per-joint correction `Δ_j` that is added to the initial pose. The correction is gated by a sigmoid so the module is identity-at-init and cannot drift far from the geometric solution. In addition to the learned correction, an explicit physical prior is applied jointly: a bone-length regularizer, a floor-plane penalty derived from ankle/foot keypoints, and a left/right symmetry term. The floor plane is estimated online from the lowest stable foot joints across the batch, avoiding reliance on a hard-coded ground assumption. The module is inserted after triangulation and before the v46 sparse-view head, so sparse-view training benefits directly from physically plausible intermediate poses.

## Config flags

```yaml
use_v50_physical_space_alignment_refinement: false
v50_psa_hidden: 64
v50_psa_num_layers: 2
v50_psa_residual_input: true
v50_psa_identity_init: true
v50_psa_bone_weight: 0.01
v50_psa_floor_weight: 0.005
v50_psa_symmetry_weight: 0.005
v50_psa_loss_weight: 0.05
```

## Loss term

The total training loss adds an auxiliary physical-space term weighted by `v50_psa_loss_weight` (default `0.05`):

```
L_psa = v50_psa_loss_weight * (
    v50_psa_bone_weight   * L_bone_length +
    v50_psa_floor_weight  * L_floor +
    v50_psa_symmetry_weight * L_symmetry
)
```

`L_bone_length` is the mean absolute deviation of predicted bone lengths from a dataset bone-length prior; `L_floor` penalizes foot joints below the estimated floor plane plus a small margin; `L_symmetry` penalizes asymmetric limb lengths. The refiner’s own output `Δ_j` is also regularized by a tiny L2 penalty (`v50_psa_delta_l2 = 1e-4`) to keep it close to zero at initialization.

## Evaluation metric

Primary metrics are `val_MPJPE` on full views and `MPJPE@2/3/4` on sparse views. Diagnostic metrics include per-sequence bone-length error (`BLE`) and foot-to-floor error (`FHE`). We require `MPJPE@full` within `0.5 mm` of the v46 baseline; a successful refinement should also reduce `BLE` and `FHE` without regressing sparse-view metrics.

## Expected MPJPE impact

On the local RTX 4090 smoke we expect `val_MPJPE@full` to improve by `0.5–1.0 mm` and the larger gain to appear in sparse-view regimes: `MPJPE@2` and `MPJPE@3` should drop by `2–4 mm` because the physical prior compensates for the ambiguity of dropped views. Bone-length error is expected to fall by `15–20 %` and foot-floor penetration by `>30 %`.

## Main risk / mitigations

**Over-regularization / pose collapse.** If the physical prior is too strong the refiner may collapse the pose toward a mean skeleton and erase projective detail. **Mitigation:** keep the MLP small, initialize the output gate near zero, scale the loss by `v50_psa_loss_weight = 0.05`, and freeze the refiner for the first epoch so it only learns after triangulation has stabilized.

**Noisy floor-plane estimate.** The online floor estimate from foot joints can be unstable when feet are occluded or raised. **Mitigation:** estimate the plane from a temporal window of `≥3` frames, reject outlier foot candidates using RANSAC, and fall back to a dataset floor prior when fewer than two reliable feet are visible.

**Interaction with v46 view dropout.** Sparse-view dropout may expose the refiner to poses with large triangulation ambiguity, where the physical prior can dominate and create a biased correction. **Mitigation:** only apply the refiner during training when at least three views are present; at test time and in `MPJPE@2` evaluation, apply it after the pose is already triangulated from the available views.
