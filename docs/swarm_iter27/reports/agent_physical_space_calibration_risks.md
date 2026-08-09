# v53 Physical-Space Calibration — Risk Report

This document lists the main technical risks for the proposed `physical_space_calibration_v53` module and concrete mitigations for each.

## Risk 1: Floor-plane assumption is violated

**Description:** PSC assumes the lowest foot/ankle joints lie on a ground plane (`floor_height`). In clips where the subject is jumping, lying down, or captured from an elevated platform, forcing a floor contact constraint can pull valid 3-D poses toward an incorrect plane and raise MPJPE.

**Mitigation:**
- Make the floor head **gated** and optional via `v53_psc_use_floor`.
- Initialize the floor loss weight to a small value (`v53_psc_floor_weight=0.01`) and apply `v53_psc_warmup_epochs` so the constraint activates only after the pose backbone is stable.
- Use a **soft floor loss** that penalizes feet only below the estimated plane (`clamp(f_t - h, 0)^2`), not a hard projection.

## Risk 2: Canonical bone-length prior conflicts with a new dataset

**Description:** The bone-length calibration head uses a learned canonical skeleton. If the canonical lengths are initialized from H36M statistics but the validation set uses a different skeleton (MPI-INF-3DHP 28 joints, 3DPW, etc.), the prior can bias bone lengths in the wrong direction and hurt accuracy.

**Mitigation:**
- Initialize canonical lengths from the **training-set empirical mean** per domain rather than a hard-coded H36M skeleton.
- Use `domain_id` to select a per-domain canonical skeleton when `v48_domain_generalization` is enabled.
- Keep the bone-length correction gated and identity-at-init; let the network learn to ignore the prior when residuals are large.

## Risk 3: Identity-at-init fails and v52 checkpoints regress

**Description:** If the residual gate or the final MLP layers are not initialized correctly, enabling PSC on a trained v52 checkpoint could change the output pose by more than the planned `< 0.1 mm`, breaking warm-start compatibility.

**Mitigation:**
- Zero-initialize the final linear layer of the physical residual MLP.
- Initialize the residual gate logit to `-6.0` so `σ(gate) ≈ 0.0025` at start.
- Add a unit test that loads a v52 checkpoint, enables PSC, and asserts `||pred_v53 - pred_v52||_∞ < 1e-4 mm`.

## Risk 4: Extra parameters and auxiliary losses cause overfitting

**Description:** PSC adds a residual MLP, canonical bone-length parameters, and three auxiliary losses (floor, bone, reprojection). On small smoke datasets this can overfit and increase validation MPJPE.

**Mitigation:**
- Keep the module small: `v53_psc_hidden=64` and `v53_psc_n_layers=2` by default.
- Use loss weights an order of magnitude smaller than the main pose loss (`0.01` or below).
- Monitor validation curves; if MPJPE rises, disable the floor head or drop `v53_psc_reproj_weight`.

## Risk 5: Interaction with existing physical losses (v28 / v31 / v40)

**Description:** PSC runs after v52 and before v28 physical-space alignment and v31 physical collision penalty. Double-counting physical constraints can over-constrain the pose and degrade performance, especially when `v28_floor_loss_weight` and `v31_collision_loss_weight` are also non-zero.

**Mitigation:**
- Document the ordering clearly: `v52 → v53 PSC → residual MLP → v28/v31 physical losses`.
- When PSC is enabled, recommend reducing or zeroing `v28_floor_loss_weight` because PSC already enforces a floor constraint.
- Run an ablation matrix on smoke configs comparing (v52 only), (v52+v53), (v52+v28), and (v52+v53+v28).
