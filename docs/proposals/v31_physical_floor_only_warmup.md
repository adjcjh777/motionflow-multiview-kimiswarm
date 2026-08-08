# v31: physical_floor_only_warmup

## Problem statement

v29/v30 experiments show that the physical-space temporal loss can help, but it is also fragile. The v29 sweep includes a `v29w_floor_only` run, yet even that variant applies the loss at full strength from epoch 0. The broader pattern from v29/v30 is that **physical loss needs warmup**: when bone-temporal and COM-jitter terms are active early, they force strong temporal smoothness on top of still-noisy pose estimates, which distorts limb dynamics and can lock the model into a locally smooth but inaccurate basin. v29a overfits after epoch 1, and runs such as `v26+UDP+v28 full` degraded once physical-space alignment weights became too large too soon.

The current `PhysicalSpaceTemporalLossV29` already supports `warmup_epochs`, but the existing v31 smoke config warms up for only one epoch and keeps all three loss terms active. We need to isolate the most reliable physical prior—the floor/foot-contact constraint—and ramp it in gradually so the model first learns coarse pose from reprojection and 3-D losses, then is nudged toward physically plausible foot placement.

## Concrete proposed change

Train a v30-hardened model (`use_hierarchical_multiview_v30`) with the physical-space temporal loss enabled but restricted to the **floor term only**:

- `v29_floor_loss_weight = 0.01`
- `v29_bone_temporal_weight = 0.0`
- `v29_com_jitter_weight = 0.0`
- `v29_physical_loss_warmup_epochs = 3`

This is a strict ablation from the current v30 smoke recipe that uses `0.01 / 0.01 / 0.001` with a one-epoch warmup. All other settings mirror the v30 smoke/local baseline (v18 deformable attention, v25 geometry fusion, set-view aggregator, variable-view training, early stopping). No TTE module is used, in line with the hard constraint that TTE is broken.

The warmup is implemented by the existing `PhysicalSpaceTemporalLossV29.set_epoch` path: the floor loss is scaled linearly from 0 to 1 over the first three epochs. Because the floor term depends only on foot-joint height relative to a fixed floor plane, it adds a mild, stable regularizer during early training rather than a strong temporal smoothing force.

## Expected impact on val_MPJPE / overfitting

- **Stability:** Removing bone-temporal and COM-jitter terms should remove the largest sources of early-gradient distortion. We expect fewer epoch-1/2 regressions.
- **val_MPJPE:** If the floor prior is well aligned with gravity, it should improve foot placement and global pose height with minimal side effects. A modest improvement over the v30 baseline (current smoke ~28 mm range) is plausible, but the main signal is whether the run sustains low validation error beyond epoch 3 instead of overfitting.
- **Overfitting:** The floor-only term is far weaker than the full physical loss, so it should not dominate capacity. Combined with a 3-epoch warmup, it should reduce the sharp v29a-style overfit after epoch 1.

## Main risk

The floor loss assumes the canonical gravity direction and a floor plane at a fixed height. If WebBridge data preprocessing shifts the root height or the foot-joint selection (`foot_joint_indices`) does not robustly identify true foot joints across 17-joint H36M and MPI skeletons, the loss can bias global translation rather than improve it. The warmup mitigates but does not eliminate this; we should monitor the `floor` loss term and val_MPJPE jointly. If the floor term does not drop consistently, the loss weight or foot-joint set may need tuning.
