# v31: physical_bone_only_warmup

## Problem statement

v29 introduced `PhysicalSpaceTemporalLossV29`, a training-time regulariser that
mixes three physical priors: foot-floor penetration, bone-length temporal
consistency, and center-of-mass (COM) jitter.  Early v29/v30 runs showed two
patterns:

1. **Physical loss needs warmup.**  Applying the full physical loss from epoch
   1 over-constrains the network before the 2-D-to-3-D triangulation baseline has
   converged, raising val_MPJPE.
2. **The priors have very different failure modes.**  Floor loss assumes a
   reliable gravity-aligned ground plane and correct foot-joint labels.
   COM-jitter loss is safe but extremely weak.  The bone-temporal term sits in
   the middle: it encodes a real skeleton prior (limb lengths should change
   smoothly over time) and needs no floor assumption, but it can over-regularise
   fast motions if it is turned on too early or too strongly.

v30 hardened the hierarchical encoder with stochastic depth, gated residuals,
and cross-scale fusion, which reduced the v29a-style overfitting.  We therefore
want to test the smallest geometrically meaningful physical prior, isolated and
warmed up, on top of the stable v30 encoder.

## Concrete proposed change

Run the v30 hierarchical encoder (`use_hierarchical_multiview_v30`) and enable
`use_physical_space_temporal_loss_v29`, but **zero the floor and COM-jitter
weights and keep only the bone-temporal term**, with a short warmup:

```text
--use_hierarchical_multiview_v30
--v30_n_part_layers 2
--v30_stochastic_depth_prob 0.1
--use_physical_space_temporal_loss_v29
--v29_floor_loss_weight 0.0
--v29_bone_temporal_weight 0.01
--v29_com_jitter_weight 0.0
--v29_physical_loss_warmup_epochs 3
```

All other settings mirror the v30 smoke/full baseline: v18 deformable cross-view
attention, v25 geometry fusion, variable-view training, set-view aggregator, and
mixed H36M+MPI loader.  TTE remains disabled because the v29/v30 TTE module is
broken and must not be used.

The bone-temporal loss is implemented in `PhysicalSpaceTemporalLossV29` in
`motionflow_mv/fusion/self_evolving_hierarchical_multiview_v29.py`; the warmup
is handled by `model.set_epoch()` in `motionflow_mv/training/trainer_v2.py`.

## Expected impact

* **val_MPJPE:** the prior directly regularises limb-length changes over time,
  which should reduce high-frequency jitter in the predicted 3-D poses.  If the
  v30 hierarchical-only baseline already suffers from jitter, this can lower
  val_MPJPE by a few millimetres; if the baseline is already smooth, the effect
  will be smaller.
* **Overfitting:** the warmup keeps the bone term at zero for the first three
  epochs, so the network first learns the 2-D-to-3-D mapping without physical
  distortion.  Once the prior ramps in, it should act as a mild structural
  regulariser and delay the v29a-style overfitting that appears after epoch 1.
* **Training stability:** unlike the floor term, the bone-temporal term needs
  no foot-joint labels or ground-plane assumption, so it is equally applicable
  to H36M and MPI-INF-3DHP skeletons (both parent lists are already supported by
  the loss class).

## Main risk

The bone-temporal loss can **over-regularise fast, legitimate motion** (e.g.,
arm swings or jumps) if its weight is too high relative to the warmup length.
If this happens, val_MPJPE may rise because the model is forced to keep bone
lengths artificially constant.  The mitigation is either to lower
`v29_bone_temporal_weight` or to lengthen the warmup so the prior only engages
after the pose baseline is fully stable.
