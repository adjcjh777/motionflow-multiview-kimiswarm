# v31: physical_com_only_warmup

## Problem statement

v29 introduced a `PhysicalSpaceTemporalLossV29` that mixes three priors:
foot-floor penetration, bone-length temporal consistency, and center-of-mass
(COM) jitter.  Early v29/v30 runs showed two things:

1. **Physical loss needs warmup.**  Applying the full physical loss from epoch 1
tends to over-constrain the network before the 3-D triangulation baseline has
converged, pulling val_MPJPE up.
2. **Not all priors are equally safe.**  Floor loss assumes a reliable ground
plane and foot-joint indices; bone-temporal loss can over-regularize motion by
penalising legitimate bone-length changes.  The COM-jitter term is the most
agnostic: it only penalises unrealistic high-frequency acceleration of the mean
joint position and needs no skeleton-specific foot labels.

v29a overfits after epoch 1, and v30 hardened the encoder with stochastic
depth, gated residuals, and cross-scale fusion.  We therefore want the smallest
possible physical prior that still injects temporal smoothness, warmed up so it
only engages once pose estimation is stable.

## Concrete proposed change

Run the v30 hierarchical encoder (`use_hierarchical_multiview_v30`) and enable
`use_physical_space_temporal_loss_v29`, but **zero the floor and bone-temporal
weights and keep only the COM-jitter term**:

```text
--use_physical_space_temporal_loss_v29
--v29_floor_loss_weight 0.0
--v29_bone_temporal_weight 0.0
--v29_com_jitter_weight 0.01
--v29_physical_loss_warmup_epochs 2
```

All other settings mirror the v30 smoke/full baseline (v18 deformable
cross-view attention, v25 geometry fusion, variable-view training, set-view
aggregator, mixed H36M+MPI loader).  TTE remains disabled because the v29/v30
TTE module is broken and must not be used.

## Expected impact

* **val_MPJPE:** should track close to the v30 hierarchical-only baseline.  The
COM term is weak and data-agnostic, so it is unlikely to raise error; if the
baseline suffers from high-frequency jitter, it may slightly improve smooth
metrics without hurting 3-D accuracy.
* **Overfitting:** COM is a mild regulariser, and the warmup keeps it at zero
for the first two epochs.  This should delay the v29a-style overfitting that
appears after epoch 1 when stronger physical priors are active from the start.
* **Training stability:** COM needs no foot-joint indices or floor plane, so
it is safe across both H36M and MPI-INF-3DHP without dataset-specific tuning.

## Main risk

The COM term may be **too weak to produce a measurable improvement**, making
this variant indistinguishable from the v30 baseline.  If so, the next step is
either to increase `v29_com_jitter_weight` or to re-introduce the bone-temporal
term after a longer warmup, while keeping floor loss disabled because of its
floor-plane assumptions.
