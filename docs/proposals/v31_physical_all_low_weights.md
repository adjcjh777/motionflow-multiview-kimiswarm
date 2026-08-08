# v31 `physical_all_low_weights`

## Problem statement

v29/v30 added a training-time physical-space temporal loss (floor contact, bone-length temporal smoothness, and centre-of-mass jitter) on top of the v25 geometry-fusion stack. The v30 smoke run uses weights of `floor=0.01`, `bone_temporal=0.01`, and `com_jitter=0.001`. Early v29 ablations varied one weight at a time (e.g. `v29r_physical_floor_low`, `v29s_physical_bone_low`), but none asked whether *all three* terms should be scaled down together. Because the physical loss is applied in world-space millimetres, even small weights can dominate the main pose loss once the model has reached moderate accuracy. If the loss is too strong it can regularise away subtle pose details; if it is too weak it does nothing. We therefore need an ablation that uniformly lowers every physical weight to find a minimal, non-harmful regularisation level.

## Proposed change

Run the **v30 hierarchical encoder + v25 geometry-fusion stack** with the v29 physical-space temporal loss enabled, but with all physical weights set to one tenth of the v30 smoke values and a short warmup ramp. Concretely:

- Keep `--use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1`.
- Keep the v25 geometry fusion stack, deformable attention, set-view aggregator, and variable-view training identical to the v30 smoke run.
- Set physical loss weights to `--v29_floor_loss_weight 0.001 --v29_bone_temporal_weight 0.001 --v29_com_jitter_weight 0.0001`.
- Add `--v29_physical_loss_warmup_epochs 2` so the prior ramps in smoothly rather than fighting the main loss during the first epochs.
- Do **not** use any TTE module.

This tests whether a much weaker physical prior can still stabilise training and reduce overfitting without distorting the learned pose distribution.

## Expected impact on `val_MPJPE` / overfitting

With all physical weights reduced, the loss should behave as a gentle constraint rather than a strong prior. We expect epoch-1 `val_MPJPE` to be very close to the v30 smoke run, because the physical loss contributes little during warmup and is small even after. The key signal is the *shape* of the validation curve after epoch 1: if overfitting is reduced relative to v30 smoke while validation MPJPE stays flat or improves, the physical loss is beneficial even at low weights. If the curve tracks v30 smoke exactly, the physical weights can be left at the v30 defaults; if it overfits more, the physical loss was providing needed regularisation. We do not expect a large single-epoch MPJPE drop from this ablation alone.

## Main risk

The main risk is that **lowering all weights simultaneously removes too much regularisation**, reproducing the v29a-style overfitting (epoch-1 drop followed by later-epoch degradation) despite the v30 hardening. If this happens, the sweep will need to isolate which physical term matters most or whether a higher total physical loss weight is required. Conversely, because the weights are already small, there is limited downside: the run is cheap and the result cleanly bounds the sensitivity of the v30 stack to the physical prior.
