# v31 Proposal: `physical_collision_penalty`

## Problem Statement

The v29/v30 physical-space temporal loss improves pose plausibility through foot-floor contact, bone-length temporal smoothness, and center-of-mass jitter, but it does not prevent self-intersection of body parts. Predicted skeletons can still place an arm inside the torso, cross the legs impossibly, or push the head through the upper chest, especially when the model overfits to a small training set. Adding an explicit self-collision penalty would make the 3-D output anatomically safer and should reduce rare-but-costly failure modes that inflate val_MPJPE.

## Concrete Proposed Change

Add a new auxiliary loss module `PhysicalCollisionPenaltyV31` in `motionflow_mv/losses/physical_collision_penalty_v31.py`.

* **Representation.** Each bone is treated as a capsule: a line segment between a joint and its parent, with a fixed radius (default 0.07 m). Bones are derived from the existing skeleton `parents` list.
* **Penalty.** For every pair of non-adjacent bones (bones that do not share a joint and are not parent-child), compute the minimum distance between the two line segments. If the distance is smaller than the sum of the two capsule radii plus a safety margin, apply a squared repulsion term.
* **Warmup.** The loss is gated by the same physical-loss warmup used for v29 floor/bone/com losses, so it only ramps up after the pose network has learned a coarse solution. Recommended 1-epoch warmup for smoke tests, 2-3 epochs for full runs.
* **Integration.** Wire the module through `OmniMultiViewFusionV5` under a new flag `--use_physical_collision_penalty_v31`, with hyperparameters `--v31_collision_loss_weight` (default 0.001), `--v31_collision_margin` (default 0.05 m), `--v31_collision_bone_radius` (default 0.07 m), and `--v31_collision_warmup_epochs` (default 1).

The new loss is applied to the final 3-D prediction `pred_3d` inside the model, next to the existing `PhysicalSpaceTemporalLossV29`. It is disabled at inference and does not affect TTE (which remains disabled per project constraint).

## Expected Impact on val_MPJPE / Overfitting

* **val_MPJPE.** A small improvement (1-3 mm) is expected on the full validation set, mainly from suppressing gross self-penetrations. The effect is largest on poses with extreme articulation (sitting, crouching) where the current model produces impossible bone intersections.
* **Overfitting.** Because the penalty acts as a regularizer that is independent of the ground-truth labels, it should reduce the train-val gap on small smoke configs such as the v30 200-clip run. It will not help if overfitting is driven by capacity or by the hierarchical encoder alone.
* **Physical plausibility.** Qualitatively, predicted skeletons should exhibit fewer arm-through-torso and leg-crossing artifacts, which matters for downstream paper figures and few-view robustness.

## Main Risk

The largest risk is **over-penalizing valid near-contact poses**: two hands near the face, arms folded across the chest, or a hand resting on a hip can all produce small bone-bone distances even though they are anatomically correct. If the margin or bone radius is too large, the model may push these joints apart, increasing val_MPJPE rather than decreasing it. Mitigation: start with a small radius (0.05 m), a conservative weight (0.0005), and a small margin (0.03 m), then scale up only if the smoke test shows no regression.

A secondary risk is **computational cost**: the pairwise line-segment distance is O(B T N²_bones) and will add noticeable overhead for `J=28` MPI skeletons. If profiling shows a slowdown larger than ~5%, restrict the penalty to a smaller set of collision-prone body parts (torso vs. arms/legs) rather than all non-adjacent pairs.

## Launch

A local RTX 4090 smoke launch script is provided in `scripts/launch_v31_physical_collision_penalty_local4090.sh` and a matching config in `configs/v31_physical_collision_penalty.yaml`. The run requires wiring the new flag into `OmniMultiViewFusionV5` and `experiments/train_omniview_fusion_v5_webbridge_multi.py`; the standalone loss module itself is in `motionflow_mv/losses/physical_collision_penalty_v31.py`.
