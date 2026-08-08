# v31: Skeleton-Residual Gate

## Problem statement

The v30 hardened hierarchical encoder stabilises training, but the downstream
3-D pose still relies on the skeleton-graph residual refiner to clean up
noisy distal joints. In `SkeletonGraphResidualRefiner` the correction is
scaled by a single scalar, `residual_scale`, so every joint receives the same
update magnitude. This is sub-optimal: torso joints are already well
triangulated and should stay close to the raw estimate, while wrists and
ankles are often noisy and would benefit from a larger learned correction.
A uniform scale either under-corrects distal joints or over-corrects the
torso, which can drive over-fitting after the first epoch (the v29a pattern).

## Concrete proposed change

Introduce a **per-joint residual gate** inside
`SkeletonGraphResidualRefiner`:

1. Replace `self.residual_scale = nn.Parameter(torch.tensor(0.01))` with
   `self.residual_gate = nn.Parameter(torch.full((j,), -6.0))`.
2. In `forward`, compute `gate = torch.sigmoid(self.residual_gate)` and
   reshape to `(1, J, 1)`.
3. Return `gate * self.output_proj(h)` instead of the scalar-scaled output.

The sigmoid initialised at `-6` keeps every gate near zero at start
(`sigmoid(-6) ≈ 0.002`), preserving the identity-at-init property of v30.
During training the gate for each joint learns how much of the graph
refiner’s correction to apply. The change touches only
`motionflow_mv/fusion/skeleton_graph_residual_refiner.py` and adds a
single boolean flag `use_skeleton_residual_gate_v31` to
`OmniMultiViewFusionV5` so the gate can be enabled as a drop-in ablation.

Training run: start from the v30a recipe (hierarchical encoder + physical
loss, no TTE) and set `--use_skeleton_residual_gate_v31`. Keep physical-loss
warm-up (`--v29_physical_loss_warmup_epochs 3`) and the v30 stochastic-depth
probability of 0.1.

## Expected impact

- **val_MPJPE**: small but consistent improvement (estimated 1–3 mm on
  WebBridge/H36M mixed). Distal joints should benefit most because the
  refiner can selectively increase their correction while leaving the
  torso anchored.
- **Over-fitting**: reduced compared with v29a. A per-joint gate acts as a
  structured regulariser: joints with reliable triangulation will keep
  their gate small, preventing the refiner from memorising training-set
  pose biases.
- **Compatibility**: the gate is orthogonal to the v30 hierarchical encoder
  and to physical-loss warm-up; it can be combined with both.

## Main risk

- **Per-joint gate drift**: if the torso gate grows too fast early in
  training, the model may unlearn the strong triangulation prior. The
  `-6` initial value and the existing `-0.01` output-projection
  zero-initialisation mitigate this, but the gate values should be logged
  per joint.
- **Dataset-specific gates**: with only 17/28 parameters, the risk is
  small, but if one dataset dominates the mixed loader the gates may
  over-fit to that skeleton distribution. Monitoring per-dataset
  val_MPJPE is recommended.

## Verification

After the code change, run the attached local 4090 smoke script
`scripts/launch_v31_skeleton_residual_gate_local4090.sh`. A healthy run
should show the gate magnitudes staying small for torso joints and growing
for distal joints, with val_MPJPE below the v30 smoke baseline.
