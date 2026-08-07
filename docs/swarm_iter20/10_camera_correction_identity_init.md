# Swarm Iteration 20 — Camera-Correction Identity Initialization

## Contract defect

`PrincipalPointCorrection`, legacy `IntrinsicCorrection`, and
`CameraCentricCoordinateTransform` describe their residual corrections as
identity or near-identity at initialization. Their final `nn.Linear` layers
previously kept PyTorch's random default weights and biases. `tanh` bounds a
random output; it does not make that output zero.

This matters most on the production raw-observation path. Its descriptor
contains unnormalized pixel and focal values in the hundreds. The existing
paper draft separately records that a trained PP head saturated at its
`max_offset` regardless of input while the residual MLP compensated. Random
initialization is not proven to be the sole cause of that trained state, but it
creates the same saturation tendency at step zero.

## CPU distribution diagnostic

`experiments/prototypes/swarm_iter20/camera_correction_init_probe.py` mirrors
the distribution used by default `nn.Linear` initialization with NumPy and a
typical raw descriptor `(320,240,1,320,240,800,800,0)`. Across 1000 independent
initializations, it reports:

- mean initial absolute PP correction: `19.49 px` out of `20 px`;
- more than `93%` of PP components exceed `19 px`;
- mean absolute focal-scale delta: `0.0977` out of `0.1`.

This is a distributional diagnostic, not a bitwise reproduction of PyTorch's
random-number stream and not a trained-model result. A separate Torch 2.4 CPU
100-seed audit corroborated the raw-path result (`19.48 px` mean absolute PP
delta); the repaired focused Torch tests pass on CPU.

## Minimal repair

Only each correction head's final `Linear` weight and bias are initialized to
zero. For every input this gives:

- `pp_delta = 0` and corrected `K = K`;
- `focal_scale = 1`;
- `delta_R = I`, corrected `R = R`, corrected `t = t`;
- camera-centric depth scale `= 1`.

`CameraCentricCoordinateTransform.max_rot_offset_deg` is a per-component so(3)
bound, so its total angle can reach `sqrt(3)` times that value. This pre-existing
parameter meaning is now documented but is not changed in this repair.

The parameter names and shapes do not change. A complete existing checkpoint
overwrites the initialization and remains structurally loadable. A partial
warm start that lacks one of these heads now leaves that missing residual at
identity instead of a random camera perturbation. The older shared focal-head
schema from commit `c7ac9e3` already has a separate output-shape incompatibility;
this initialization repair does not change or solve that historical boundary.

No GPU experiment was run. The NumPy probe and static contract do not establish
an MPJPE improvement; new training or same-checkpoint evaluation is still
required for an accuracy claim.
