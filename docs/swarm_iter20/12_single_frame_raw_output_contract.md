# Swarm Iteration 20 — Single-Frame Raw Output Contract

## Observed defect

Temporal fusion models accept either `(B,T,V,J,3)` clips or `(B,V,J,3)` single
frames. For single frames, public `pred` and `weights` remove the synthetic
`T=1` dimension. Nine independently implemented principal-point variants still
returned raw triangulation as `(B,1,J,3)`, producing incompatible shapes inside
one tuple.

The affected forward implementations were the base PP model, camera-centric,
canonical-skeleton, hierarchical, camera-conditioned, epipolar-bias-v2, splat,
visibility-transformer, and completion variants. Classes that inherit the base
forward receive the base repair automatically.

## Minimal repair

Each implementation now reshapes raw triangulation once to `(B,T,J,3)` and
applies the same single-frame squeeze already used for `pred`. Tuple order and
arity are unchanged. PP/focal outputs retain their existing flattened `(B*T,V)`
contract for clip input; this patch does not silently redefine those training
interfaces.

A focused Torch CPU test covers all nine independent forward implementations,
the raw-only return branch, and the combined PP+focal+raw clip tuple.

## Remaining audit lanes

This is the first batch of the output-contract audit. Several wrappers hide the
original 4D input from their parent by adding `T=1` before delegation; their
weights or auxiliary tensors therefore remain unsqueezed. The physics wrapper
also parses parent tuples positionally under incompatible flag combinations.
In addition, the base class's default visibility multiplier is a Python scalar
even though its public `return_visibility` path reshapes it as a tensor, and
some combinations of `return_visibility`, `return_pp_delta`, and `return_raw`
silently omit requested outputs. Those are separate tuple-protocol defects and
are not claimed fixed here.

The change adds no parameters and does not affect checkpoint loading. No GPU
experiment was run and no accuracy claim is made.
