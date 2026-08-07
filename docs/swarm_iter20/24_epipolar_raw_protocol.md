# Epipolar raw-output protocol

## Finding

The classic Epipolar model inherited `return_raw` but its custom forward never
returned the raw triangulation. The training route also failed to pass the flag,
so raw reprojection loss consumed a PP or focal diagnostic instead of a pose.
The full Epipolar Bias V2 training route had the same missing constructor flag.

## Fix

The classic model now follows the anchor tuple contract and appends raw 3D after
PP/focal diagnostics. The trainer passes `return_raw` to both affected routes.
Default output tuples remain unchanged.

## CPU evidence

Focused raw-output contract tests cover the sequence trainer slot and the public
single-frame shape: `16 passed`. No GPU experiment was run.
