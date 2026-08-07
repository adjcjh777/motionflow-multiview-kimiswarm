# TemporalMixed single-frame outputs

## Finding

All three TemporalMixed forwards accepted 4D input by adding `T=1`, but never
removed that synthetic axis. Pose and joint mask therefore returned sequence
shapes for a single frame. In the PP variant, auxiliary PP/focal tensors already
used the correct view axis, producing a mixed-rank tuple.

## Fix

Each forward now records whether the input was 4D and removes only `dim=1` from
the final pose and mask. PP and focal outputs are untouched because their second
axis is the view axis. The 5D sequence contract is unchanged.

## CPU evidence

Focused tests for base, residual, and PP+focal variants: `3 passed`. No GPU
experiment was run.
