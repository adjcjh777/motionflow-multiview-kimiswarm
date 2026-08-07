# Spatiotemporal principal-point shapes

## Finding

For a 4D single-frame input, the model removed `T=1` from pose and weights but
returned principal-point and focal outputs as `(B, 1, V, ...)`. For sequence
training, the model returned `(B, T, V, 2)` while the trainer flattened its
target to `(B*T, V, 2)`, so PP loss shapes did not agree.

## Fix

The model now removes only the explicit time axis from PP/focal outputs for
4D input. The trainer keeps its PP target as `(B, T, V, 2)`. Sequence output
layout and tuple ordering are unchanged.

## CPU evidence

`tests/test_spatiotemporal_pp_model.py`: `6 passed`. No GPU experiment was run.
