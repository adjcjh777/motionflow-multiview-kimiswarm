# Base visibility output contract

## Finding

`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` exposed
`return_visibility=True`, but its identity visibility hook returned the Python
scalar `1.0`. The forward path then called `.view(...)` on that scalar and
failed before returning an output.

## Fix

The identity hook now returns `torch.ones_like(confidences)`. Weighting remains
numerically unchanged, while the optional visibility output has the expected
per-view/per-joint layout on the same dtype and device as the input.

The regression test covers direct 4D single-frame and 5D one-frame sequence
inputs. It does not redefine tuple ordering when several optional return flags
are enabled together; that compatibility question remains a separate audit.

## CPU evidence

- Before: `AttributeError: 'float' object has no attribute 'view'`.
- After: focused visibility and raw-output tests pass (`12 passed`).
- No GPU experiment was run.
