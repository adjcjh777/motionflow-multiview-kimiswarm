# Adaptive pyramid scale diagnostics

## Finding

`return_scale_weights=True` was a public model option, but the pyramid layer
discarded the scale weights it computed. The model then referenced an undefined
local variable and raised `NameError`.

## Fix

The layer keeps its default tensor-only API and optionally returns
`(fused, scale_weights)`. The model requests that diagnostic only when its flag
is enabled and removes the synthetic time axis for 4D input. Existing tuple
slots remain in place; scale weights are appended as documented.

## CPU evidence

The focused PP + scale diagnostic regression passed (`1 passed`). No GPU
experiment was run.
