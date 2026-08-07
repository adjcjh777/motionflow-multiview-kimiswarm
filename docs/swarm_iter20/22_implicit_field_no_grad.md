# Implicit-field no-grad inference

## Finding

The implicit refiner needs a spatial derivative of its learned field. Under
`torch.no_grad()`, that local derivative graph was never created, so ordinary
evaluation failed inside `torch.autograd.grad`.

## Fix

The refiner records the caller's grad mode and temporarily enables autograd only
for the field derivative. Training keeps the existing graph-building behavior;
no-grad inference uses the derivative numerically and returns detached pose and
field outputs.

## CPU evidence

The implicit suite now includes two-step no-grad inference with field output:
`4 passed`. No GPU experiment was run.
