# Delegating wrapper single-frame contract

## Finding

Four wrappers converted 4D single-frame inputs to 5D before calling their
parent. The parent therefore returned sequence-shaped tensors, while each
wrapper squeezed only some outputs. One tuple could mix `(B, J, 3)` with
weights, covariance, or raw poses that still contained `T=1`.

## Fix

- Bundle adjustment passes the original input to the parent and creates a
  sequence view only for DBA.
- Bayesian visibility passes the original input to its Bayesian parent and
  reshapes only its own visibility tensor.
- Kinematic chain replaces only the parent's pose slot.
- Multi-person association delegates both 4D and 5D single-person inputs
  directly to the parent.

No tuple slots were reordered and no model parameters changed.

## CPU evidence

Focused wrapper, physics, visibility, and raw-contract tests: `21 passed`.
No GPU experiment was run.
