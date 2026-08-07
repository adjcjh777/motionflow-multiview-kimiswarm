# Bayesian PP diagnostic flag

## Finding

Bayesian triangulation accepted `return_pp_delta=False` but always forced the
parent flag to `True`. Default calls therefore returned an undocumented PP slot,
and subclasses inherited the same tuple mismatch.

## Fix

The constructor now forwards the requested flag. Existing smoke paths that
explicitly consume PP diagnostics request them explicitly; default V2 and V3
tests exercise the shorter tuples.

The correction layer still runs and affects geometry. Only diagnostic exposure
changes.

## CPU evidence

Focused Bayesian triangulation suites cover V2 and V3 flag behavior:
`17 passed, 1 skipped` (the existing CUDA-only case). No GPU experiment was run.
