# Implicit-field diagnostic output

## Finding

The full model exposed `return_field=True`, which made its residual refiner
return `(delta, field_values)`. The inherited anchor forward then tried to add
that tuple to the raw pose and failed with a type error.

## Fix

The refiner can now capture its final field values without changing its tensor
return. The full model uses that internal path and appends the reshaped field as
the final diagnostic slot. Direct refiner `return_field=True` behavior remains
unchanged, and the default full-model output is unchanged.

## CPU evidence

The implicit refiner/full-model suite, including PP + field backward, passed
(`3 passed`). No GPU experiment was run.
