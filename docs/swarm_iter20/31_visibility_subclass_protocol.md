# Visibility protocol in custom subclasses

## Finding

Dynamic Gate computed and reshaped visibility but never returned it. ActionAware
accepted the inherited flag yet neither applied the visibility multiplier nor
exposed the diagnostic.

## Fix

- Dynamic Gate places visibility before its stable final gate pair.
- ActionAware applies the same visibility multiplier as the anchor and appends
  visibility after raw output.

Default tuples and default numerical paths remain unchanged.

## CPU evidence

Focused visibility and raw/gate tuple tests cover both subclasses: `22 passed`.
No GPU experiment was run.
