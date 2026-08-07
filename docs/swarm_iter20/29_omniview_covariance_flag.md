# OmniView covariance output flag

## Finding

OmniView V2 and V3 accepted `return_covariance=False` but always returned the
covariance tensor. The flag therefore had no effect on tuple arity.

## Fix

Both models now append covariance only when requested. Their default remains
`True`, so all existing training and evaluation callers keep the same tuple.

## CPU evidence

One focused test covers the false branch for both prototypes: `1 passed`. No
GPU experiment was run.
