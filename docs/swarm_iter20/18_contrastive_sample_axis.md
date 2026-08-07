# Contrastive sample-axis contract

## Finding

The explicit contrastive feature path already flattened time into its sample
axis `(B*T, V, J, d)`. For a 4D input it then called `squeeze(0)`, so only the
`B=1` case lost the required sample axis and failed in the contrastive loss.

## Fix

The helper now always returns `(B*T, V, J, d)`. No caller accepts a 3D feature
tensor, and the hook-based training path already used this layout.

## CPU evidence

The focused contrastive suite, including a 4D `B=1` regression: `4 passed`.
No GPU experiment was run.
