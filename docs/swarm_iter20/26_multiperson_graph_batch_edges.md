# Multi-person graph batch edges

## Finding

The association graph flattened all `N` samples but used edge indices for only
one graph. Consequently, only the first sample received graph messages; later
samples passed through residual LayerNorm without association updates.

## Fix

Each graph now receives a node-index offset of `n * (V * P * J)`, and edge types
are repeated once per sample. The message-passing equations are otherwise
unchanged.

## CPU evidence

A two-sample graph now matches an independent forward of its second sample;
focused graph and wrapper tests report `5 passed`. No GPU experiment was run.
