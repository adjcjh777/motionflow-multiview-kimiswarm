# Batched DLT weight-routing oracle

## Finding

The NumPy comparison loop used each batch's 2D points but always supplied
`weights[0]`. Its noiseless fixture still passed because every positive weighting
shared the same exact geometric solution, so the test did not verify batch
weight routing.

## Fix

The existing loop now uses `weights[n]`. A small deterministic noisy fixture
uses opposite view-weight patterns for two batches and compares the Torch result
with the same inhomogeneous NumPy least-squares formulation. The fixture is
chosen so reusing batch 0 weights changes batch 1 by more than 1 mm.

## CPU evidence

Extended batched-DLT suite: `11 passed, 1 skipped` (CUDA-only case skipped).
No GPU experiment was run.
