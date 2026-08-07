# Profiler raw-output protocol

## Finding

The instrumented Bayesian V2 forward mirrored the production model but omitted
its `return_raw` branch. Requesting raw output therefore had no effect in the
profiler class.

## Fix

The existing raw triangulation is reshaped like the prediction and inserted
immediately before the scalar epipolar loss, matching production tuple order.

## CPU evidence

A focused tuple-contract suite covers the instrumented path: `6 passed`. No GPU
experiment was run.
