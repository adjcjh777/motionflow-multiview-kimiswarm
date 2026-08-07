# Custom raw-output protocol

## Finding

Action-aware, semantic-action and Epipolar Dynamic Gate subclasses inherited
`return_raw` but their custom forwards never returned the existing raw DLT pose.
The dynamic Epipolar gate also dropped visibility whenever PP diagnostics were
disabled.

## Fix

- Action-aware and semantic-action variants now expose raw 3D without changing
  their default tuples.
- Epipolar Dynamic Gate places raw and visibility before the final gate pair, so
  gate diagnostics keep stable `-2/-1` slots with or without PP output.

## CPU evidence

Focused single-frame and tuple-contract tests exercise these paths: `19 passed`.
No GPU experiment was run.
