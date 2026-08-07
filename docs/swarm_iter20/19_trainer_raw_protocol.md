# Trainer raw-output protocol

## Findings

- The factorized trainer always passed `return_raw`, but the model constructor
  did not accept it, so the route failed during construction.
- Dynamic gate accepted the flag but omitted raw 3D; the trainer then treated
  gate logits as a pose.
- Hierarchical entropy appends a scalar loss last, while its trainer treated
  that scalar as raw 3D.

## Fixes

- Factorized supports raw output and appends it after existing PP/focal slots.
- Dynamic gate inserts raw immediately before its stable final gate pair; its
  trainer reads slot `-3`.
- The entropy trainer reads raw from slot `-2`; entropy remains last.

Default tuples and model parameters are unchanged.

## CPU evidence

Focused model and tuple-contract tests: `8 passed`. No GPU experiment was run.
