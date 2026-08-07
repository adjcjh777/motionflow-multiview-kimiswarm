# Physics wrapper output protocol

## Finding

The physics wrapper forced its parent to return raw 3D and then interpreted
`out[:3]` as `(pred, weights, raw)`. With principal-point or visibility output
enabled, slot 2 has a different meaning, so the wrapper silently returned the
wrong tensor and discarded requested outputs.

## Fix

The dynamics GRU only needs the refined pose. The wrapper now calls the parent
with the user's flags unchanged, applies dynamics to `out[0]`, and returns all
remaining parent slots without reordering them. Single-frame inputs receive a
temporary time axis only for the GRU.

## CPU evidence

- PP + focal + raw single-frame output keeps all five semantic slots.
- Visibility single-frame output remains the third slot.
- Focused physics, visibility, and raw-contract suite: `17 passed`.
- No GPU experiment was run.
