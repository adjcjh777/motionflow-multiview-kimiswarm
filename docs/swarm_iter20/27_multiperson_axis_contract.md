# Multi-person axis contract

## Finding

The six-dimensional wrapper flattened people in `(B, P, T)` order but expanded
cameras in `(B, T, P)` order. Its association-graph entry and exit permutations
also mixed time, view, person, joint and feature axes. Predictions could differ
from independent per-person inference even when graph depth was zero.

## Fix

- Camera rows now follow the same `(B, P, T)` order as detections.
- Association features use the exact bridge
  `(B,P,T,V,J,d) <-> (B,T,V,P,J,d)`.
- PP, focal and visibility diagnostics restore explicit time/person axes.

The public multi-person shapes are `(B,T,P,J,3)`, `(B,T,V,P,J)`,
`(B,T,V,P,2)`, `(B,T,V,P)` and `(B,T,V,P,J)` respectively.

## CPU evidence

With zero graph layers, joint multi-person inference now matches independent
per-person inference while preserving all documented auxiliary axes; focused
tests report `6 passed`. No GPU experiment was run.
