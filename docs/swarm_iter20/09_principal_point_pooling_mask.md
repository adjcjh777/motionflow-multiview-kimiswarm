# Swarm Iteration 20 — Principal-Point Pooling Mask

## Static finding

`PrincipalPointCorrection` added `1e-8` to every joint weight before pooling.
Consequently, an all-zero feature map returned the unweighted feature mean, and
an all-zero raw confidence map returned the unweighted pixel mean. Zero-weight
joints were therefore not exact masks.

The raw-observation path also multiplied its supplied `weights` by the input
confidence. The 31 primary callers already pass `weights=confidences`, so the
effective historical weight was `confidence^2`, despite the API describing
`weights` as the pooling weights.

This directly affects confidence-only augmentation: training retains the real
pixel coordinates while setting confidence to zero, whereas variable-view
inference supplies `(0,0,0)`. The historical principal-point descriptor could
read the retained training pixels from an otherwise inactive view. Other
attention, graph, and residual paths still have the broader masking mismatch
described in `08_variable_view_failure_audit.md`.

## Minimal contract repair

Both pooling helpers now treat `weights` as the final weights, apply them once,
and replace only an exactly zero total weight with a denominator of one. Thus:

- a zero-weight joint has no contribution;
- an all-zero weighted pool returns zero instead of an unweighted mean;
- arbitrarily small positive totals remain exact weighted means;
- soft confidence is applied once rather than squared.

For the raw path, an all-zero view now contributes `p_mean=(0,0)`. The complete
descriptor still includes camera intrinsics, so this change does not assert
that the learned principal-point delta itself must be zero.

## CPU diagnostic

`experiments/prototypes/swarm_iter20/principal_point_pooling_probe.py` reproduces
three exact algebra cases:

| Case | Historical | Corrected |
|---|---:|---:|
| all-zero feature weights | `(51, 510)` | `(0, 0)` |
| all-zero raw confidence | `(6, 60)` | `(0, 0)` |
| soft confidence `[0.5, 1]` | `(8.4, 84)` via `confidence^2` | `(7.3333, 73.3333)` via confidence once |

The implementation has no parameter or state-dict shape change, so existing
checkpoints still load strictly. It is not numerically backward-compatible for
non-binary confidence: historical metrics require re-evaluation, and training
with the repaired contract is preferable before making an accuracy claim.

No GPU experiment was run, and this audit makes no recovered-MPJPE claim.
