# P06 Temporal Velocity/Acceleration Loss (v2)

**Branch:** `feat/swarm-iter18-omniview`  
**Author:** Kimi Code subagent  
**Date:** 2026-08-07  
**Status:** Implemented + CPU smoke-tested

## 1. Goal

Provide a more flexible temporal consistency loss that can be used as an
auxiliary training term in OmniMultiViewFusion.  The existing
``motionflow_mv.losses.temporal_consistency`` already combines first-order
(velocity) and second-order (acceleration) finite differences, but it lacks:

* robustness to single-frame outliers,
* per-joint visibility/validity masking,
* per-joint weighting,
* per-frame loss outputs (``reduction="none"``).

## 2. Implementation

New file: `motionflow_mv/losses/temporal_consistency_v2.py`

### API

```python
from motionflow_mv.losses import (
    TemporalConsistencyLossV2,
    velocity_loss_v2,
    acceleration_loss_v2,
    temporal_consistency_loss_v2,
)

loss = TemporalConsistencyLossV2(
    velocity_weight=1.0,
    acceleration_weight=1.0,
    loss_type="l2",   # or "huber"
    delta=1.0,
    reduction="mean",
)
scalar = loss(pred_3d, gt_3d, mask=None, joint_weights=None)
```

### Key features

| Feature | How it works |
|---------|---------------|
| Velocity term | First-order finite difference along temporal axis. |
| Acceleration term | Second-order central finite difference. |
| Robust loss | Switch between ``"l2"`` and ``"huber"``. |
| Validity mask | ``(..., T, J)`` mask; transitions touching invalid frames are ignored. |
| Per-joint weights | ``(J,)`` weights scale each joint's contribution. |
| Reduction | ``"mean"``, ``"sum"``, or ``"none"``. |
| Short-sequence safe | Returns ``0`` when ``T < 2`` (velocity) or ``T < 3`` (acceleration). |

## 3. Smoke tests

File: `tests/test_temporal_consistency_v2.py`

Covers:

1. Zero loss for perfect predictions.
2. Zero acceleration loss for constant-velocity sequences.
3. Loss increases monotonically with additive jitter.
4. Combined loss equals the weighted sum of velocity and acceleration terms.
5. Module wrapper behaves the same as the functional API.
6. Huber loss is smaller than L2 for outlier-heavy data.
7. Masking ignores frames marked invalid.
8. Per-joint weights scale the loss.
9. ``reduction="none"`` returns a per-frame tensor.
10. Gradients flow back to predictions without NaNs.

Run:

```bash
python tests/test_temporal_consistency_v2.py
```

## 4. Integration notes

The loss is exported from `motionflow_mv.losses` alongside the original v1
counterparts.  It is intended to replace the current ``velocity_smoothness``
auxiliary term in the OmniMultiViewFusion training recipe once that model is
ready.

## 5. Next steps

* Hook ``TemporalConsistencyLossV2`` into the OmniMultiViewFusion training
  script (to be written) and tune ``velocity_weight`` / ``acceleration_weight``.
* Evaluate whether Huber threshold ``delta=1.0`` (mm² for L2 branch) is
  optimal for the MPI-INF-3DHP scale.
