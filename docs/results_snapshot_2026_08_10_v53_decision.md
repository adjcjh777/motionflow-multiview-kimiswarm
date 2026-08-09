# Results Snapshot (2026-08-10) — v53/v54/v55 Decision

This snapshot records the v53/v54/v55 smoke/medium results and the decision on whether to keep or drop each module.

## v53 Physical-Space Calibration (PSC)

| Scale | val_MPJPE | Notes |
|---|---|---|
| Tiny smoke (A800) | 102.56 mm (epoch 1) | In line with v52 tiny 102.70 mm / 78.68 mm best |
| Tiny smoke (RTX 4090) | **78.76 mm** | Same ballpark as v52 tiny 78.68 mm |
| Medium run epoch-1 (RTX 4090) | **48.24 mm** | v52 UWT medium best = 60.09 mm → large gain |

**Decision:** Keep. Add A800 full-scale run (`v53_physical_space_calibration_on_v52`) to the active queue.

## v54 PSC-v2

| Scale | val_MPJPE | Notes |
|---|---|---|
| Tiny smoke (A800) | 103.06 mm | Worse than v52/v53 tiny |
| Tiny smoke (RTX 4090) | 100.88 mm | Worse than v52 78.68 mm |

**Decision:** Drop. Full run removed from `launch_v33_a800_queue.py`.

## v55 ORR

| Scale | val_MPJPE | Notes |
|---|---|---|
| Tiny smoke (RTX 4090) | 98.11 mm | Worse than v52 78.68 mm |
| Tiny smoke (A800) | 102.69 mm | Epoch-2 loss exploded (2255 → 1187) |

**Decision:** Drop. Unstable and not beneficial.

## v56 APL (in progress)

`motionflow_mv/fusion/adaptive_physical_loss_v56.py` implemented. Tiny smoke running on RTX 4090 to verify identity-at-init and compare with v53 tiny 78.76 mm.

## Next steps

1. Wait for v56 tiny smoke.
2. If v56 tiny is within ~1 mm of v53 tiny, run v56 medium and then A800 full.
3. Keep monitoring local v53 PSC medium run and A800 v53 tiny smoke for convergence.
