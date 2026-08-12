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

## v56 APL

| Scale | val_MPJPE | Notes |
|---|---|---|
| Tiny smoke (RTX 4090) | **104.30 mm** | Worse than v53 78.76 mm; identity-at-init was fixed (exp weighting) |

**Decision:** Drop. Adaptive per-sample PSC loss weighting does not help at tiny scale and may destabilise training.

## Summary

- **Keep v53 PSC** and run A800 full-scale (`v53_physical_space_calibration_on_v52`).
- **Drop v54, v55, v56** — none improved over v52/v53 tiny smoke.
- The next iteration should not add another loss-weighting module on v53; instead, focus on scaling v53 and on data/regularisation (e.g. expanded manifest, larger model, longer training).

## Next steps

1. Monitor local v53 PSC medium run to completion.
2. Monitor A800 v53 tiny smoke to completion.
3. Launch A800 full v53 run when GPU memory frees.
4. Consider v57 only after v53 full results are in.
