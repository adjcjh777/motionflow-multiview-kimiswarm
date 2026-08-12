# Variable-View Model Comparison (v25 / v81 / v82)

## Data source

- `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json`
- `outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_k23.json`
- `outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.json`

All numbers are H36M true-GT test MPJPE (mm) for variable-view evaluation on S9/S11.

## Results

| Model | k | S9 MPJPE | S11 MPJPE | Combined* | Temporal jerk (S9/S11) | Notes |
|-------|---|----------|-----------|-----------|------------------------|-------|
| v25 stability | 2 | 58.18 | 49.35 | 53.76 | 71.20 / 60.50 | DLT fallback (k < 4) |
| v25 stability | 3 | 33.32 | 25.28 | 29.30 | 36.09 / 29.52 | DLT fallback (k < 4) |
| v25 stability | 4 | 116.98 | 110.58 | 113.78 | 33.66 / 27.88 | Learned model, catastrophic |
| v81 temporal-pose-attn | 2 | 58.18 | 49.35 | 53.76 | 71.20 / 60.50 | DLT fallback only |
| v81 temporal-pose-attn | 3 | 33.32 | 25.28 | 29.30 | 36.09 / 29.52 | DLT fallback only |
| v82 multi-scale temporal-pose-attn | 2 | 58.18 | 49.35 | 53.76 | 71.20 / 60.50 | DLT fallback (k < 4) |
| v82 multi-scale temporal-pose-attn | 3 | 33.32 | 25.28 | 29.30 | 36.09 / 29.52 | DLT fallback (k < 4) |
| v82 multi-scale temporal-pose-attn | 4 | 47.81 | 42.36 | 45.09 | 33.48 / 27.02 | Learned model, strong |

*Combined = unweighted average of S9 and S11.

## Key takeaways

1. **Sparse-view (k = 2, 3) performance is identical across v25, v81, and v82.** For these evaluations the learned model is bypassed and a confidence-weighted DLT fallback is used. This confirms the underlying 2D observations are sound: direct triangulation already gives ~54 mm at k=2 and ~29 mm at k=3.
2. **v25's full-view learned model fails catastrophically at k = 4.** Its k=4 MPJPE (113.8 mm) is far worse than its k=3 DLT-fallback result (29.3 mm) and worse than the v82 full-view learned model (45.1 mm).
3. **v82's learned full-view model is substantially better than v25's.** v82 k=4 improves over v25 k=4 by roughly 69 mm (S9: 116.98 → 47.81; S11: 110.58 → 42.36). Temporal jerk is similar, so the gain is not coming at the cost of increased jitter.
4. **v81 only covers k = 2, 3.** It uses the same DLT fallback and is therefore directly comparable to v25/v82 for sparse views, but it provides no learned full-view data point.

## Implications for v85

- v85 random view dropout is the first model trained natively on k = 2/3/4 input. The v25/v81/v82 DLT-fallback baseline gives a clear target: ~54 mm at k=2 and ~29 mm at k=3.
- The large gap between v82 (45 mm) and v25 (113 mm) at k=4 shows that full-view learned-model quality is very sensitive to architectural differences. v85 needs to beat or match v82 at k=4 while also handling k=2/3 without DLT fallback.
