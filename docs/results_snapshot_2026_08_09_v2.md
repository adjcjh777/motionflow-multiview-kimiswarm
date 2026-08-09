# Results Snapshot 2026-08-09 v2

Snapshot time: 2026-08-09 (active).

## Local RTX 4090 smoke results

| Run | Epochs | Best val_MPJPE | Notes |
|-----|--------|----------------|-------|
| v46-SVG smoke | 2 | **32.97 mm** | epoch 1; epoch 2 overfit to 208.67 mm |
| v47 temporal smoke | in progress | — | step ~1750, loss ~6.39 |
| v48 domain smoke | queued | — | waiting for v47 smoke |
| v50 SEFH smoke | queued | — | will run after v46/v47/v48 chain |

## A800-D full runs

| Run | GPU | Status | Latest loss | Notes |
|-----|-----|--------|-------------|-------|
| v46 sparse-view generalization on v45 | GPU6 | Running | ~14.7 at step 350 | first validation pending |
| v47 temporal aggregation on v46 | — | Queued | — | waiting for v46 |
| v48 domain generalization on v47 | — | Queued | — | waiting for v47 |
| v49-Lite temporal on v46 | — | Queued | — | after v46/v47 results |
| v49-Lite temporal on v46 scaled | — | Queued | — | 10k samples |
| v50 SEFH on v46 | — | Queued | — | after v49-Lite |
| v50 SEFH on v48 | — | Queued | — | ablation |
| v50 SEFH on v49-Lite | — | Queued | — | ablation |
| v50 SEFH low loss weight (0.001) | — | Queued | — | ablation |
| v50 SEFH aleatoric weight (0.1) | — | Queued | — | ablation |

## Design decisions

- v50 top-1 module: **Self-Evolution Feedback Head (SEFH)**.
- v51 will be decided after v50 smoke and A800 full results.

## Next gates

1. v46-SVG full run on A800 reaches first validation.
2. Local v47/v48 smoke chain finishes, triggering v49 ablation + v50 smoke.
3. v50 smoke shows `val_MPJPE@full` within 1 mm of v46 baseline and `MPJPE@2` improvement ≥2 mm.
