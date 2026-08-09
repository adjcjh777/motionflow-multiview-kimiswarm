# Results Snapshot 2026-08-09 v2

Snapshot time: 2026-08-09 (active).

## Local RTX 4090 smoke results

| Run | Epochs | Best val_MPJPE | Notes |
|-----|--------|----------------|-------|
| v46-SVG smoke | 2 | **32.97 mm** | epoch 1; epoch 2 overfit to 208.67 mm |
| v47 temporal smoke | in progress | — | step ~2100, loss ~6.23 |
| v48 domain smoke | queued | — | waiting for v47 smoke |
| v50 SEFH smoke | queued | — | will run after v46/v47/v48 chain |
| v51 DAE smoke | done | epoch 1: 35.29 mm (best) | epoch 2 overfit to 158.55 mm; fast smoke (100 samples/1 epoch) 103.64 mm |
| v52 scale v45/v46 smoke | done | epoch 1: 34.89 mm (best) | epoch 2 overfit to 148.92 mm; fast reg check (500s/1ep) 104.13 mm; A800 full run queued; smoke YAML added |
| v46-SVG medium local (fast) | 5 | — | 1k samples/seq (29k clips total), 5 epochs on RTX 4090; relaunched with nohup/watchdog (PID 32003); step 600 loss ~7.84, first validation pending |
| v51 TTSER module | — | — | `motionflow_mv/fusion/test_time_self_evolution_v51.py` implemented + 4 unit tests passing |

## A800-D full runs

| Run | GPU | Status | Latest loss | Notes |
|-----|-----|--------|-------------|-------|
| v46 sparse-view generalization on v45 | GPU6 | Running | ~4.78 at step 300 | first validation pending |
| v47 temporal aggregation on v46 | — | Queued | — | waiting for v46 |
| v48 domain generalization on v47 | — | Queued | — | waiting for v47 |
| v49-Lite temporal on v46 | — | Queued | — | after v46/v47 results |
| v49-Lite temporal on v46 scaled | — | Queued | — | 10k samples |
| v50 SEFH on v46 | — | Queued | — | after v49-Lite |
| v50 SEFH on v48 | — | Queued | — | ablation |
| v50 SEFH on v49-Lite | — | Queued | — | ablation |
| v50 SEFH low loss weight (0.001) | — | Queued | — | ablation |
| v50 SEFH aleatoric weight (0.1) | — | Queued | — | ablation |
| v51 DAE on v46 | — | Queued | — | after v50 SEFH on v46 |
| v51 DAE on v50 | — | Queued | — | after v50 SEFH on v50 |

## Design decisions

- v50 top-1 module: **Self-Evolution Feedback Head (SEFH)**.
- v51 top-1 module: **Domain-Agnostic Ensemble (DAE)** — implemented and queued for smoke/full runs.

## Data update

- New manifest `configs/splits/webbridge_h36m_mpi_aist_mixed_train_val_expanded.yaml` (1307 train / 227 val files, ~1.59M train clips) added and CPU smoke-tested.
- v52 A800 queue now uses this expanded manifest for the full-scale run.

## Next gates

1. v46-SVG medium local reaches first validation; decide if v52 smoke needs stronger regularisation.
2. v46-SVG full run on A800 reaches first validation.
3. Launch v52 scaling run on A800: v45-AGF + v46-SVG, d=128, n_st_layers=3, 10k samples, expanded manifest.
4. Revisit v51 DAE only as an ablation on the scaled v46 checkpoint if v52 shows headroom.
