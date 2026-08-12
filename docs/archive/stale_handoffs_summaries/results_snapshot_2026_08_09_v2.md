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
| v46-SVG medium local (fast) | 5 | — | 1k samples/seq (29k clips total), 5 epochs on RTX 4090; relaunched as background task; step 650 loss ~7.67, first validation pending |
| v51 TTSER module | — | — | Wired into `OmniMultiViewFusionV5`; pose refinement enabled; eval script + tests passing |

## A800-D full runs

| Run | GPU | Status | Best/Current val_MPJPE | Notes |
|-----|-----|--------|------------------------|-------|
| v46 sparse-view generalization on v45 | GPU6 | **Done** | **64.00 mm** (epoch 4, early stopped at epoch 5) | Used 200 samples; under-trained for d=128 |
| v47 temporal aggregation on v46 | GPU5 | Running | 63.37 mm at epoch 4 | 5 epochs, 200 samples |
| v48 domain generalization on v47 | — | Ready/Queued | — | 200 samples -> bumped to 1000 samples/10 epochs after v46 result |
| v49-Lite temporal on v46 | — | Ready/Queued | — | 200 samples -> bumped to 1000 samples/10 epochs after v46 result |
| v49-Lite temporal on v46 scaled | — | Queued | — | 10k samples |
| v50 SEFH on v46 | — | Queued | — | 10k samples |
| v50 SEFH on v48 | — | Queued | — | ablation |
| v50 SEFH on v49-Lite | — | Queued | — | ablation |
| v50 SEFH low loss weight (0.001) | — | Queued | — | ablation |
| v50 SEFH aleatoric weight (0.1) | GPU6 | Running | — | currently running |
| v51 DAE on v46 | — | Queued | — | after v50 SEFH on v46 |
| v51 DAE on v50 | — | Queued | — | after v50 SEFH on v50 |
| v51 TTSER | — | Wired | — | inference-only pose/reliability/uncertainty refinement |

## Recent code changes

- Fixed `configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml` Windows backslash paths for A800 Linux.
- Added safety path normalisation in `motionflow_mv/data/webbridge_mixed_dataset.py`.
- Updated `scripts/launch_v33_a800_queue.py` to skip runs whose `.pth` already exists, avoiding duplicate v46/v47 re-runs.
- Added `experiments/eval_v51_test_time_self_evolution.py` and test.
- Enabled `refine_pose` in v51 TTSER from `OmniMultiViewFusionV5`.

## Design decisions

- v50 top-1 module: **Self-Evolution Feedback Head (SEFH)**.
- v51 top-1 module: **Domain-Agnostic Ensemble (DAE)** — implemented and queued for smoke/full runs.
- v51 TTSER now closes the self-evolution loop by refining the 3-D pose at test time, not only reliability/uncertainty.

## Data update

- New manifest `configs/splits/webbridge_h36m_mpi_aist_mixed_train_val_expanded.yaml` (1307 train / 227 val files, ~1.59M train clips) added and CPU smoke-tested.
- v52 A800 queue now uses this expanded manifest for the full-scale run.

## Next gates

1. v46-SVG medium local reaches first validation; decide if v52 smoke needs stronger regularisation.
2. v47 A800 finishes epoch 5 and reports final val_MPJPE.
3. A800 GPU frees -> poller launches v48/v49/v50/v51/v52 with updated sample counts.
4. Measure v51 TTSER impact on v46 A800 checkpoint using `experiments/eval_v51_test_time_self_evolution.py`.
