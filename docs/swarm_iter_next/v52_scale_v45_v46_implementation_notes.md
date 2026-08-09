# v52 — Scale v45-AGF + v46-SVG

**Tracking issue:** #179

## Goal

Stop stacking new modules and instead scale the strongest proven stack: v45 adaptive geometry fusion + v46 sparse-view generalization. The hypothesis is that the v46 baseline (32.97 mm smoke val_MPJPE) has more headroom from capacity and data than from another gating/ensemble module.

## Local smoke (RTX 4090)

`scripts/run_v52_scale_v45_v46_smoke_local_4090.sh`

| Setting | Value |
|---------|-------|
| d | 64 |
| n_st_layers | 2 |
| batch_size | 8 |
| clip_len | 9 |
| train_samples | 1000 |
| epochs | 2 |
| weight_decay | 1e-4 |
| v25_dropout | 0.2 |
| v30_stochastic_depth_prob | 0.1 |

### Results

| Epoch | val_MPJPE |
|-------|-----------|
| 1 | **34.89 mm** |
| 2 | 148.92 mm (overfit) |

The local smoke overfit, suggesting the original regularization was too weak for the scaled capacity and 1000 samples.

## A800 full run

`scripts/launch_v33_a800_queue.py` entry `v52_scale_v45_v46`

Updated after the local overfit to use stronger regularization:

| Setting | Value |
|---------|-------|
| d | 128 |
| n_st_layers | 3 |
| batch_size | 16 |
| clip_len | 13 |
| train_samples | 10000 |
| epochs | 10 |
| weight_decay | **5e-4** |
| v25_dropout | **0.3** |
| v30_stochastic_depth_prob | **0.2** |

## Acceptance

- val_MPJPE < 30 mm
- MPJPE@2 improves over v46 smoke baseline
- No epoch-1/epoch-2 blow-up

## Risks

| Risk | Mitigation |
|------|------------|
| Overfitting at d=128 | stronger weight decay, dropout, stochastic depth |
| Long training time | early stopping patience 3 |
| Queue starvation | v52 is first in A800 queue |
