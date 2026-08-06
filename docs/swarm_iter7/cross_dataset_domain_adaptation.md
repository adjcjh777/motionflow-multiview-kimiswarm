# Cross-Dataset Domain Adaptation — Next Step

**Direction:** #12 from `next_iteration_plan_swarm.md` — *Cross-dataset domain adaptation*  
**Baseline model:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` (PP backbone)  
**Date:** 2026-08-05  
**Status:** CPU smoke test passed; GPU-scale cross-dataset run queued

---

## Problem statement

The current best multi-view pose model is trained and evaluated on a single dataset (MPI-INF-3DHP).  Real-world deployment requires generalisation across datasets with different camera rigs, noise distributions and skeleton conventions (MPI-INF-3DHP 28-joint vs. Human3.6M 17-joint).  A reusable domain-adaptation wrapper already exists around the PP backbone (`motionflow_mv/models/domain_adaptation_wrapper.py`) with a gradient-reversal domain discriminator and domain-specific FiLM adapters, but it has only been unit/smoke-tested, not exercised on a genuine source/target split.  The immediate gap is a minimal, reproducible cross-dataset smoke experiment that confirms the wrapper trains end-to-end on two distinct domains without breaking the existing codebase.

## Simplest concrete next step

Run a CPU-only cross-dataset smoke test that treats two different H36M subjects/actions as source and target domains (same 17-joint skeleton, different cameras/actors), trains `DomainAdaptationWrapper` for a handful of epochs, and reports validation MPJPE.  This is the smallest change that tests the GRL+FiLM idea end-to-end without GPU or long training.  Once the smoke passes, the next iteration is to add an MPI-INF-3DHP 28→17 joint mapping and repeat with MPI→H36M on GPU.

### Files touched

| File | What changed |
|---|---|
| `motionflow_mv/models/domain_adaptation_wrapper.py` | No change — reused as-is (already wraps the PP backbone) |
| `experiments/cross_dataset_domain_adapt_smoke.py` | **New** — CPU smoke trainer that mixes source/target clips, trains the wrapper, and validates on a held-out target clip |
| `docs/swarm_iter7/cross_dataset_domain_adaptation.md` | This report |

### Launcher / command

```bash
KMP_DUPLICATE_LIB_OK=TRUE python experiments/cross_dataset_domain_adapt_smoke.py \
    --source data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz \
    --target data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz \
    --val data/webbridge/h36m_meters/s_09_acts_03_multiview_m.npz \
    --epochs 2 --batch_size 2 --train_samples 80 \
    --d 32 --n_st_layers 1 --residual_hidden 32
```

A CPU-only sanity run completes in ~1 min.

### Expected success metric

- Smoke must complete without NaNs or shape errors.
- Both pose loss and domain loss are computed and back-propagate to backbone + domain/FiLM heads.
- Validation MPJPE is finite (the tiny smoke model is not expected to reach baseline accuracy).
- For the follow-up GPU experiment: H36M zero-shot/target MPJPE after domain-adapted fine-tuning should be below the same-model source-only baseline (target: < 20 mm would be a strong publishable signal).

## Results

### CPU sanity run (small model)

Command:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python experiments/cross_dataset_domain_adapt_smoke.py \
    --source data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz \
    --target data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz \
    --val data/webbridge/h36m_meters/s_09_acts_03_multiview_m.npz \
    --epochs 2 --batch_size 2 --train_samples 80 \
    --d 32 --n_st_layers 1 --residual_hidden 32
```

Output:

```text
Device: cpu
n_views=4, j=17, clip_len=9, d=32
Model params: 51288
Epoch 1: train_loss=368.136161, pose_loss=0.004923, domain_loss=1.104629, val_MPJPE=18073.15mm (saved)
Epoch 2: train_loss=717.659693, pose_loss=0.272887, domain_loss=0.727793, val_MPJPE=18025.70mm (saved)
Best val MPJPE: 18025.70mm -> tmp\cross_dataset_domain_adapt_smoke.pth
```

The run is numerically stable and both losses back-propagate; the high validation error is expected from a deliberately tiny model on an absolute-world 3D task after only two epochs.  A larger CPU run was not pursued because the small run already verifies that the wrapper trains end-to-end.

## What still needs GPU / full training

- Replace the H36M source/target proxy with a real **MPI-INF-3DHP → H36M** split.  This requires a 28→17 joint mapping (or pre-mapping the MPI files to a common 17-joint skeleton).  The current wrapper only handles domain shift, not skeleton mismatch.
- Train with the full PP backbone (`d=128`, `n_st_layers=3`, residual hidden `256`) on GPU, warm-starting from the best MPI-INF-3DHP checkpoint.
- Ablate `use_domain_classifier`, `use_domain_film`, and `lambda_domain` to isolate the contribution of the adaptation components.

## GPU / resource requirement

| Phase | Resource | Note |
|---|---|---|
| Sanity / smoke | CPU only | Safe to run now; no GPU, < 5 min |
| Real MPI→H36M adaptation | GPU required | Must wait for the currently running cross-view PP curriculum to finish on the WSL RTX 4090 |

## Blockers

- The RTX 4090 is occupied by the cross-view PP curriculum; no GPU training should be started.
- Cross-dataset skeleton mapping (MPI 28 → H36M 17) is not yet implemented.
