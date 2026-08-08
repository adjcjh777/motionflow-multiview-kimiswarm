# v31: Domain-Balanced Sampling

## Problem statement

The WebBridge mixed loader (`configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`) currently builds a `ConcatDataset` of H36M and MPI-INF-3DHP sequences and trains with uniform shuffling. Because the manifest contains slightly more H36M train files than MPI train files, and each file contributes the same `train_samples` clips, H36M clips dominate every epoch. The learnable domain embedding tells the model *which* domain it is seeing, but it does not remove the imbalance: the model still optimises primarily for H36M statistics and can overfit to the smaller, indoor 4-view rig.

## Concrete proposed change

Add a `DomainBalancedSampler` that rebalances the training epoch so each domain contributes equally. The sampler is implemented in a new module, `motionflow_mv/data/domain_balanced_sampler.py`, and used through a thin wrapper script, `experiments/train_omniview_fusion_v5_webbridge_multi_v31_domain_balanced.py`, which monkey-patches `build_webbridge_mixed_dataloaders` before delegating to the standard v5 trainer. No existing source files are modified.

Sampler behaviour per epoch:
1. Collect global indices belonging to each domain from the underlying `ConcatDataset`.
2. Shuffle each domain independently.
3. Yield samples round-robin across domains (e.g., H36M, MPI, H36M, MPI, ...). Domains with fewer samples are resampled with replacement until the largest domain is exhausted, so every epoch has the same number of samples per domain.

The smoke run (`scripts/launch_v31_domain_balanced_sampling_local4090.sh`) enables the v30 hardened hierarchical encoder plus physical-space temporal loss, with a 2-epoch physical-loss warmup, and simply swaps in the wrapper script. TTE is left disabled.

## Expected impact

- **val_MPJPE**: We expect the largest gain on MPI validation, because the model no longer under-samples the harder, 14-view domain. H36M val_MPJPE should stay flat or improve slightly as the model learns more domain-invariant features.
- **Overfitting**: v29a showed severe overfitting after epoch 1; by preventing H36M memorisation, domain-balanced sampling can act as a mild regulariser and delay the rise in validation error.
- **View robustness**: Equal exposure to 4-view (H36M) and 14-view (MPI) clips should improve the variable-view training curriculum and stabilise the view-aggregation heads.

## Main risk

MPI-INF-3DHP has more views and noisier calibration/labels than H36M. Forcing equal sampling means the optimiser sees relatively more MPI data, which can introduce jitter if the physical-loss warmup is too short or the multi-view geometry fusion has not yet learned to handle 14-view rigs. The 2-epoch physical-loss warmup is therefore important; if the smoke run shows instability, the next step is to lower MPI's sampling weight (e.g., 0.6 H36M / 0.4 MPI) rather than use strict equality.

## Files / how to run

- `motionflow_mv/data/domain_balanced_sampler.py`: new sampler implementation.
- `experiments/train_omniview_fusion_v5_webbridge_multi_v31_domain_balanced.py`: wrapper trainer.
- `scripts/launch_v31_domain_balanced_sampling_local4090.sh`: local RTX 4090 smoke test.

```bash
bash scripts/launch_v31_domain_balanced_sampling_local4090.sh
```
