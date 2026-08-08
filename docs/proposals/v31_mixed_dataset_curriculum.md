# v31 Ablation: Mixed-Dataset Curriculum

## Problem statement

v29/v30 trains on a fixed 50/50 mix of Human3.6M (4 views, studio cameras) and MPI-INF-3DHP (14 views, diverse outdoor/indoor rigs) from epoch 1. The two domains differ in camera count, noise level, skeleton conventions after canonicalisation, and motion style. Throwing MPI into the full model immediately forces the geometry-fusion and hierarchical-view blocks to learn two very different multi-view statistics at once. This matches the v29a observation: strong early-epoch validation MPJPE followed by rapid overfitting, because the model memorises easy H36M cues and never stabilises on the harder MPI domain. A dataset-level curriculum—starting with the simpler domain and gradually increasing the harder one—should let the model build stable geometric priors before it has to handle MPI's variability.

## Concrete proposed change

Replace the single mixed-dataset run with a three-stage warm-started pipeline. Each stage uses the same v30 architecture (`use_hierarchical_multiview_v30`, stochastic depth, gated cross-scale fusion, no TTE) and the same physical-space temporal loss with warmup, but varies the training mix through separate WebBridge manifests files.

1. **Stage 1** (`configs/v31_mixed_dataset_curriculum/stage1_h36m_only.yaml`) trains on H36M only for ~10 epochs (2 epochs in the local smoke). This lets the model learn a clean 17-joint, 4-view baseline.
2. **Stage 2** (`configs/v31_mixed_dataset_curriculum/stage2_h36m_mpi_3_1.yaml`) warm-starts from the Stage 1 checkpoint and trains on a 3:1 H36M:MPI mix (by file count) for another ~10 epochs. MPI is introduced gradually without destabilising the already learned H36M representation.
3. **Stage 3** (`configs/v31_mixed_dataset_curriculum/stage3_h36m_mpi_1_1.yaml`) warm-starts from Stage 2 and trains on the full 1:1 H36M:MPI mix to fine-tune the shared domain-agnostic pose prior.

The launch scripts `scripts/launch_v31_mixed_dataset_curriculum_local4090.sh` (smoke) and `scripts/launch_v31_mixed_dataset_curriculum_a800.sh` (full run) chain the three stages via `--warm_start`. Validation always reports both H36M and MPI MPJPE, so we can measure domain-specific transfer. No existing source files are modified; only new manifest/config and launch files are added.

## Expected impact on val_MPJPE / overfitting

- **Earlier stability**: Stage 1 should reach a low H36M val_MPJPE quickly without MPI noise interfering with geometry-fusion weight initialisation.
- **Reduced v29a-style overfitting**: By the time MPI appears in Stage 2, the model has a stable feature backbone, so the extra domain is more likely to be absorbed rather than cause the validation curve to diverge after epoch 1.
- **Better MPI generalisation**: Gradual exposure should improve MPI val_MPJPE compared with a cold start on the 1:1 mix, because the domain embedding for MPI is initialised in a model that already understands multi-view geometry.
- **Trade-off**: Total wall-clock time triples because the run is staged. The curriculum is only worthwhile if the final Stage 3 MPJPE improves over the baseline or overfitting is meaningfully delayed.

## Main risk

The biggest risk is **warm-start fragility**: MPI introduces domain ID 1, whose embedding is random at the start of Stage 2. If the MPI geometry is too different from the H36M representation learned in Stage 1, Stage 2 can still diverge or overfit. A mitigating factor is that the v30 encoder is identity-at-init and the domain embedding is small, but the curriculum still needs to be validated empirically. If Stage 2 MPI val_MPJPE does not improve over the baseline, the staged warm-start is not worth the extra compute.
