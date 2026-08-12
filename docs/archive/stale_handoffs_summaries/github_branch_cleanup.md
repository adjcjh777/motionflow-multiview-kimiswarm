# Local Git Branch Cleanup Plan

- **Audit date:** 2026-08-11
- **Repository:** `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`
- **Default branch:** `main`
- **Local branches found:** 96
- **Branches on origin:** 79
- **Local-only branches:** 17
- **Exact duplicate-tip groups:** 5

## Executive Summary

This plan audits all 96 local branches and proposes a cleanup. No branches are merged into `main` except `main` itself. Most branches are months-old prototype/ablation branches that are far behind `main` (500-1000 commits) and already mirrored on `origin`. The recommended cleanup keeps only the active/default branches and a handful of local-only utility/data-foundation branches. All other branches are recommended for deletion; local-only branches should be archived (pushed to `origin`) before removal.

> **Safety note:** Do not delete any branch that is currently checked out or being used by an in-progress training/evaluation job. The two background agents (`agent-51` and `agent-67`) are operating on `main`/configs, not on any of the listed feature branches, so this cleanup does not conflict with them.

## Methodology

1. `git for-each-ref refs/heads` to enumerate local branches.
2. `git rev-list --count main..<branch>` and `git rev-list --count <branch>..main` for ahead/behind.
3. `refs/remotes/origin/<branch>` presence to detect remote backups.
4. Exact duplicate tip detection by commit SHA.

## Recommendation Categories

- **KEEP:** Default branch and active/reusable branches.
- **DELETE:** Remote-backed stale branches; can be restored from `origin`.
- **ARCHIVE then DELETE:** Local-only branches with unique commits; push to `origin` (or tag) before deleting.
- **REVIEW:** None by default, but the table can be adjusted if a branch contains unmerged work you want to rescue.

## Branch Inventory & Recommendations

| Branch | Tip | Last Commit | Remote? | Ahead | Behind | Recommendation | Notes |
|--------|-----|-------------|---------|-------|--------|----------------|-------|
| main | `3f81edf` | 2026-08-11 | Yes | 0 | 0 | KEEP | default branch |
| v33-uncertainty-aware-triangulation | `de2414c` | 2026-08-09 | Yes | 1 | 458 | KEEP | active release branch (1 ahead); rebase/merge before CVPR |
| feat/v29-self-evolving-hierarchical-multiview | `c0ef8ae` | 2026-08-08 | Yes | 1 | 542 | KEEP | active feature branch (1 ahead); rebase/merge before CVPR |
| swarm/v26_temporal_geometry_fusion_clean | `15ee78b` | 2026-08-08 | Yes | 1 | 587 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/v18_deformable_attention_baseline | `07e94e5` | 2026-08-08 | Yes | 1 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/v17_crossview_transformer_baseline | `d666774` | 2026-08-08 | Yes | 5 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/v28_physical_space_alignment_redesign | `8340a8b` | 2026-08-08 | Yes | 6 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/outlier_view_detector_improvement | `1307a9d` | 2026-08-08 | Yes | 1 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/v26_temporal_geometry_fusion_design | `a27767b` | 2026-08-08 | Yes | 4 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/smpl_prior_fusion_experiment | `07aecf3` | 2026-08-08 | Yes | 2 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/failure-analysis-v26-v27-v28 | `744131d` | 2026-08-08 | Yes | 5 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/v20_diffusion_refiner_prototype | `952ec8a` | 2026-08-08 | Yes | 6 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/v22_kap_integration | `17f70d9` | 2026-08-08 | Yes | 5 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/adaptive_view_selector_tuning | `af812b0` | 2026-08-08 | Yes | 2 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/v19_temporal_perceiver_integration | `f653e66` | 2026-08-08 | Yes | 1 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/webbridge_data_expansion | `09cc757` | 2026-08-08 | Yes | 6 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/v21_neural_ba_diagnosis | `e889351` | 2026-08-08 | Yes | 2 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| swarm/v27_uncertainty_depth_improvement | `1630d8a` | 2026-08-08 | Yes | 2 | 607 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-variable-view-subset-training | `8b1c461` | 2026-08-07 | Yes | 1 | 833 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| fix/eval-v4-geometry-toggles | `3f89646` | 2026-08-07 | Yes | 1 | 834 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-stabilize-skeleton-graph | `62d8d2a` | 2026-08-07 | Yes | 1 | 835 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-vectorize-skeleton-graph | `43d3677` | 2026-08-07 | Yes | 1 | 836 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-mixed-dataset-balanced-sampling | `805bc53` | 2026-08-07 | Yes | 6 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-confidence-aware-view-dropout | `7b3262c` | 2026-08-07 | Yes | 54 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-splatv2-view-dependent-covariance | `e4e5869` | 2026-08-07 | Yes | 53 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-visibility-uncertainty-v1 | `d07a285` | 2026-08-07 | Yes | 52 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-cross-view-graph-attention | `4672c57` | 2026-08-07 | Yes | 1 | 936 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-attention-entropy-regularization | `a48fdec` | 2026-08-07 | Yes | 51 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-temporal-velocity-acceleration-loss | `d06e5a7` | 2026-08-07 | Yes | 49 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-epipolar-bias-v2-lite | `e639129` | 2026-08-07 | Yes | 51 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-ema-checkpoint-save-load | `2e82313` | 2026-08-07 | Yes | 52 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-realtime-kd-student | `957e18e` | 2026-08-07 | Yes | 2 | 964 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-camera-conditioned-pp | `7a52274` | 2026-08-07 | Yes | 1 | 964 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-graph-joint-relation | `67b05a3` | 2026-08-07 | Yes | 50 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-crossview-contrast-ssl | `7c27db4` | 2026-08-07 | Yes | 49 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-physics-motion-prior | `adf8a90` | 2026-08-07 | Yes | 1 | 964 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-kinematic-chain-constraints | `763dfd6` | 2026-08-07 | Yes | 49 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-deeper-st-attention | `7642427` | 2026-08-07 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-bayesian-tri-v3 | `5214576` | 2026-08-07 | Yes | 49 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-extended-camera-perturbation-curriculum | `fd6bec5` | 2026-08-07 | Yes | 3 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter17-semi-supervised-pseudo-labeling | `fd6bec5` | 2026-08-07 | Yes | 3 | 965 | DELETE | exact duplicate tip of feat/iter17-extended-camera-perturbation-curriculum |
| feat/iter17-adaptive-scale-spatial-pyramid | `e4cccd7` | 2026-08-07 | Yes | 2 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-integration | `e3e347c` | 2026-08-06 | Yes | 43 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-trainer-cosine-warmup-clip-amp | `ffb5301` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-confidence-aware-view-dropout | `48dffa2` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-ema-checkpoint-save-load-support | `c4b5a69` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-prototype-deeper-st-attention | `5e3432e` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| wt-proto-14460 | `5e3432e` | 2026-08-06 | No | 5 | 965 | DELETE | exact duplicate tip of feat/iter-next-prototype-deeper-st-attention |
| feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability-wt | `baa26b3` | 2026-08-06 | No | 5 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| feat/iter-next-hp-search-large | `13ad9dc` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-learned-per-joint-precision-and-refinement | `03e73a4` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-extend-robustness-matrix | `2805401` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-cross-view-graph-attention | `0381188` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-ablation-csv-plotting | `a2cbbf7` | 2026-08-06 | Yes | 4 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-bayesian-tri-v2-batched-dlt-tests | `5af961f` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-synthetic-joint-occlusion | `078c958` | 2026-08-06 | Yes | 7 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-synthesize-swarm-outputs | `5f4fc47` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-temporal-velocity-acceleration-consistency-loss | `66e5032` | 2026-08-06 | Yes | 6 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-synchronized-multiview-2d-augmentation | `9e907b9` | 2026-08-06 | Yes | 6 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-draft-icra-cvpr-paper-story | `91c1d39` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-ensemble-inference-multi-checkpoint | `deec223` | 2026-08-06 | Yes | 4 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-update-gh-issue-25 | `1701e31` | 2026-08-06 | Yes | 5 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability | `ef1f6ac` | 2026-08-06 | Yes | 4 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/iter-next-extend-camera-perturbation-ranges-and-intrinsics-curriculum | `ef1f6ac` | 2026-08-06 | Yes | 4 | 965 | DELETE | exact duplicate tip of feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability |
| feat/iter-next-roadmap | `ef1f6ac` | 2026-08-06 | Yes | 4 | 965 | DELETE | exact duplicate tip of feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability |
| feat/kinematic-chain-constraints-aux | `19a2606` | 2026-08-06 | Yes | 2 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feature/data-augmentation-multiview-vstj | `2c66887` | 2026-08-06 | Yes | 3 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| my-attention-entropy | `76c63e5` | 2026-08-06 | Yes | 3 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| fix/h36m-corrected-track | `344e14c` | 2026-08-06 | No | 2 | 965 | KEEP | data-foundation fix aligned with current true-GT work (local-only) |
| feat/unified-results-csv | `e479d0e` | 2026-08-06 | No | 2 | 965 | KEEP | eval utility (results.csv logger); local-only, cherry-pick/rebase |
| feat/set-transformer-crossview | `4164fd8` | 2026-08-06 | No | 3 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| feat/fast-epipolar-bias-v2-pp | `4fe4d05` | 2026-08-06 | No | 3 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| feat/temporal-skeleton-consistency-loss | `c0c548a` | 2026-08-06 | No | 4 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| feat/multiscale-temporal-residual | `273b9b5` | 2026-08-06 | No | 2 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| run/visibility-uncertainty-v1 | `935a6a7` | 2026-08-06 | No | 5 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| feat/adaptive-scale-spatial-pyramid-fusion | `81f684e` | 2026-08-06 | Yes | 3 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/bayesian-tri-v2-batched-lstsq | `0451772` | 2026-08-06 | Yes | 5 | 965 | KEEP | Bayesian tri v2 eval support; local-only, cherry-pick/rebase |
| realtime-kd-student-iter16 | `01ecc24` | 2026-08-06 | Yes | 2 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feature/epipolar-bias-v2-lite | `3367540` | 2026-08-06 | Yes | 4 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| domain_adaptation_shelf_campus_v2 | `8fbb702` | 2026-08-06 | Yes | 1 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/crossview-visibility-uncertainty-v1 | `4115a36` | 2026-08-06 | Yes | 1 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feat/temporal-ray-attention-deeper | `a4499f6` | 2026-08-06 | Yes | 2 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feature/webbridge-mixed-17joint-v3 | `9418edc` | 2026-08-06 | No | 2 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| clean-data-aug | `3d75dba` | 2026-08-06 | No | 3 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| feature/splatv2-view-dependent-covariance-clean2 | `3d75dba` | 2026-08-06 | No | 3 | 965 | DELETE | exact duplicate tip of clean-data-aug |
| attention-entropy-interpretability | `d9922bc` | 2026-08-06 | No | 2 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| feature/splatv2-view-dependent-covariance-clean | `988a63e` | 2026-08-06 | No | 1 | 965 | DELETE | exact duplicate tip of feature/splatv2-view-dependent-covariance-final |
| feature/splatv2-view-dependent-covariance-final | `988a63e` | 2026-08-06 | Yes | 1 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feature/graph-joint-relation-full-run | `629e111` | 2026-08-06 | No | 2 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| feature/splatv2-view-dependent-covariance | `b9a65c2` | 2026-08-06 | No | 3 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| feature/webbridge-mixed-17joint | `7e7cb95` | 2026-08-06 | No | 2 | 965 | ARCHIVE then DELETE | local-only; push to origin/<branch> or tag before deleting |
| feat/semi-supervised-pseudo-labeling | `07ab4de` | 2026-08-06 | Yes | 1 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feature/robustness-matrix-multi-model | `ffcd66a` | 2026-08-06 | Yes | 1 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| camera-conditioned-pp-module-registration | `843e271` | 2026-08-06 | Yes | 2 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| ssl-view-contrast | `dba4621` | 2026-08-06 | Yes | 1 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |
| feature/mixed-dataset-balanced-sampling | `2096202` | 2026-08-06 | Yes | 1 | 965 | DELETE | stale remote-backed branch; restore from origin if ever needed |

## Duplicate Tip Groups

These local branches point to the exact same commit. Keep the canonical name (chosen below) and delete the duplicates.

- **fd6bec5** canonical: `feat/iter17-extended-camera-perturbation-curriculum`; duplicates: `feat/iter17-semi-supervised-pseudo-labeling`
- **5e3432e** canonical: `feat/iter-next-prototype-deeper-st-attention`; duplicates: `wt-proto-14460`
- **ef1f6ac** canonical: `feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability`; duplicates: `feat/iter-next-extend-camera-perturbation-ranges-and-intrinsics-curriculum`, `feat/iter-next-roadmap`
- **3d75dba** canonical: `clean-data-aug`; duplicates: `feature/splatv2-view-dependent-covariance-clean2`
- **988a63e** canonical: `feature/splatv2-view-dependent-covariance-final`; duplicates: `feature/splatv2-view-dependent-covariance-clean`

## Proposed Execution Commands

Run these from the repo root **after reviewing** the table above. Replace `<branch>` with the actual names.

### 1. Archive local-only branches before deletion
```bash
git push origin feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability-wt:feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability-wt  # archive feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability-wt
git push origin feat/set-transformer-crossview:feat/set-transformer-crossview  # archive feat/set-transformer-crossview
git push origin feat/fast-epipolar-bias-v2-pp:feat/fast-epipolar-bias-v2-pp  # archive feat/fast-epipolar-bias-v2-pp
git push origin feat/temporal-skeleton-consistency-loss:feat/temporal-skeleton-consistency-loss  # archive feat/temporal-skeleton-consistency-loss
git push origin feat/multiscale-temporal-residual:feat/multiscale-temporal-residual  # archive feat/multiscale-temporal-residual
git push origin run/visibility-uncertainty-v1:run/visibility-uncertainty-v1  # archive run/visibility-uncertainty-v1
git push origin feature/webbridge-mixed-17joint-v3:feature/webbridge-mixed-17joint-v3  # archive feature/webbridge-mixed-17joint-v3
git push origin clean-data-aug:clean-data-aug  # archive clean-data-aug
git push origin attention-entropy-interpretability:attention-entropy-interpretability  # archive attention-entropy-interpretability
git push origin feature/graph-joint-relation-full-run:feature/graph-joint-relation-full-run  # archive feature/graph-joint-relation-full-run
git push origin feature/splatv2-view-dependent-covariance:feature/splatv2-view-dependent-covariance  # archive feature/splatv2-view-dependent-covariance
git push origin feature/webbridge-mixed-17joint:feature/webbridge-mixed-17joint  # archive feature/webbridge-mixed-17joint
```

### 2. Delete stale remote-backed and local-only branches
```bash
git branch -D swarm/v26_temporal_geometry_fusion_clean
git branch -D swarm/v18_deformable_attention_baseline
git branch -D swarm/v17_crossview_transformer_baseline
git branch -D swarm/v28_physical_space_alignment_redesign
git branch -D swarm/outlier_view_detector_improvement
git branch -D swarm/v26_temporal_geometry_fusion_design
git branch -D swarm/smpl_prior_fusion_experiment
git branch -D swarm/failure-analysis-v26-v27-v28
git branch -D swarm/v20_diffusion_refiner_prototype
git branch -D swarm/v22_kap_integration
git branch -D swarm/adaptive_view_selector_tuning
git branch -D swarm/v19_temporal_perceiver_integration
git branch -D swarm/webbridge_data_expansion
git branch -D swarm/v21_neural_ba_diagnosis
git branch -D swarm/v27_uncertainty_depth_improvement
git branch -D feat/iter-next-variable-view-subset-training
git branch -D fix/eval-v4-geometry-toggles
git branch -D feat/iter-next-stabilize-skeleton-graph
git branch -D feat/iter-next-vectorize-skeleton-graph
git branch -D feat/iter17-mixed-dataset-balanced-sampling
git branch -D feat/iter17-confidence-aware-view-dropout
git branch -D feat/iter17-splatv2-view-dependent-covariance
git branch -D feat/iter17-visibility-uncertainty-v1
git branch -D feat/iter17-cross-view-graph-attention
git branch -D feat/iter17-attention-entropy-regularization
git branch -D feat/iter17-temporal-velocity-acceleration-loss
git branch -D feat/iter17-epipolar-bias-v2-lite
git branch -D feat/iter17-ema-checkpoint-save-load
git branch -D feat/iter17-realtime-kd-student
git branch -D feat/iter17-camera-conditioned-pp
git branch -D feat/iter17-graph-joint-relation
git branch -D feat/iter17-crossview-contrast-ssl
git branch -D feat/iter17-physics-motion-prior
git branch -D feat/iter17-kinematic-chain-constraints
git branch -D feat/iter17-deeper-st-attention
git branch -D feat/iter17-bayesian-tri-v3
git branch -D feat/iter17-extended-camera-perturbation-curriculum
git branch -D feat/iter17-semi-supervised-pseudo-labeling
git branch -D feat/iter17-adaptive-scale-spatial-pyramid
git branch -D feat/iter-next-integration
git branch -D feat/iter-next-trainer-cosine-warmup-clip-amp
git branch -D feat/iter-next-confidence-aware-view-dropout
git branch -D feat/iter-next-ema-checkpoint-save-load-support
git branch -D feat/iter-next-prototype-deeper-st-attention
git branch -D wt-proto-14460
git branch -D feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability-wt
git branch -D feat/iter-next-hp-search-large
git branch -D feat/iter-next-learned-per-joint-precision-and-refinement
git branch -D feat/iter-next-extend-robustness-matrix
git branch -D feat/iter-next-cross-view-graph-attention
git branch -D feat/iter-next-ablation-csv-plotting
git branch -D feat/iter-next-bayesian-tri-v2-batched-dlt-tests
git branch -D feat/iter-next-synthetic-joint-occlusion
git branch -D feat/iter-next-synthesize-swarm-outputs
git branch -D feat/iter-next-temporal-velocity-acceleration-consistency-loss
git branch -D feat/iter-next-synchronized-multiview-2d-augmentation
git branch -D feat/iter-next-draft-icra-cvpr-paper-story
git branch -D feat/iter-next-ensemble-inference-multi-checkpoint
git branch -D feat/iter-next-update-gh-issue-25
git branch -D feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability
git branch -D feat/iter-next-extend-camera-perturbation-ranges-and-intrinsics-curriculum
git branch -D feat/iter-next-roadmap
git branch -D feat/kinematic-chain-constraints-aux
git branch -D feature/data-augmentation-multiview-vstj
git branch -D my-attention-entropy
git branch -D feat/set-transformer-crossview
git branch -D feat/fast-epipolar-bias-v2-pp
git branch -D feat/temporal-skeleton-consistency-loss
git branch -D feat/multiscale-temporal-residual
git branch -D run/visibility-uncertainty-v1
git branch -D feat/adaptive-scale-spatial-pyramid-fusion
git branch -D realtime-kd-student-iter16
git branch -D feature/epipolar-bias-v2-lite
git branch -D domain_adaptation_shelf_campus_v2
git branch -D feat/crossview-visibility-uncertainty-v1
git branch -D feat/temporal-ray-attention-deeper
git branch -D feature/webbridge-mixed-17joint-v3
git branch -D clean-data-aug
git branch -D feature/splatv2-view-dependent-covariance-clean2
git branch -D attention-entropy-interpretability
git branch -D feature/splatv2-view-dependent-covariance-clean
git branch -D feature/splatv2-view-dependent-covariance-final
git branch -D feature/graph-joint-relation-full-run
git branch -D feature/splatv2-view-dependent-covariance
git branch -D feature/webbridge-mixed-17joint
git branch -D feat/semi-supervised-pseudo-labeling
git branch -D feature/robustness-matrix-multi-model
git branch -D camera-conditioned-pp-module-registration
git branch -D ssl-view-contrast
git branch -D feature/mixed-dataset-balanced-sampling
```

### 3. Optional: prune stale remote-tracking branches
```bash
git fetch --prune
```

## Risk Notes

- `v33-uncertainty-aware-triangulation` and `feat/v29-self-evolving-hierarchical-multiview` are the only active feature branches; both are one commit ahead of an older `main` and will need a rebase before merge.
- `fix/h36m-corrected-track`, `feat/unified-results-csv`, and `feat/bayesian-tri-v2-batched-lstsq` are local-only but contain code relevant to the current data-foundation and evaluation effort; they are kept in this plan. Consider rebasing or cherry-picking them onto current `main`.
- Deleting the `swarm/*`, `feat/iter17-*`, and `feat/iter-next-*` branches is low-risk because the vast majority are mirrored on `origin`.
- No new GPU jobs were started for this audit; it is purely a documentation/local-branch review task.
