# Deprecated circular / stale-data configs

Configs in this directory reference circular-label H36M or pre-true-GT Shelf/Campus data.
They are preserved for history but should not be used for model selection or paper numbers.

## Audit status (2026-08-12)

A full search of `configs/**/*.yaml` found **no direct references** to the following
circular data paths outside this directory:

- `data/h36m_hf/`
- `data/webbridge/h36m_meters/`
- `data/webbridge/shelf_campus/`

All YAML configs containing those paths are listed under "Moved configs" below.

A separate scan found active benchmark / ablation configs that still include the
moved split manifests (indirect references). Those configs will fail to resolve
their manifests and must be updated to true-GT splits files. They are listed
under "Active references still pointing here".

## Why these are deprecated

- `data/h36m_hf/` — circular H36M labels (3D labels are triangulated from input 2D).
- `data/webbridge/h36m_meters/` — same circular H36M labels in WebBridge format.
- `data/webbridge/shelf_campus/` — pre-true-GT Shelf/Campus files.

Running experiments with these configs measures DLT reproduction, not pose accuracy.

## What to use instead

| Dataset | Deprecated path | Replacement |
|---------|------------------|-------------|
| H36M true-GT | `data/h36m_true_gt/*.npz` | `configs/splits/h36m_true_gt_standard.yaml` |
| Shelf/Campus detected | `data/webbridge/shelf_campus_detected/*.npz` | `configs/splits/shelf_campus_detected_smoke.yaml` |
| MPI-INF-3DHP | `data/webbridge/mpi_inf_3dhp/*.npz` | `configs/splits/mpiinf3dhp_detected_2d.yaml` |

## Moved configs

- `configs/deprecated/circular/train_ray_attention_reproducible.yaml`
- `configs/deprecated/circular/benchmark_webbridge_h36m_test_smoke.yaml`
- `configs/deprecated/circular/benchmark_webbridge_mpi_smoke.yaml`
- `configs/deprecated/circular/splits/webbridge_h36m_train_val.yaml`
- `configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_train_val.yaml`
- `configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_train_val_smoke.yaml`
- `configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml`
- `configs/deprecated/circular/splits/webbridge_h36m_mpi_aist_mixed_train_val_expanded.yaml`
- `configs/deprecated/circular/splits/webbridge_all_train.yaml`
- `configs/deprecated/circular/splits/webbridge_all_train_mixed.yaml`
- `configs/deprecated/circular/splits/webbridge_all_train_mixed_no_3dpw.yaml`
- `configs/deprecated/circular/splits/webbridge_proposed_mixed.yaml`
- `configs/deprecated/circular/splits/mpi_shelf_campus_noncircular_smoke.yaml`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage1_h36m_only.yaml`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage2_h36m_mpi_3_1.yaml`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage3_h36m_mpi_1_1.yaml`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage1_h36m_only_smoke.yaml`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage2_h36m_mpi_3_1_smoke.yaml`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage3_h36m_mpi_1_1_smoke.yaml`
- `configs/deprecated/circular/experiments/prototypes/swarm_iter18/P18_cross_dataset_manifest.yaml`

## Active references updated to true-GT manifests

The active configs listed below previously referenced split files that were moved
into this directory. They have now been updated to point to true-GT split
manifests. The lists are retained here for historical reference.

### `webbridge_h36m_train_val.yaml` references

- `configs/benchmark_icra_cvpr_2027.yaml`
- `configs/benchmark_v25_small.yaml`
- `configs/ablations/v25_ablation_matrix.yaml`

### `webbridge_h36m_mpi_mixed_train_val.yaml` references

- `configs/benchmark_v46_svg_smoke.yaml`
- `configs/benchmark_v47_temporal_svg_smoke.yaml`
- `configs/benchmark_v48_domain_smoke.yaml`
- `configs/benchmark_v49_lite_temporal_smoke.yaml`
- `configs/benchmark_v51_cdsvr_smoke.yaml`
- `configs/benchmark_v51_domain_agnostic_ensemble_smoke.yaml`
- `configs/benchmark_v52_scale_smoke.yaml`
- `configs/benchmark_v52_uwt_smoke.yaml`
- `configs/benchmark_v53_physical_space_calibration_smoke.yaml`
- `configs/benchmark_v54_psc_v2_smoke.yaml`
- `configs/benchmark_v55_orr_smoke.yaml`
- `configs/benchmark_v57_domain_conditional_psc_smoke.yaml`
- `configs/benchmark_v59_view_count_uwt_smoke.yaml`
- `configs/benchmark_v60_sefh_uwt_feedback_smoke.yaml`
- `configs/benchmark_v79_canonical_view_refinement_smoke.yaml`
- `configs/v31_domain_balanced_sampling.yaml`
- `configs/v31_epipolar_guided_sampling.yaml`
- `configs/v31_hierarchical_no_variable_views.yaml`
- `configs/v31_hierarchical_wider_d128.yaml`
- `configs/v31_outlier_view_adaptive_threshold.yaml`
- `configs/v31_physical_all_low_weights.yaml`
- `configs/v31_physical_collision_penalty.yaml`
- `configs/v31_physical_floor_only_warmup.yaml`
- `configs/v31_rotation_correction_quaternion.yaml`
- `configs/v31_skeleton_residual_gate.yaml`
- `configs/v45_smoke.yaml`
- `configs/ablations/example_kap_ba_sweep.yaml`
- `configs/ablations/v25_ablation_matrix.yaml`
- `configs/ablations/v25_geometry_loss_weight_ablation.yaml`
- `configs/ablations/v25_true_gt_mixed_dataset.yaml`

### `webbridge_h36m_mpi_mixed_train_val_smoke.yaml` references

- `configs/v31_uncertainty_depth_reweight.yaml`

To find these references locally:

```bash
grep -R "webbridge_h36m_train_val\|webbridge_h36m_mpi_mixed_train_val" configs/ scripts/ experiments/ docs/
```
