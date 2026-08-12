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

A separate scan found active benchmark / ablation configs and scripts that still
referenced the moved split manifests or circular `.txt` manifests. They have been
updated to true-GT split manifests and are listed under "Active references fixed"
below. The historical lists under "Active references updated to true-GT manifests"
are retained for reference.

## Why these are deprecated

- `data/h36m_hf/` — circular H36M labels (3D labels are triangulated from input 2D).
- `data/webbridge/h36m_meters/` — same circular H36M labels in WebBridge format.
- `data/webbridge/shelf_campus/` — pre-true-GT Shelf/Campus files.

Running experiments with these configs measures DLT reproduction, not pose accuracy.

## What to use instead

| Dataset | Deprecated path | Replacement |
|---------|------------------|-------------|
| H36M true-GT | `data/h36m_true_gt/*.npz` | `configs/splits/h36m_true_gt_standard.yaml` |
| H36M true-GT v2 | `data/h36m_true_gt_v2/*.npz` | `configs/splits/h36m_true_gt_v2_standard.yaml` |
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
- `configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_val_for_eval.txt`
- `configs/deprecated/circular/splits/v51_dae_eval_single.txt`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage1_h36m_only.yaml`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage2_h36m_mpi_3_1.yaml`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage3_h36m_mpi_1_1.yaml`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage1_h36m_only_smoke.yaml`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage2_h36m_mpi_3_1_smoke.yaml`
- `configs/deprecated/circular/v31_mixed_dataset_curriculum/stage3_h36m_mpi_1_1_smoke.yaml`
- `configs/deprecated/circular/experiments/prototypes/swarm_iter18/P18_cross_dataset_manifest.yaml`

## Active references fixed during this audit

| File | Old reference | Updated reference |
|------|---------------|-------------------|
| `scripts/run_omniview_fusion_v4_a800.sh` | `configs/deprecated/circular/splits/webbridge_h36m_train_val.yaml` + non-existent `configs/splits/webbridge_mpi_train_val.yaml` | `configs/splits/h36m_true_gt_standard.yaml` + `configs/splits/mpiinf3dhp_train_val_test.yaml` |
| `scripts/run_v51_dae_eval_local_4090.sh` | `configs/splits/webbridge_h36m_mpi_mixed_val_for_eval.txt` (pointed to `data/webbridge/h36m_meters/`) | same path; file content updated to `data/h36m_true_gt_v2/` + `data/webbridge/mpi_inf_3dhp/` |
| `configs/splits/webbridge_h36m_mpi_mixed_val_for_eval.txt` | `data/webbridge/h36m_meters/` | `data/h36m_true_gt_v2/` |
| `configs/splits/v51_dae_eval_single.txt` | `data/webbridge/h36m_meters/` | `data/h36m_true_gt_v2/` |
| `scripts/tmux_v6_h36m_isab.sh` | `configs/deprecated/circular/splits/webbridge_h36m_train_val.yaml` | `configs/splits/h36m_true_gt_standard.yaml` |
| `experiments/train_omniview_fusion_v2_webbridge_multi.py` | docstring: `configs/deprecated/circular/splits/webbridge_h36m_train_val.yaml` | `configs/splits/h36m_true_gt_standard.yaml` |
| `experiments/train_omniview_fusion_v4_webbridge_multi.py` | docstring: `configs/deprecated/circular/splits/webbridge_h36m_train_val.yaml` | `configs/splits/h36m_true_gt_standard.yaml` |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | docstring: `configs/deprecated/circular/splits/webbridge_h36m_train_val.yaml` | `configs/splits/h36m_true_gt_standard.yaml` |

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

## Known remaining circular data-path references

The broader codebase still contains references to the old circular data paths
(`data/h36m_hf/`, `data/webbridge/h36m_meters/`, `data/webbridge/shelf_campus/`)
in scripts and experiments. Most of these are intentional:

- **Diagnostic / converter scripts** such as `scripts/diagnose_circular_labels.py`
  intentionally reference `data/h36m_hf/` to demonstrate the circular-label issue.
- **Legacy training / eval scripts** are documented in
  `docs/circular_label_references_audit.md` and are tracked separately for
  migration or deprecation.
- **Archive docs** under `docs/archive/` and `docs/swarm_iter*/` retain historical
  references and are not used for current model selection.

No new code should reference the moved **split manifests** listed in
"Moved configs" above. To verify:

```bash
grep -R -e "configs/splits/webbridge_h36m_train_val" \
          -e "configs/splits/webbridge_h36m_mpi_mixed_train_val" \
          -e "configs/splits/webbridge_all_train" \
          -e "configs/splits/webbridge_proposed_mixed" \
          -e "configs/splits/mpi_shelf_campus_noncircular" \
          scripts/ experiments/
```

(References inside `configs/deprecated/circular/` are expected.)
