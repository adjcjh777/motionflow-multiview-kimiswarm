# Deprecated circular / stale-data configs

Configs in this directory reference circular-label H36M or pre-true-GT Shelf/Campus data.
They are preserved for history but should not be used for model selection or paper numbers.

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

## Active references still pointing here (will fail loudly)

Several active benchmark / ablation configs still reference the moved split files.
Those configs will now fail to resolve their manifests and must be updated to true-GT splits.
Examples:

- `configs/benchmark_icra_cvpr_2027.yaml` → `configs/splits/webbridge_h36m_train_val.yaml` (moved)
- `configs/benchmark_v25_small.yaml` → `configs/splits/webbridge_h36m_train_val.yaml` (moved)
- `configs/benchmark_v25_smoke.yaml`, `configs/benchmark_v46_svg_smoke.yaml`, etc. → `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml` (moved)

Search for the moved filenames to find all references:

```bash
grep -R "webbridge_h36m_train_val.yaml\|webbridge_h36m_mpi_mixed_train_val" configs/ scripts/ experiments/ docs/
```
