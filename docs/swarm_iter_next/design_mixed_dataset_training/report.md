# Mixed-Dataset Training Harness (MPI + H36M + AIST)

**Author:** research-swarm task_06  
**Status:** prototype implemented  
**Related files:**
- `motionflow_mv/data/mixed_dataset.py`
- `experiments/train_mixed_dataset.py`
- `motionflow_mv/fusion/ray_attention_temporal_mixed_residual_v1.py` (re-used, not modified)

## Problem

The project already contains two ad-hoc mixed-dataset training scripts:
`experiments/train_ray_attention_temporal_mixed_v1.py` and
`experiments/train_ray_attention_temporal_mixed_residual_v1.py`.  Both inline the
same `MixedTemporalDataset` class and hard-code dataset-specific view/joint
padding.  This makes the logic hard to reuse and error-prone when adding new
datasets.

## Goal

Provide a single, reusable mixed-dataset training harness that supports
MPI-INF-3DHP, Human3.6M, and AIST++ in one training run, while keeping the model
and dataset code cleanly separated.

## Design

### 1. `motionflow_mv/data/mixed_dataset.py`

A new data module with:

- `DATASET_REGISTRY`: central registry mapping dataset names to canonical
  view/joint counts and integer ids.
- `MixedDataset`: a `torch.utils.data.Dataset` that loads one canonical `.npz`,
  pads views and joints to the largest dataset (MPI-INF-3DHP), and tags each
  sample with a dataset id.
- `mixed_collate_fn`: standard collator for the mixed loader.
- `build_mixed_dataloaders`: helper to build a concatenated training loader and
  a validation loader from a dict of dataset-name → list of `.npz` paths.

Key invariants:
- All outputs have shape `(T, MAX_VIEWS=14, MAX_JOINTS=28, 3)` for inputs and
  `(T, MAX_JOINTS=28, 3)` for 3D targets.
- Dummy views are padded with identity intrinsics/extrinsics and zero
  translation; dummy joints are zero-padded.
- The model (`RayAttentionFusionModelTemporalMixedResidual`) is responsible for
  masking out dummy joints via its per-dataset branches.

### 2. `experiments/train_mixed_dataset.py`

A clean training script that:
- Uses the new `motionflow_mv.data.mixed_dataset` loader.
- Reuses the existing `RayAttentionFusionModelTemporalMixedResidual` model
  without modification.
- Exposes CLI arguments for MPI/H36M/AIST train files, validation file, model
  dimensions, and training hyperparameters.
- Provides a `--smoke` flag that overrides hyperparameters to tiny values for
  fast CPU validation.

### 3. Model

Unchanged.  `RayAttentionFusionModelTemporalMixedResidual` already supports the
three datasets through its `_DATASET_SPECS`:

```python
_DATASET_SPECS = {
    0: {"name": "mpi",  "n_views": 14, "n_joints": 28},
    1: {"name": "aist", "n_views":  9, "n_joints": 17},
    2: {"name": "h36m", "n_views":  4, "n_joints": 17},
}
```

## How to validate

### Unit / shape test

Run a quick import and shape sanity check:

```python
python - <<'PY'
import torch
from motionflow_mv.data.mixed_dataset import MixedDataset, build_mixed_dataloaders

# Assuming a canonical .npz exists at tmp/test_mpi.npz
ds = MixedDataset("tmp/test_mpi.npz", "mpi", clip_len=9, n_samples=2)
x, y, K, R, t, did = ds[0]
assert x.shape == (9, 14, 28, 3)
assert y.shape == (9, 28, 3)
assert did == 0
print("MixedDataset shape test passed")
PY
```

### Smoke training

```bash
python experiments/train_mixed_dataset.py \\
    --mpi_train <mpi_train.npz> \\
    --aist_train <aist_train.npz> \\
    --h36m_train <h36m_train.npz> \\
    --val <val.npz> \\
    --val_dataset mpi \\
    --smoke
```

The `--smoke` flag sets `d=8`, `n_temporal_layers=1`,
`residual_hidden=16`, `train_samples=4`, `batch_size=2`, `epochs=1`,
`clip_len=9` and should complete on CPU in seconds.

## Expected impact

- **Reusability:** New datasets can be added by extending
  `DATASET_REGISTRY` and the model's `_DATASET_SPECS`, without duplicating
  loading code.
- **Cleaner experiments:** Training scripts are reduced to model and optimizer
  configuration; data handling lives in the library.
- **Baseline for multi-dataset generalisation:** Enables systematic study of
  whether mixing MPI, H36M, and AIST improves generalisation vs. single-dataset
  training.

## Open questions / blockers

1. **Skeleton alignment:** The current harness pads different skeletons to a
   common 28-joint grid.  It does not remap anatomically equivalent joints
   across datasets (e.g. H36M hip ↔ MPI hip).  A future improvement would be a
   joint-mapping table and a unified skeleton head.
2. **AIST++ availability:** The harness expects AIST++ canonical `.npz` files
   produced by `motionflow_mv.data.webbridge_loader.convert_aistpp`.  If those
   files are not present, the script must skip AIST or generate them first.
3. **Camera units:** AIST++ raw keypoints are in centimeters; the converter
   multiplies by `0.01`.  The harness assumes the `.npz` files are already in
   meters.  If this assumption is violated, the residual head may learn
   dataset-specific bias.
4. **Validation metric across datasets:** Currently validation MPJPE is computed
   on a single validation dataset.  A more rigorous evaluation would report
   per-dataset MPJPE when the validation set contains multiple datasets.

## Next steps

- Generate canonical AIST++ `.npz` clips and run a full (non-smoke) 10-epoch
  mixed-dataset training run.
- Compare mixed training against the single-dataset baseline on MPI-INF-3DHP
  test set to measure generalisation gains.
