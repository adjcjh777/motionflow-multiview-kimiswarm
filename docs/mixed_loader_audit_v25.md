# Mixed WebBridge / H36M / MPI Loader Audit (v25)

**Scope:** `motionflow_mv/data/webbridge_mixed_dataset.py`, `configs/splits/*.yaml`

**Goal:** Identify loader-level blockers for the v25 iteration that adds geometry-aware cross-view attention and learned depth triangulation, and provide a minimal, testable fix.

## Key finding

The v25 `MultiViewGeometryFusionV25` module accepts an optional `view_mask: (B, T, V)` argument so that padded/invalid views are ignored in cross-view attention and depth-proposal triangulation. The current `WebBridgeCanonical17Dataset` / `WebBridgeMixedDataset` loaders pad every sequence to `MAX_VIEWS = 14` but do **not** expose which views are real. This can leak zero-padded placeholder views into geometry computations.

## Changes made

1. **Backward-compatible view-mask support** in `motionflow_mv/data/webbridge_mixed_dataset.py`:
   - Added `return_view_mask: bool = False` to `WebBridgeCanonical17Dataset`, `WebBridgeMixedDataset`, and `build_webbridge_mixed_dataloaders`.
   - Added `webbridge_mixed_collate_fn_with_mask` that returns `(x, y, K, R, t, dataset_ids, view_mask)`.
   - Exported the new collate function from `motionflow_mv/data/__init__.py`.
   - Default behavior is unchanged; existing callers continue to receive the original 6-tuple.

2. **Audit script** `scripts/audit_mixed_loader.py`:
   - Validates YAML split manifests against the canonical `.npz` layout.
   - Detects missing arrays, shape mismatches, unit mismatches (m vs mm), and mixed `(n_views, n_joints)`.
   - Supports both the `train_paths/train_names` format and the plain `train/val/test` format.

3. **Tests** `tests/test_webbridge_mixed_dataset_v25.py`:
   - Verifies view-mask shape and values for H36M (4 views) and MPI (14 views).
   - Verifies the new collate function and dataloader builder.
   - Verifies backward compatibility with the old collate function.

## Usage

```python
from motionflow_mv.data import build_webbridge_mixed_dataloaders

train_loader, val_loader = build_webbridge_mixed_dataloaders(
    train_paths=[...],
    train_names=[...],
    val_paths=[...],
    val_names=[...],
    clip_len=9,
    batch_size=4,
    return_view_mask=True,
)

for x, y, K, R, t, dataset_ids, view_mask in train_loader:
    # view_mask: (B, 14) bool, True for real views
    pred, geom_loss = v25_model(x, K=K, R=R, t=t, view_mask=view_mask)
```

Run the audit:

```bash
python scripts/audit_mixed_loader.py configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml
```

## Issues detected in current configs

Running the audit on `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml` reports:

- `data/webbridge/h36m_meters/s_09_acts_07_multiview_m.npz`: `joints_3d` range ~105, looks like millimeters instead of meters.
- `data/webbridge/h36m_meters/s_09_acts_09_multiview_m.npz`: `joints_3d` range ~140, looks like millimeters instead of meters.
- Mixed `n_views` across manifest: `[4, 14]`. `MultiViewGeometryFusionV25` is instantiated once per run and expects a fixed view count.
- Mixed `n_joints` across manifest: `[17, 28]`. Use `WebBridgeCanonical17Dataset` for a common skeleton.

## Proposed next steps

1. **Unit consistency:** Re-convert the flagged H36M files to meters, or confirm the intended unit and document it. The v25 reprojection loss assumes metric 3D points.
2. **Per-subset manifests:** Do not mix 4-view (H36M) and 14-view (MPI) files in the same training manifest. Create separate manifests (or separate loader instances) for each fixed `(n_views, n_joints)` pair.
3. **Wire view mask into v25 training:** Update `experiments/train_*_v25*.py` to pass `return_view_mask=True` and feed the resulting mask into `MultiViewGeometryFusionV25.forward`.
4. **Joint visibility mask (future):** For datasets with padded joints or different skeletons, consider emitting a per-joint validity mask in addition to the view mask.
5. **Reproducibility:** Replace the global `random` module in `WebBridgeCanonical17Dataset.__getitem__` with a local `torch.Generator` or NumPy RNG seeded in `__init__` to make clip sampling deterministic across workers.

## Verification

- `python -m py_compile motionflow_mv/data/webbridge_mixed_dataset.py motionflow_mv/data/__init__.py scripts/audit_mixed_loader.py` passes.
- `python -m pytest tests/test_webbridge_mixed_dataset_v25.py -v` passes (7/7).
