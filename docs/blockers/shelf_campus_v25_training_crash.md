# Blocker: v25 training crashes on Shelf/Campus non-circular smoke

**Date:** 2026-08-10

## What we tried

```bash
bash scripts/run_v25_shelf_campus_noncircular_smoke_local_4090.sh
```

Manifest: `configs/splits/shelf_campus_noncircular_smoke.yaml`

## What we found

Training crashes inside `motionflow_mv/fusion/epipolar_attention_bias.py::compute_epipolar_distance` with a CUDA device-side assert followed by a PyTorch indexing error:

```
RuntimeError: CUDA error: device-side assert triggered
...
Assertion `srcIndex < srcSelectDimSize` failed.
```

The same error occurs when mixing MPI (14 views) with Shelf/Campus (5 views), but here the manifest contains **only** Shelf/Campus `.npz` files.

## Debug attempted

Created `scripts/run_v25_shelf_campus_noncircular_smoke_debug_local_4090.sh` with:
- `--use_epipolar_bias false`
- `--use_camera_view_embedding` removed
- `--use_variable_view_training` removed

The same `srcIndex < srcSelectDimSize` CUDA assert still occurs.

## Suspected cause

- `Shelf/Campus` uses only 5 views, while the v25 geometry-fusion stack may have hard-coded view-count assumptions in other modules such as `use_deformable_cross_view_attention_v18`, graph structures, or view-embedding lookups.
- The assert pattern suggests an embedding/indexing layer receives a view index >= its vocabulary size.

## How to unblock

1. Run with `CUDA_LAUNCH_BLOCKING=1` to get the exact line of the assert.
2. Bisect by disabling `--use_deformable_cross_view_attention_v18` and other cross-view modules one-by-one.
3. Alternatively, treat Shelf/Campus only as an evaluation set (via `experiments/eval_shelf_campus_standard.py`) rather than a training set, since the dataset is small.
