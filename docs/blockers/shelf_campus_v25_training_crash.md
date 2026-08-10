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

## Suspected cause

- `Shelf/Campus` uses only 5 views, while the v25 geometry-fusion stack may have hard-coded view-count assumptions (camera view embeddings, graph structures, or variable-view max_views=14).
- `use_camera_view_embedding` / `use_variable_view_training` may assume a larger number of views than 5.
- Alternatively, `epipolar_bias` receives a degenerate camera after view padding/augmentation.

## How to unblock

1. Run with `CUDA_LAUNCH_BLOCKING=1` to get the exact line of the assert.
2. Disable `--use_epipolar_bias` and `--use_camera_view_embedding` in the Shelf/Campus script as a first sanity check.
3. If that works, re-enable components one-by-one to identify the culprit.
4. Alternatively, treat Shelf/Campus only as an evaluation set (via `experiments/eval_shelf_campus_standard.py`) rather than a training set, since the dataset is small.
