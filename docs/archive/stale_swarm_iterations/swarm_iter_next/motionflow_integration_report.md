# Design: MotionFlow MultiViewFusionPlugin Pipeline Integration

## 1. Motivation

The project already has a rich set of multi-view fusion backends under
`motionflow_mv/fusion/`, from geometric DLT baselines to the current best
`RayAttentionFusionModelTemporalResidual` (10.46 mm MPJPE on MPI-INF-3DHP).
However, each backend is currently consumed in a slightly different way:

- `motionflow_mv/pipeline.py` exposes only a low-level `MultiViewPipeline` that
  triangulates one frame at a time.
- `motionflow_mv/ir/multiview_adapter.py` knows how to fuse multiple
  `HumanMotionIR` objects, but it expects the caller to already have a
  `FusionModule` instance in hand.
- Learned models require manual checkpoint loading and device placement before
  they can be used.\nThis inconsistency makes it hard for downstream MotionFlow stages (visualization,
retargeting, policy training) to swap fusion backends transparently.  We need a
single, high-level **MultiViewFusionPlugin** that:

1. Exposes one API regardless of backend.
2. Handles camera unit normalization, device placement, and checkpoint loading
   automatically.
3. Produces a standardized `HumanMotionIR` so the rest of the pipeline does not
   need to know which fusion module generated the 3D pose.

## 2. Design Decisions

### 2.1 Plugin as a thin wrapper around `FusionModule`

`MultiViewFusionPlugin` (`motionflow_mv/pipeline_multiview_plugin.py`) wraps any
registered `FusionModule` and dispatches to it.  It does not re-implement fusion
logic; it only provides:

- **Camera scale normalization**: the plugin accepts an `input_scale` factor so
  that cameras given in millimeters or other units are converted to meters before
  fusion.
- **Device management**: learned backends are moved to `cuda` when available (or
  to a user-supplied `torch.device`).
- **Multiple input flavors**:
  - Raw arrays: `plugin.fuse(points_2d, confidences, cameras)`.
  - Per-view IRs: `plugin.fuse_irs(irs, cameras)`.
  - Prediction dictionaries: `plugin.fuse_from_predictions(predictions, cameras)`.

### 2.2 Reuse existing IR adapter

The plugin reuses `fuse_multiple_irs` from
`motionflow_mv/ir/multiview_adapter.py` rather than duplicating IR construction
logic.  This keeps the new file small and ensures the output IR is identical to
the one produced by the adapter.

### 2.3 Registry-driven backend selection

The plugin selects backends from the global `FUSION_REGISTRY`
(`motionflow_mv/fusion/fusion_module.py`).  This lets users switch between DLT,
attention, temporal-residual, or future variants without changing any pipeline
code:

```python
from motionflow_mv.pipeline_multiview_plugin import MultiViewFusionPlugin

plugin = MultiViewFusionPlugin("dlt")
plugin = MultiViewFusionPlugin("ray_attention_temporal_residual")
```

A convenience factory `create_multiview_plugin` is also provided for checkpointed
learned backends.

### 2.4 Return format

- `fuse()` returns a raw `(T, J, 3)` numpy array in meters.
- `fuse_irs()` returns a full `HumanMotionIR` with aligned root translation,
  per-view 2D observations, and provenance metadata.
- `fuse_from_predictions(..., return_ir=True)` constructs a minimal
  `HumanMotionIR` directly from raw predictions, lowering the barrier for
  production inference where the full upstream IR may not be available.

## 3. Expected Benefits

- **Unified API**: one plugin serves geometric baselines and learned models.
- **Backend-agnostic downstream code**: consumers receive a `HumanMotionIR` and
  do not need to know which fusion module produced it.
- **Easier benchmarking**: switching between `dlt`, `attention`, and
  `ray_attention_temporal_residual` for ablation studies is a one-line change.
- **Production-ready inference**: camera scaling and device placement are handled
  automatically.

## 4. Files Added/Modified

- `motionflow_mv/pipeline_multiview_plugin.py` (new)
  - `MultiViewFusionPlugin`
  - `create_multiview_plugin`
- `docs/swarm_iter_next/motionflow_integration_report.md` (new, this document)
- No existing files were modified.

## 5. Validation

A smoke test in `tests/test_pipeline_multiview_plugin.py` verifies:

1. The plugin can be instantiated with the `dlt` backend.
2. `fuse()` recovers a synthetic 3D skeleton projected into 4 views.
3. `fuse_irs()` produces a valid `HumanMotionIR` with the expected fields.
4. The plugin correctly lists all available backends.

The test runs on CPU with small synthetic data and does not require training.

## 6. Future Work

- Add async/batched inference for long sequences to avoid loading the entire
  temporal window into memory.
- Expose uncertainty/quality fields from learned backends in the output IR.
- Add a CLI entry point (e.g. `python -m motionflow_mv.pipeline_multiview_plugin`)
  for direct use from shell scripts.
