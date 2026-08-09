# v49: Multi-View Video Input Format

**Status:** Proposal  
**Labels:** `experiment`, `P1-next`, `data`  
**Tracking issue:** #167 (proposed)  
**Depends on:** v46-SVG (#160), v47-temporal (#162), v48-domain (#164)  

---

## 1. Problem Statement

The current pipeline relies on an implicit canonical `.npz` layout (`points_2d`, `confidences`, `joints_3d`, `camera_K`, `camera_R`, `camera_t`) that was sufficient for static, fixed-view studio captures. As v46–v48 add sparse-view training, temporal aggregation, domain generalization, and the coming v49 real-time/streaming mode, the input format has become a bottleneck:

1. **No explicit metadata.** Frame rate, frame indices, video ids, view names, and dataset/domain ids are not part of the loaded sample. v48 domain conditioning therefore requires ad-hoc `dataset_id` plumbing, and v47 temporal windows cannot distinguish true sequence boundaries from clip padding.
2. **Static cameras only by convention.** Per-frame moving-camera arrays (`camera_K_frames`, etc.) are supported only as a special-case 3DPW `actual` mode hack, not as a first-class input.
3. **No standard reliability feedback channel.** The v37 self-critique view-reliability scores and v46 sparse-view dropout masks live in separate modules; there is no canonical tensor in the input that carries per-view uncertainty into the model.
4. **Streaming/causal ambiguity.** The model cannot tell whether a clip is a contiguous video window or a batch of independent frames, which complicates v47 causal masks and v49 streaming state resets.
5. **Loader/model coupling.** `OmniMultiViewFusionV5.forward` expects positional tensor arguments, so every new modality (domain id, per-frame cameras, reliability feedback) requires a new positional argument and matching collate logic.

v49 therefore proposes a single, self-describing **multi-view video input format** that all loaders emit and all model variants consume.

---

## 2. Proposed Approach

Introduce `MultiviewVideoInputFormatV49`, a dictionary/dataclass that wraps tensors and metadata. It is **backward compatible** with existing `.npz` files and becomes the canonical input for v46–v49.

```text
.npz / raw video
    |
    ▼
[ Existing WebBridge loader ]  ──►  MultiviewVideoInputFormatV49
    |                                |
    |    ┌───────────────────────────┤
    |    ▼                           ▼
    |  v46 Sparse-View           v48 Domain
    |  Generalization            Conditioning
    |    |                           |
    |    ▼                           ▼
    |  v49 Reliability         v47 Temporal /
    |  Feedback Channel        v49 Streaming
    |    |
    |    ▼
    └► 3-D pose + reprojection residual
```

Key additions:

- **`metadata`** carries `fps`, `frame_idx`, `video_id`, `dataset_id`, `view_names`, and a `causal` flag. This unblocks clean v47 causal masks and v49 streaming state resets.
- **`camera_valid`** is a first-class boolean mask `(T, V)` for per-frame camera availability, replacing ad-hoc 3DPW special cases.
- **`reliability_feedback`** `(T, V, J)` carries per-view/joint reliability or reprojection residuals from the previous forward pass, closing the self-evolution loop with v37/v46.
- **`input_format_version`** string keeps the format versioned for future changes.

The format is **additive**: any field not present falls back to the existing default, so old `.npz` files and old model code keep working.

---

## 3. Concrete Code-Level Changes

### New files

| File | Purpose |
|------|---------|
| `motionflow_mv/data/multiview_video_input_format_v49.py` | `MultiviewVideoInputFormatV49` dataclass, validation, conversion helpers, and a `collate_v49` function. |
| `tests/test_multiview_video_input_format_v49.py` | Unit tests for format validation, backward conversion, and collate behavior. |
| `configs/benchmark_v49_input_format_smoke.yaml` | Smoke config that exercises the new input path. |
| `scripts/run_v49_input_format_smoke_local_4090.sh` | Local RTX 4090 smoke launch script. |

### Modified files

| File | Change |
|------|--------|
| `motionflow_mv/data/webbridge_mixed_dataset.py` | Emit `MultiviewVideoInputFormatV49` samples when `use_v49_input_format=True`; otherwise keep the old tuple output. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Accept either the legacy positional signature `(x, K, R, t, ...)` or a `MultiviewVideoInputFormatV49` dict; extract tensors and pass metadata to v46/v47/v48 modules. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add CLI flags, switch to `collate_v49` when enabled, and pass `metadata`/`reliability_feedback` through the training loop. |
| `motionflow_mv/data/view_dropout_augmentation_v46.py` | Read/write `view_mask` directly from the format and optionally update `reliability_feedback`. |

### New training/model flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_input_format` | bool | `False` | Master switch; loaders emit and the model consumes the v49 format. |
| `v49_input_format_version` | str | `"1.0"` | Version string checked by the loader and model. |
| `v49_input_include_metadata` | bool | `True` | Include `fps`, `frame_idx`, `video_id`, etc. |
| `v49_input_reliability_feedback` | bool | `True` | Reserve the `reliability_feedback` channel. |
| `v49_input_per_frame_cameras` | bool | `True` | Normalize all camera arrays to `(T, V, 3, 3)` / `(T, V, 3)` inside the format. |

### `MultiviewVideoInputFormatV49` schema

```python
@dataclass
class MultiviewVideoInputFormatV49:
    points_2d: torch.Tensor           # (T, V, J, 2)
    confidences: torch.Tensor         # (T, V, J)
    joints_3d: torch.Tensor           # (T, J, 3)
    camera_K: torch.Tensor            # (T, V, 3, 3) or (V, 3, 3)
    camera_R: torch.Tensor            # (T, V, 3, 3) or (V, 3, 3)
    camera_t: torch.Tensor            # (T, V, 3)    or (V, 3)
    view_mask: torch.Tensor           # (T, V) bool
    dataset_id: int                   # domain id for v48
    metadata: Optional[Dict] = None  # fps, frame_idx, video_id, view_names, causal
    reliability_feedback: Optional[torch.Tensor] = None  # (T, V, J)
    input_format_version: str = "1.0"
```

---

## 4. Risks / Failure Modes

| Risk | Mitigation |
|------|------------|
| Breaks old `.npz` loading paths | Keep the v49 path optional; existing loaders/model code remain the default. |
| Per-frame camera arrays increase memory | Store `(V, ...)` static arrays as-is; broadcast to `(T, V, ...)` lazily inside the model. |
| Downstream modules assume `(V, 3, 3)` camera shape | Add a single helper `broadcast_cameras_to_T(...)` in the new format module. |
| `reliability_feedback` leaks future information | Detach and treat as an input-only signal during training; never backprop through it. |
| Metadata fields vary across datasets | Define a minimal required schema and allow optional extras; validate in unit tests. |
| Collate becomes slower | Implement `collate_v49` as a thin `default_collate` wrapper with metadata batching. |

---

## 5. Success Metrics and Recommended Experiments

### Smoke experiment (RTX 4090)

| Field | Value |
|-------|-------|
| Config | `configs/benchmark_v49_input_format_smoke.yaml` |
| Hardware | Local RTX 4090 (24 GiB) |
| Duration | ~30–60 min |
| Goal | `val_MPJPE` matches v46 smoke within `0.1 mm`; no NaN/OOM; loader throughput `>100 samples/sec`. |

The smoke config should set `use_v49_input_format: true` but otherwise reuse the v46 smoke model and training recipe. Success means the new input path is a pure refactor.

### Full experiment (A800-D)

| Field | Value |
|-------|-------|
| Config | `configs/benchmark_v49_input_format_full.yaml` |
| Hardware | A800-D |
| Baseline | v46/v47/v48 checkpoint trained without the v49 format |
| Goal | No regression in `val_MPJPE@full` and `MPJPE@2/3/4`; per-domain MPJPE unchanged. |

### Evaluation

Extend `experiments/eval_variable_views.py` to optionally emit the v49 format and report:

- `loader_samples_per_sec`
- `input_format_version` used
- `val_MPJPE` vs the non-v49 baseline on the same checkpoint

---

## 6. Self-Evolution Feedback Loop

The v49 input format closes the self-evolution loop introduced in v36/v37 and used by v46 sparse-view reliability:

1. **Forward pass:** the model receives `reliability_feedback` as an additional input channel alongside `points_2d` and `confidences`.
2. **v37 self-critique / v46 reliability:** these modules predict per-view reprojection residuals and reliability scores.
3. **Feedback write-back:** at the end of the forward pass, the per-view residuals are detached and written back into `reliability_feedback` for the next iteration.
4. **Test-time / streaming:** in v49 streaming mode, this feedback can be persisted across frames so the model rapidly down-weights a consistently noisy camera view.

This makes the input format itself a carrier of the model's own uncertainty, turning the static `.npz` pipeline into a self-improving multi-view video input.

---

## 7. Next Steps

1. Implement `MultiviewVideoInputFormatV49` and unit tests.
2. Add the v49 conversion path to `WebBridgeCanonical17Dataset`.
3. Wire `use_v49_input_format` into `OmniMultiViewFusionV5` and the trainer.
4. Run smoke on RTX 4090 and compare against the v46 smoke baseline.
5. Update downstream v47/v48/v49 modules to consume `metadata` and `reliability_feedback` from the format.
