# Agent-01 Analysis: 3DPW Loader and `actual`-Mode Gaps

**Date:** 2026-08-09  
**Branch:** `v48-domain`  
**Scope:** `motionflow_mv/data/webbridge_mixed_dataset.py`, `motionflow_mv/data/webbridge_loader.py`, `experiments/convert_3dpw_multiview.py`  

## Executive summary

`experiments/convert_3dpw_multiview.py` already implements both `pseudo` and `actual` conversion modes, and the repository contains 60 pseudo-mode `.npz` files. However, the loader side does **not** consume the per-frame moving-camera data that `actual` mode writes, and no actual-mode manifest, test, or training path exists yet. The main blockers for v48 are: (1) per-frame camera arrays are ignored by the canonical loader, (2) the trainer/model expect static cameras, and (3) no actual-mode dataset wrapper or evaluation path has been created.

## What currently works

### 1. `convert_3dpw_multiview.py` supports `actual` mode

- Writes the real moving camera as three per-frame arrays:
  - `camera_K_frames` `(T, 3, 3)`
  - `camera_R_frames` `(T, 3, 3)`
  - `camera_t_frames` `(T, 3)`
- Keeps the standard `camera_K/R/t` slots as **first-frame placeholders** with `V=1` so that shape-based validators do not break.
- Computes 2D projections per frame with the correct frame-specific camera.
- Batch mode mirrors directory structure and appends `_actual.npz` / `_pseudo.npz` suffixes.

### 2. `WebBridgeCanonical17Dataset` loads pseudo-mode 3DPW correctly

- Maps 3DPW's 24-joint SMPL skeleton to the canonical 17-joint layout (`SKELETON_MAPS["3dpw"]`).
- Pads from `V=4` (pseudo) to `MAX_VIEWS=14` and returns `dataset_id=5` for 3DPW.
- With `return_view_mask=True`, the mask correctly marks the real views count.

### 3. Existing 3DPW pseudo manifest

`configs/splits/webbridge_3dpw_train_val.yaml` lists 60 pseudo files across train/validation/test.

## Actual-mode gaps

### Gap 1: Per-frame camera arrays are written but never read

`WebBridgeCanonical17Dataset.__init__` only loads:

```python
camera_K, camera_R, camera_t = _pad_cameras(
    data["camera_K"], data["camera_R"], data["camera_t"]
)
```

For an `actual` file, these are the **first-frame placeholders**, not the true per-frame cameras. The per-frame arrays `camera_K_frames`, `camera_R_frames`, `camera_t_frames` are present in the `.npz` but never accessed. Any geometry model that triangulates or reprojects using `K/R/t` will therefore use a wrong static camera for every frame after the first.

### Gap 2: Trainer datasets assume static cameras

`experiments/train_omniview_fusion_v5_webbridge_multi.py` has two hard-coded loaders:

```python
self.K = torch.from_numpy(data["camera_K"]).float()  # (V, 3, 3)
self.R = torch.from_numpy(data["camera_R"]).float()  # (V, 3, 3)
self.t = torch.from_numpy(data["camera_t"]).float()  # (V, 3)
```

These tensors are returned once per clip and broadcast across time. There is no code path to load per-frame `(T, V, 3, 3)` / `(T, V, 3)` cameras.

### Gap 3: `OmniMultiViewFusionV5` expects static cameras

`motionflow_mv/fusion/omniview_fusion_v5.py` accepts `cameras: List[Camera]` or `(K, R, t)` tensors with shape `(B, V, 3, 3)` / `(B, V, 3)`. A per-frame camera API would require a new forward signature or a wrapper that expands `(B, T, V, ...)` cameras.

### Gap 4: `return_view_mask` defaults to `False`

Actual mode has `V=1` real view and 13 padded views. Without `return_view_mask=True`, geometry code has no reliable way to know which view is real. The v46/v47 code paths already use `return_view_mask`, but:

- `build_webbridge_mixed_dataloaders` defaults `return_view_mask=False`.
- The existing 3DPW test only exercises `return_view_mask`, but does not exercise `actual` mode.

### Gap 5: No actual-mode loader wrapper

The v48 proposal references:

```python
motionflow_mv/data/webbridge_3dpw_actual_loader.py
```

This file does **not** exist. A dedicated wrapper is needed to:

- Load per-frame cameras from `camera_*_frames`.
- Return cameras with shape `(T, V, ...)` instead of `(V, ...)`.
- Optionally expose a single-view `(V=1)` tensor that the model can consume without the full geometry stack (since 3DPW actual is monocular).

### Gap 6: No actual-mode manifest or tests

- `configs/splits/webbridge_3dpw_train_val.yaml` only lists `_pseudo.npz` files.
- `tests/test_webbridge_3dpw_loader.py` hard-codes a `_pseudo.npz` path and tests neither `actual` mode nor per-frame cameras.
- `experiments/eval_residual_3dpw_pseudo.py` only evaluates pseudo-mode 3DPW.

### Gap 7: `webbridge_loader.py` still has a stub for 3DPW

```python
def convert_3dpw(...):
    raise NotImplementedError("3DPW converter is a stub.")
```

`webbridge_loader.py` is the canonical converter CLI, but its 3DPW entry is a stub. It is not wired to `experiments/convert_3dpw_multiview.py`, so users cannot run a single CLI command to convert 3DPW.

### Gap 8: Unit / scale ambiguity

- 3DPW `jointPositions` are in meters.
- H36M/MPI/AIST++ manifests conventions vary; AIST is scaled by `0.01` in `convert_aistpp`.
- No runtime check verifies that 3DPW and studio data share the same world unit before mixing.
- The 3DPW skeleton map maps `head_top` to `head` (index 15 twice), which is only an approximation.

## Impact on v48

| Gap | Severity | Blocks v48? | Owner task in action plan |
|-----|----------|-------------|---------------------------|
| 1 (per-frame cameras ignored) | High | Yes | Agent-04 (loader), Agent-06 (model wiring) |
| 2 (trainer static cameras) | High | Yes | Agent-07 (trainer) |
| 3 (model expects static cameras) | High | Yes | Agent-06 (model wiring) |
| 4 (view_mask default) | Medium | Partially | Agent-04, Agent-10 (tests) |
| 5 (no actual-mode wrapper) | High | Yes | Agent-04 |
| 6 (no manifest/tests) | Medium | Yes | Agent-10, Agent-11 (eval) |
| 7 (stub converter) | Low | No, but confusing | Agent-04 or backlog |
| 8 (unit/scale) | Medium | Potentially | Agent-04 (data), Agent-11 (eval) |

## Recommendations for downstream agents

1. **Agent-04 (loader):** Create `WebBridge3DPWActualDataset` or extend `WebBridgeCanonical17Dataset` with an `actual` flag that reads `camera_*_frames` and returns per-frame cameras. Ensure `return_view_mask=True` is enforced for actual mode.
2. **Agent-06 (model wiring):** Decide whether actual-mode 3DPW trains through the full multi-view geometry stack (using per-frame cameras) or through a dedicated monocular branch. If the former, update `OmniMultiViewFusionV5.forward` to accept optional per-frame cameras.
3. **Agent-07 (trainer):** Update `TemporalClipDataset` / `RandomClipDataset` or the mixed-loader path to pass per-frame cameras when present. Add a CLI flag such as `use_3dpw_actual_train`.
4. **Agent-10 (tests):** Add `tests/test_webbridge_3dpw_actual_loader.py` that verifies per-frame camera shapes and that the first-frame placeholder equals the first per-frame camera.
5. **Agent-11 (eval):** Add actual-mode paths to `eval_variable_views.py` and report `MPJPE@1` for 3DPW actual.
6. **Agent-08/09 (config/script):** Create the smoke config/script to exercise both pseudo and actual modes.

## Files reviewed

- `motionflow_mv/data/webbridge_mixed_dataset.py`
- `motionflow_mv/data/webbridge_loader.py`
- `experiments/convert_3dpw_multiview.py`
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `tests/test_webbridge_3dpw_loader.py`
- `configs/splits/webbridge_3dpw_train_val.yaml`

## Conclusion

The converter is ready for `actual` mode, but the loader/trainer/model pipeline is not. The smallest next step is to implement a dedicated actual-mode loader that surfaces per-frame cameras and to wire that loader into the trainer with an explicit `return_view_mask=True` requirement. Until that happens, v48 cannot train or evaluate on real moving-camera 3DPW data.
