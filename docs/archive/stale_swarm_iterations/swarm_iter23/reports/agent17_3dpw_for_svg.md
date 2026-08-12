# Agent-17: WebBridge 3DPW Loader Analysis for v46 Sparse-View Generalization

**Owner:** Agent-17 (ANALYZE)  
**Tracking issue:** #160  
**Branch:** `v46-svg`  
**Date:** 2026-08-09

## Executive summary

The WebBridge 3DPW loader is already integrated into the mixed-dataset pipeline and is technically ready for v46 sparse-view generalization. Its distinctive feature is that it is produced by `experiments/convert_3dpw_multiview.py` in two modes — `pseudo` (static virtual 4-view rig) and `actual` (single moving camera). For v46 we should treat 3DPW as both a **multi-view training source** (pseudo) and a **sparse-view stress-test domain** (actual), while reusing the existing `return_view_mask` and view-padding machinery.

## Current loader architecture

### Files involved

| File | Role |
|------|------|
| `motionflow_mv/data/webbridge_mixed_dataset.py` | Main `WebBridgeCanonical17Dataset` / `WebBridgeMixedDataset`; pads views to `MAX_VIEWS=14` and maps all sources to a canonical 17-joint skeleton. |
| `experiments/convert_3dpw_multiview.py` | Converts raw 3DPW `.pkl` sequences into canonical `.npz`. Supports `pseudo` (4 virtual views) and `actual` (1 moving camera) modes. |
| `configs/splits/webbridge_all_train_mixed.yaml` | Mixed manifest that already lists 3DPW pseudo files under `train_paths` / `val_paths`. |
| `tests/test_webbridge_3dpw_loader.py` | Smoke test for 3DPW `.npz` loading and 17-joint / 14-view padding. |

### 3DPW-specific mapping

In `webbridge_mixed_dataset.py`:

```python
"3dpw": np.array(
    [0, 2, 5, 8, 1, 4, 7, 6, 12, 15, 16, 18, 20, 17, 19, 21, 15],
    dtype=np.int64,
),
```

This maps the 24-joint SMPL skeleton to the canonical 17-joint layout. `head_top` is aliased to `head` (index 15), which is the only approximation in the map. All other joints have a direct anatomical match.

### Data shape and padding

A 3DPW pseudo sample loaded through `WebBridgeCanonical17Dataset(..., return_view_mask=True)` returns:

- `x`: `(T, 14, 17, 3)` — 2D points + confidence, padded to 14 views.
- `y`: `(T, 17, 3)` — 3D ground truth.
- `K, R, t`: `(14, 3, 3)` / `(14, 3)` — padded camera parameters.
- `dataset_id`: `5` (3DPW).
- `view_mask`: `(14,)` bool mask, first `n_views` (typically 4 for pseudo, 1 for actual) are `True`.

This is identical to how H36M/MPI/AIST samples are exposed, so 3DPW fits the v46 module with no loader-side changes.

## Strengths for v46

1. **Already multi-view in pseudo mode.** The 4-view pseudo rig gives a genuine multi-view signal for training and is smaller than the 14-view H36M/MPI rigs, which is ideal for testing sparse-view robustness.
2. **Existing view-mask support.** `return_view_mask=True` is already wired through `build_webbridge_mixed_dataloaders` and the collator, so the v46 dropout/reliability module can ignore padded views.
3. **Domain tag available.** `dataset_id == 5` lets the model learn domain-specific behavior if desired (e.g., smaller expected view count for 3DPW).
4. **Mixed manifest already includes 3DPW.** No manifest plumbing is required; `configs/splits/webbridge_all_train_mixed.yaml` already references 3DPW pseudo files.

## Gaps and risks for v46

1. **Pseudo mode is synthetic but single-scene.** The 4 virtual cameras are static and co-circular; it is a weaker multi-view test than H36M/MPI. Dropping views in pseudo mode still leaves perfect calibration, so it cannot fully validate real-world sparse capture.
2. **`actual` mode is under-used.** It produces a single moving camera with per-frame extrinsics. The current loader reads only the static placeholder `camera_K/R/t`; the per-frame arrays `camera_K_frames`, `camera_R_frames`, `camera_t_frames` are stored but ignored. This mode is the closest to true sparse/mobile capture and should be surfaced for evaluation.
3. **View count is hard-coded in pseudo mode.** `convert_3dpw_sequence` defaults to `n_views=4`. For v46 ablations we may want 2, 3, 4, or 6 pseudo views without re-converting the whole dataset.
4. **No per-dataset dropout config.** 3DPW pseudo only has 4 real views. Applying the same `v46_svg_view_dropout_prob=0.3` uniformly across H36M (14 views) and 3DPW (4 views) could leave 3DPW clips with only ~2.8 effective views on average, which may be too aggressive for stable gradients early in training.
5. **Confidence values are binary visibility masks.** In pseudo mode, `confidences` are computed from positive camera-space depth, not detector confidence. The v46 reliability head should not assume Gaussian detector uncertainty is present.

## Proposed v46 data integration

### Option A — Minimal (recommended for first smoke)

Use the existing mixed loader as-is and rely solely on runtime view dropout in the training loop (`Agent-07`).

- **No loader changes.**
- `v46_svg_view_dropout_prob` and `v46_svg_min_views` are applied to batches after collation.
- 3DPW pseudo samples naturally exercise the sparse path because they already have only 4 real views within the 14-view padded tensor.
- Validation: run the smoke config with 3DPW included in the manifest; verify `view_mask` correctly zeros padded views.

### Option B — Dataset-aware dropout

Add a per-dataset dropout schedule so 3DPW is treated more gently during curriculum.

- Extend `view_dropout_augmentation_v46.py` (Agent-07) to accept a `dataset_id` vector and a mapping `dataset_id -> (dropout_prob, min_views)`.
- Suggested defaults:
  - `h36m/mpi/aist`: `dropout_prob=0.30`, `min_views=2`
  - `3dpw`: `dropout_prob=0.15`, `min_views=2` (fewer total views)
- This is still a training-time change, not a loader change.

### Option C — Expose `actual` mode for sparse-view evaluation

Create a small wrapper that loads the per-frame moving-camera arrays for evaluation only.

- Add a helper in `motionflow_mv/data/webbridge_3dpw_actual.py` (or inside `convert_3dpw_multiview.py`) that returns `(T, V, J, 2/3)` with `V=1` but time-varying camera parameters.
- Update `experiments/eval_variable_views.py` (Agent-13) to load 3DPW `actual` files and report `MPJPE@k`.
- This is a **data source only**; it does not affect training manifests.

### Option D — Re-project 3DPW with variable pseudo rigs

To get true variable-view training data without re-capturing, add an on-the-fly re-projection mode.

- In `convert_3dpw_multiview.py`, support `n_views` as a list or callable that samples a random number of virtual cameras per sequence.
- Store the sampled `n_views` and camera parameters in the `.npz` so the loader can expose the correct mask.
- **Cost:** requires re-running the converter and updating manifests. Recommended only after the initial v46 smoke.

## Recommended integration path for v46

1. **Smoke:** Option A. Keep the loader untouched, include 3DPW in `configs/benchmark_v46_svg_smoke.yaml`, and use the existing `return_view_mask` path.
2. **Full run:** Adopt Option B (dataset-aware dropout) so 3DPW is not over-dropped.
3. **Evaluation:** Adopt Option C for a dedicated 3DPW `actual`-mode `MPJPE@k` benchmark. This gives a clean in-the-wild sparse-view metric without touching training data.
4. **Post-v46:** Consider Option D if ablations show the model overfits to fixed 4-view rigs.

## Concrete files to touch (for downstream IMPLEMENT/EVAL agents)

- `motionflow_mv/data/view_dropout_augmentation_v46.py` (Agent-07): add optional `dataset_id`-aware dropout schedule (Option B).
- `experiments/eval_variable_views.py` (Agent-13): load 3DPW `actual` files and report `MPJPE@1` / `MPJPE@2` (Option C).
- `experiments/convert_3dpw_multiview.py`: already supports `n_views`; no change needed unless Option D is pursued.
- `configs/benchmark_v46_svg_smoke.yaml` (Agent-10): keep 3DPW pseudo files in the manifest.
- `docs/proposals/v46_sparse_view_generalization.md` (Agent-15): document the 3DPW actual-mode evaluation plan.

## Tests to run

- `pytest tests/test_webbridge_3dpw_loader.py -v` — confirms 3DPW still loads correctly.
- `pytest tests/test_webbridge_mixed_dataset_v25.py -v` — confirms mixed loader and `return_view_mask` work across domains.
- After v46 module lands: add a test that drops views on a 3DPW batch and asserts the reliability head respects `view_mask`.

## Open questions

1. Do we have enough disk space / bandwidth to convert 3DPW `actual` files for evaluation, or should we reuse the existing pseudo files only?
2. Should the v46 curriculum treat 3DPW as a separate domain with lower dropout, or should all domains share the same schedule and let the model adapt via `dataset_id`?
3. Is there a preference for adding the 3DPW `actual` loader to `motionflow_mv.data` as a public class, or keeping it as an evaluation-only script helper?
