# v27: Camera Refinement Inside the Geometry Fusion Loop

**Task identifier:** `design_v27_camera_refinement_in_loop`  
**Status:** Design proposal  
**Depends on:** v25 (`MultiViewGeometryFusionV25`), v26 (`TemporalGeometryFusionV26`), v26 calibration (`CameraRefinementV26`)

---

## 1. Problem

v25/v26 improve per-frame multi-view fusion, but they treat the camera intrinsics and extrinsics as fixed after the initial `principal_point_correction` and optional v21 neural-BA block. In practice the input calibration is often slightly off (focal length drift, principal-point bias, rig rotation/translation errors), and those errors propagate into:

* Ray-token computation in `compute_rays()` (`motionflow_mv/fusion/multiview_geometry_fusion_v25.py:75`).
* Epipolar and ray-intersection attention bias in `GeometryAwareCrossViewAttention` (`multiview_geometry_fusion_v25.py:196`).
* The learned depth-proposal triangulation head (`multiview_geometry_fusion_v25.py:275`).
* The reprojection loss that supervises geometry fusion.

A standalone differentiable camera-refinement module, `CameraRefinementV26`, already exists in `motionflow_mv/calibration/camera_refinement_v26.py`, but it is **not wired into the main training path**. The v27 candidate is to drop that module *inside* the v25/v26 forward pass so cameras are refined end-to-end with the same 3D-pose loss, and the refined cameras are fed back into the rest of the geometry-fusion block.

---

## 2. Proposed Method

### 2.1 Scope

Keep the change minimal and warm-startable:

1. Re-use the existing `CameraRefinementV26` class (no new heavy module).
2. Add one flag, `use_camera_refinement_v27`, to `OmniMultiViewFusionV5` and pass it down to `MultiViewGeometryFusionV25` / `TemporalGeometryFusionV26`.
3. Inside the v25/v26 forward pass, refine `K, R, t` once after the initial triangulation and before ray tokenisation.
4. Use the refined cameras for the rest of the block (ray tokens, geometry attention, depth triangulation, reprojection loss).

### 2.2 Integration points

#### `motionflow_mv/fusion/omniview_fusion_v5.py`

The v25/v26 instantiation block (around line 337–364) gets an extra toggle:

```python
self.use_multiview_geometry_fusion_v25 = use_multiview_geometry_fusion_v25
self.use_temporal_geometry_fusion_v26 = use_temporal_geometry_fusion_v26
self.v25_geom_loss_weight = v25_geom_loss_weight
self.use_camera_refinement_v27 = use_camera_refinement_v27  # new

if self.use_temporal_geometry_fusion_v26:
    self.multiview_geometry_fusion_v25 = TemporalGeometryFusionV26(
        ...,
        use_camera_refinement_v27=use_camera_refinement_v27,  # new
    )
elif self.use_multiview_geometry_fusion_v25:
    self.multiview_geometry_fusion_v25 = MultiViewGeometryFusionV25(
        ...,
        use_camera_refinement_v27=use_camera_refinement_v27,  # new
    )
```

No change is required to the existing v25/v26 hook at `omniview_fusion_v5.py:788–802`; the refinement happens inside the module.

#### `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`

In `MultiViewGeometryFusionV25.__init__`, add:

```python
self.use_camera_refinement_v27 = use_camera_refinement_v27
if self.use_camera_refinement_v27:
    from motionflow_mv.calibration.camera_refinement_v26 import CameraRefinementV26
    self.camera_refinement = CameraRefinementV26(
        n_steps=2,
        lr=0.05,
        refine_intrinsics=True,
        refine_extrinsics=True,
    )
```

In `MultiViewGeometryFusionV25.forward`, insert right after the initial triangulation / outlier step and before `compute_rays` (around line 483):

```python
if self.use_camera_refinement_v27 and self.camera_refinement is not None:
    K, R, t = self.camera_refinement(
        pts, pred_3d_init, K, R, t,
        weights=confidence,
        view_mask=view_mask,
    )
    # Refined cameras are now used for rays, attention and triangulation.

# World rays (now with refined cameras)
centre, direction = compute_rays(pts, K, R, t)
```

The same insertion applies to `TemporalGeometryFusionV26.forward` in `motionflow_mv/fusion/temporal_geometry_fusion_v26.py` before its `compute_rays` call (around line 318).

### 2.3 Why this is safe

* `CameraRefinementV26` already has a `residual_scale` gate initialised to zero, so at start of training the block is an identity mapping.
* Updates to `K, R, t` are clamped to small ranges (`max_focal_scale=0.05`, `max_rotation_deg=2`, `max_translation=0.05 m`).
* Because refinement is performed on the current `pred_3d_init`, the gradient flows back through the refined cameras into the geometry-fusion losses without a separate camera-only objective.

### 2.4 Inputs / outputs of the new loop

Inside `MultiViewGeometryFusionV25.forward` / `TemporalGeometryFusionV26.forward`:

```python
# Before refinement
pred_3d_init = triangulate_initial(...)      # (B, T, J, 3)

# v27 refinement (optional)
K, R, t = camera_refinement(pts, pred_3d_init, K, R, t, weights=confidence)

# After refinement: same downstream code, better cameras.
```

No change to the module’s external return signature: it still returns `(pred_3d_ref, geom_loss)`.

---

## 3. Expected Impact

| Metric | Expected change | Rationale |
|--------|-----------------|-----------|
| `val_MPJPE` | **5–10% relative reduction** on the best v25/v26 baseline. | Removing small camera bias improves ray triangulation and geometry-attention weighting. |
| 2-view MPJPE | **10–15% relative reduction** | Few-view triangulation is most sensitive to calibration error. |
| 4-view MPJPE | **7–10% relative reduction** | Still large gain from cleaner epipolar/ray bias. |
| 8-view MPJPE | **4–6% relative reduction** | Redundant views dilute the benefit. |
| 14-view MPJPE | **2–4% relative reduction** | Very high redundancy; camera refinement is mostly error cleanup. |

These are *targets*; the actual delta depends on how large the residual calibration errors are in the v25/v26 checkpoints.

---

## 4. Implementation Cost

| Item | Estimate |
|------|----------|
| Lines of code | ~60 (wiring + gate checks in `multiview_geometry_fusion_v25.py`, `temporal_geometry_fusion_v26.py`, and `omniview_fusion_v5.py`). |
| New files | 0 (re-uses `motionflow_mv/calibration/camera_refinement_v26.py`). |
| New trainable parameters | 1 scalar (`residual_scale` gate, already in the existing module). |
| Training time | ~5–10% per forward pass for the inner `autograd.grad` loop with `n_steps=2`. |
| Data needs | None; re-uses the same WebBridge/H36M/MPI clips. |

---

## 5. Risks / Mitigation

| Risk | Detection | Mitigation |
|------|-----------|------------|
| **Camera correction collapses** to trivial updates or over-fits to a single dataset. | Monitor `val_MPJPE`, refined-camera magnitude, and per-dataset reprojection loss. | Keep `residual_scale` gate initialised to zero; bound updates via `CameraRefinementV26` clamps. |
| **Gradient instability** from nested `autograd.grad` inside the forward pass. | Watch for NaN/Inf in `K`, `R`, `t`; check `torch.autograd.grad` works with AMP. | Use `create_graph=True`, small `n_steps`, and finite clamping. Test with `--smoke`. |
| **Conflict with existing intrinsic correction** (`principal_point_correction` in `omniview_fusion_v5.py:524`). | Compare runs with/without the existing focal/PP correction. | Start with `refine_intrinsics=False`, then enable if stable. |
| **Slower training** on A800 due to extra per-sample camera optimization. | Benchmark GPU seconds per iteration. | Reduce `n_steps` to 1 or set `lr=0.02` if overhead >10%. |
| **No measurable gain** if v25/v26 already corrects cameras implicitly. | Compare `val_MPJPE` with and without `use_camera_refinement_v27`. | Abort if first epoch shows `val_MPJPE` not improving within 1 mm. |

---

## 6. Minimal Experiment Plan

### 6.1 Smoke test

Run the existing v26 calibration smoke test (it already validates the module in isolation):

```bash
python scripts/smoke_camera_refinement_v26.py
```

Then run a model-level smoke by adding the proposed flag to the v25 training script:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --smoke \
    --use_multiview_geometry_fusion_v25 \
    --v27_use_camera_refinement_in_loop \
    --v25_geom_loss_weight 0.1 \
    --d 64 --n_heads 4 --batch_size 2 --epochs 1
```

(If `--smoke` is not supported by the current CLI, use the pytest-level smoke in `tests/test_camera_refinement_v26.py` plus a manual forward pass of `OmniMultiViewFusionV5` with `use_multiview_geometry_fusion_v25=True` and the new flag set.)

### 6.2 Small run

On the local RTX 4090 or an A800 smoke slot:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_multiview_geometry_fusion_v25 \
    --v27_use_camera_refinement_in_loop \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --d 128 --n_heads 4 --batch_size 8 --epochs 10 \
    --lr 1e-4 --v25_geom_loss_weight 0.1
```

Compare against the **identical** command without `--v27_use_camera_refinement_in_loop`.

### 6.3 Go/no-go after first epoch

* **Go:** `val_MPJPE` improves by ≥1 mm relative to the v25-only run.
* **No-go:** `val_MPJPE` is worse or within 0.3 mm; disable intrinsics refinement (`--v27_refine_intrinsics False`) and retry, or abandon the direction.

### 6.4 Variable-view test

Run inference at 2/4/8/14 views using the trained checkpoint and the variable-view test harness (e.g. `experiments/eval_omniview_fusion_v5_h36m.py` with `source_n_views=14` and per-file view subsetting). Look for the largest gains at low view counts.
