# Integrating GVHMR/ScoreHMR Single-View Detectors into Multi-View Fusion

## 1. Topic survey

GVHMR (world-scale video) and **ScoreHMR** (camera-relative diffusion) can serve as per-view frontends in a calibrated multi-view system. MotionFlow's multi-view extension already has the necessary adapters and fusion backend:

- `motionflow_mv/ir/gvhmr_adapter.py` converts `hmr4d_results.pt` into a world-meter SMPL IR.
- `motionflow_mv/ir/scorehmr_adapter.py` converts ScoreHMR camera-relative SMPL params.
- `motionflow_mv/ir/multiview_adapter.py` fuses per-view `HumanMotionIR`s by running a `FusionModule` on `(T, V, J, 2)` 2D keypoints and confidences, then repackages the fused 3D skeleton.
- `motionflow_mv/fusion/ray_attention_model.py` implements ray-aware attention + differentiable weighted DLT and already reaches **0.0021 m MPJPE** on synthetic GVHMR multi-view projections.

The calibrated fusion backend is ready. The remaining gap is **frontend integration**: running GVHMR/ScoreHMR per view, extracting 2D observations, and piping them into the fusion pipeline.

## 2. Current state

| Component | Status | Notes |
|-----------|--------|-------|
| GVHMR → IR | Done | `gvhmr_adapter.py` produces world-coordinate SMPL IRs. |
| ScoreHMR → IR | Done | `scorehmr_adapter.py` produces camera-relative SMPL IRs. |
| IR fusion wrapper | Done | `multiview_adapter.py` fuses multiple IRs into one. |
| Ray-aware fusion | Strong | Best synthetic numbers; needs real-data training. |
| Per-view 2D extraction from detectors | **Missing** | No script populates `per_view_2d` from GVHMR/ScoreHMR. |
| SMPL consistency after fusion | **Partial** | `_align_root` only shifts `transl`; shape is averaged. |
| Real multi-view training | **Not started** | `train_ray_attention_real.py` expects 3D annotations. |

## 3. Actionable recommendations

### 3.1 Add a per-view 2D keypoint extraction script

Create `experiments/prepare_per_view_ir.py` that:

1. Takes a directory of synchronized calibrated views.
2. Runs GVHMR or ScoreHMR on each view independently.
3. Extracts per-view 2D keypoints, either by projecting the detector's SMPL joints through the known calibration or by using the detector's native 2D output.
4. Populates `per_view_2d` and `per_view_confidence` in each `HumanMotionIR`.

Output a directory of per-view `.pt` IRs plus `cameras.json`, so the rest of the pipeline can run offline.

### 3.2 Standardize the detector-to-fusion data contract

`fuse_multiple_irs` currently raises an error when 2D observations are missing. We should:

- Fix a canonical 17-joint order for fusion (e.g., COCO-17 / SMPL-17 subset).
- Add `reproject_smpl_to_2d(ir, camera)` in `motionflow_mv/ir/` to generate 2D keypoints on demand.
- Make `fuse_multiple_irs` fall back to reprojection when observations are absent.

### 3.3 Train `ray_attention` on GVHMR/ScoreHMR pseudo-GT

When real 3D ground truth is unavailable, generate pseudo-supervision:

1. Run GVHMR on each view, triangulate the per-view SMPL joints with `DLTFusion`, and treat the result as pseudo-GT.
2. Train `ray_attention` on these pseudo-3D targets.
3. Add a **reprojection loss** alongside 3D MSE so the network respects multi-view geometry.
4. Fine-tune on Shelf/Campus once raw data is available.

### 3.4 Add temporal and shape consistency

Independent per-view detectors may disagree on pose and shape. After fusion:

- Add a lightweight temporal smoother for the fused root and joint velocities.
- Replace simple `betas` averaging with a confidence-weighted average, or freeze a single shape estimate across the sequence.
- Optionally add a post-fusion SMPL reprojection fitter that minimizes multi-view reprojection error while enforcing bone-length and shape consistency.

### 3.5 Build a GVHMR demo benchmark

Use `data/gvhmr_demo/hmr4d_results.pt` to create a reproducible integration test:

1. Take the GVHMR world 3D joints as pseudo-ground-truth.
2. Project them through 3–5 virtual calibrated views to obtain 2D observations.
3. Run the per-view detection path (GVHMR → IR → reprojected 2D).
4. Fuse with each plugin and report MPJPE against the original GVHMR world joints.

Extend to real multi-view video when Shelf/Campus or in-house captures are available.

## 4. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| ScoreHMR is camera-relative. | Use calibrated rig and explicit `length_unit` / `world_from_reference` metadata to recover metric scale. |
| Per-view `betas` may disagree. | Use confidence-weighted averaging and freeze shape during reprojection fitting. |
| Skeleton mismatch (23/24 vs. 17 joints). | Maintain a canonical 17-joint fusion skeleton with bidirectional mapping to SMPL. |
| No real 3D GT for training. | Use synthetic GVHMR projections as controlled benchmark; validate on Shelf/Campus reprojection. |
| Runtime cost of per-view detectors. | Use detectors for SMPL init only; use a fast 2D detector for per-frame fusion. |
| ScoreHMR dependency overhead. | Keep it adapter-wrapped so GVHMR or another detector can be swapped in. |

## 5. Fit into the paper plan

This integration is essential for the CVPR/ICRA 2027 narrative: it shows that the ray-aware fusion module is not a synthetic toy but a practical wrapper around modern single-view HMR detectors. The paper can frame the contribution as a **detector-agnostic, calibrated multi-view fusion layer** that converts per-view GVHMR/ScoreHMR outputs into a single metric 3D pose. Key experiments will compare `ray_attention`, DLT, and robust triangulation under real detector noise, using both synthetic projections and Shelf/Campus multi-view data. Clean IR/adaptor design will be highlighted as an enabler of plug-and-play single-view detector integration.
