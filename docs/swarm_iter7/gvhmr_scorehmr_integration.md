# GVHMR/ScoreHMR integration

I cannot write files in this subagent context (only read/search tools are available), but I have prepared the requested report below. The parent agent can save it to:

**`docs/swarm_iter7/gvhmr_scorehmr_integration.md`**

---

# GVHMR/ScoreHMR Integration

## 1. Current state

Relevant files:

- `motionflow_mv/ir/gvhmr_adapter.py:26` — loads `hmr4d_results.pt` into `HumanMotionIR` with world-coordinate SMPL params.
- `motionflow_mv/ir/scorehmr_adapter.py:24` — converts ScoreHMR camera-relative SMPL params into `HumanMotionIR`; marks output as camera-relative.
- `motionflow_mv/ir/multiview_adapter.py:18` — fuses per-view `HumanMotionIR`s through a `FusionModule`, but raises `NotImplementedError` when `per_view_2d` / `per_view_confidence` are missing (`motionflow_mv/ir/multiview_adapter.py:61`).
- `experiments/demo_gvhmr_multiview_projection.py:1` — loads the demo `.pt`, runs SMPL forward, projects world joints through virtual cameras, and runs `ray_attention` / `ray_attention_v3`. It does **not** yet use the current best `ray_attention_temporal_residual` model.
- `motionflow_mv/fusion/ray_attention_temporal_residual_module.py:13` — wrapper for the 11.17 mm MPI-INF-3DHP checkpoint (`outputs/ray_attention_temporal_residual_final5.pth`).
- `data/gvhmr_demo/hmr4d_results.pt` exists (667 KB).

What works:

- GVHMR → IR conversion is stable.
- Synthetic multi-view projection + `ray_attention` reaches ~0.0021 m MPJPE on clean virtual views.
- `HumanMotionIR` already carries `per_view_2d` and `per_view_confidence` fields.

What is missing:

- No helper reprojects SMPL joints to 2D to populate `per_view_2d` / `per_view_confidence` on demand.
- `multiview_adapter.fuse_multiple_irs` cannot fall back to reprojection and therefore cannot fuse raw GVHMR/ScoreHMR IRs directly.
- `demo_gvhmr_multiview_projection.py` still uses the older v1/v3 plugins and virtual cameras instead of the best residual model.

## 2. Gap / opportunity

To make the paper claim “plug-in integration with modern single-view HMR detectors” credible, we need an end-to-end path from a GVHMR (or ScoreHMR) result to a fused metric 3D skeleton using the current best residual model. The bottleneck is not the fusion network (trained to 11.17 mm) but the **frontend plumbing**: deriving calibrated per-view 2D observations from detector SMPL output and feeding them to `RayAttentionTemporalResidualFusionModule`.

Opportunities:

1. Add an SMPL reprojection helper to populate `per_view_2d`/`per_view_confidence`.
2. Extend the GVHMR demo to load the 11.17 mm residual checkpoint and compare DLT vs. residual fusion under realistic detector-style noise.
3. If multi-view video becomes available, run GVHMR/ScoreHMR independently per view and fuse the resulting IRs.

## 3. Concrete next step

Add a reprojection fallback and wire the best residual model into the existing demo.

1. Implement `motionflow_mv/ir/smpl_reprojection.py` with `reproject_ir_to_2d(ir, camera, joints_idx=None) -> (points_2d, confidence)` that:
   - Runs SMPL/SMPL-X forward on `ir.pose`.
   - Selects a canonical 17-joint subset.
   - Projects world joints through the provided `Camera`.
   - Returns confidence = 1.0 for visible joints.
2. Update `motionflow_mv/ir/multiview_adapter.py` so `fuse_multiple_irs` calls the helper when `per_view_2d` is missing, with a warning.
3. Extend `experiments/demo_gvhmr_multiview_projection.py`:
   - Add `--ray_residual_checkpoint` defaulting to `outputs/ray_attention_temporal_residual_final5.pth`.
   - Instantiate `RayAttentionTemporalResidualFusionModule(..., input_scale=1.0)`.
   - Inject realistic noise: `--gvhmr_noise_std 2.0`, `--gvhmr_outlier_rate 0.05`.
   - Report MPJPE of `dlt`, `ray_attention`, and `ray_attention_temporal_residual` against the original GVHMR world joints.

This is a single-file helper + small demo extension; it does not touch the core residual model or training code.

## 4. Expected success metric

- GVHMR demo MPJPE vs. single-view world reference:
  - DLT baseline: ~0.040–0.050 m under 2 px noise.
  - `ray_attention_temporal_residual` (best checkpoint): **≤ 0.010 m** (target ≤ 0.008 m) under 2 px noise and 5 % outliers.
- No catastrophic forgetting: rerun MPI-INF-3DHP S2/Seq1 eval and confirm MPJPE stays within 11.17–11.7 mm.
- `tests/test_multiview_adapter.py` still passes after the reprojection fallback is added.

## 5. Risks / blockers

- **A800-D / Docker read-only**: do not modify any container or vendor data. Copy outputs out if needed.
- **WebBridge data**: if a real multi-view video is needed, download WebBridge/H36M subsets to `data/` but do not commit large files.
- **GVHMR demo `.pt` quality**: `data/gvhmr_demo/hmr4d_results.pt` is only 667 KB; it may be a placeholder.
- **Windows NumPy BLAS instability**: keep the new helper torch-only, matching the existing demo workaround.
- **Skeleton mismatch**: GVHMR/ScoreHMR use SMPL 23/24 joints; the fusion model uses 17 joints. The helper must define and document a fixed 17-joint mapping.

---

**Summary:** The adapters and best residual fusion model exist, but the demo still uses older plugins and virtual cameras. The highest-value next step is to add an SMPL-to-2D reprojection helper in `motionflow_mv/ir/`, wire the 11.17 mm `ray_attention_temporal_residual` checkpoint into `experiments/demo_gvhmr_multiview_projection.py`, and validate under realistic detector noise.