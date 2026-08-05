# Integration with the MotionFlow Pipeline

## Brief survey

The `ray_attention` model (`motionflow_mv/fusion/ray_attention_model.py`) is now a first-class `FusionModule` plugin registered in `motionflow_mv/fusion/__init__.py`. It consumes per-view 2D keypoints, confidences, and calibrated cameras, embeds observation + ray features, predicts per-view weights, and triangulates with a differentiable weighted DLT layer. On synthetic tests it reaches ~2 mm MPJPE, well below the noise floor, and the `RayAttentionFusionModule` wrapper lets the rest of the pipeline call it through the same `FusionModule.fuse(points_2d, confidences, cameras)` interface as the DLT baseline.

However, the *end-to-end* MotionFlow pipeline still treats fusion as an side-car step rather than a core stage. `motionflow_mv/pipeline.py` defines `MultiViewPipeline`, but its `fuse_frame` hard-codes `triangulate_confidence_weighted` and never uses the plugin registry. Meanwhile, the IR layer (`motionflow_mv/ir/human_motion_ir.py`) and adapters (`scorehmr_adapter.py`, `gvhmr_adapter.py`, `multiview_adapter.py`) provide a stable `HumanMotionIR` container, but single-view adapters only export SMPL parameters; they do not populate `per_view_2d`/`per_view_confidence`, so `fuse_multiple_irs` currently cannot run end-to-end without external 2D keypoints. Finally, the real-data loader (`motionflow_mv/data/shelf_loader.py`) and trainer (`experiments/train_ray_attention_real.py`) assume Shelf/Campus GT is already available, but they do not yet bridge the GVHMR/ScoreHMR IRs to 2D observations used by the fusion model.

In short: the *components* are compatible, but the *plumbing* between single-view human recovery, 2D projection, multi-view fusion, and the IR is incomplete. The topic for this phase is therefore **feature preservation, minimal intrusion, and IR compatibility**: integrate `ray_attention` without breaking existing DLT/attention/temporal plugins, keep single-view IR consumers unaffected, and ensure fused output remains a valid `HumanMotionIR`.

## Concrete actionable recommendations

1. **Refactor `MultiViewPipeline` to accept any `FusionModule` by name.**
   Replace the hard-coded triangulation in `motionflow_mv/pipeline.py` with a constructor argument `fusion_module: FusionModule` (default `DLTFusion()`). This preserves the old behavior by default, requires no changes to callers, and lets scripts choose `ray_attention` via the registry.

2. **Populate `per_view_2d`/`per_view_confidence` in the single-view IR adapters.**
   Add a helper in `motionflow_mv/ir/` that projects SMPL joints from a `HumanMotionIR` to 2D given a camera, producing `per_view_2d[sequence_id]` and `per_view_confidence[sequence_id]`. Call it in the GVHMR/ScoreHMR adapters when camera intrinsics and extrinsics are known. This is the only missing piece for `fuse_multiple_irs` to work end-to-end.

3. **Add a one-line registration-side-effect guard in `motionflow_mv/fusion/__init__.py`.**
   The module currently registers all plugins at import time, including `ray_attention`. Move the heavy network imports/registration into explicit `register_*()` functions and call them lazily from `experiments/`, or at minimum document the side effect. This prevents import-time GPU model loading and makes unit tests deterministic.

4. **Standardize scale handling between the loader and the IR.**
   `shelf_loader.py` returns cameras in mm; `train_ray_attention_real.py` converts to meters by mutating `cam.t`. The `HumanMotionIR` spec already requires `length_unit="meter"`. Add a `Camera.to_meters()` utility and use it in the loader so every downstream consumer (including `fuse_multiple_irs`) receives metric data without ad-hoc division.

5. **Add a pipeline-level integration test.**
   Create `tests/test_pipeline_integration.py` that (a) loads a synthetic SMPL sequence, (b) creates per-view IRs with populated 2D observations, (c) fuses them through `fuse_multiple_irs` with `RayAttentionFusionModule`, and (d) asserts the output IR has the same schema version and a `fusion_method="ray_attention"` provenance entry. This is the minimum bar for claiming IR compatibility.

## Potential risks

- **API churn in downstream consumers.** Changing `MultiViewPipeline` to require a `FusionModule` argument could break scripts that instantiate the pipeline directly. Mitigation: keep the default `DLTFusion()`.
- **SMPL joint set mismatch.** `RayAttentionFusionModel` defaults to 17 joints (COCO), while ScoreHMR/GVHMR output 23 body joints. A direct projection helper must map the SMPL joint regressor to the same skeleton used by the fusion model.
- **Implicit import-time side effects.** Eager registration loads PyTorch models when `motionflow_mv.fusion` is imported, which is brittle for CI or headless environments. Lazy registration removes this risk.
- **Scale regression.** Centralizing mm→m conversion in the loader is safe, but any script that currently divides by 100 itself will double-convert. The fix is to move the conversion into the loader and remove the manual scaling in trainers.

## Fit with the paper plan

For ICRA/CVPR 2027 the multi-view story must be more than a standalone fusion model; it must be a **drop-in upgrade** to the existing MotionFlow pipeline. The `ray_attention` plugin is the technical contribution, but the paper's system section needs to show that (1) single-view IRs feed multi-view fusion without re-architecture, (2) the fused IR remains compatible with downstream retargeting/policy modules, and (3) the DLT baseline continues to work unchanged. Completing the above integration steps gives a clean narrative: same input IRs, new fusion plugin, better 3D accuracy, zero disruption to the rest of the stack. The ablation can then compare `dlt`, `attention`, and `ray_attention` on Shelf/Campus using a single `MultiViewPipeline` call, which is both easier to review and easier to reproduce.
