# Proposal: Multi-Scale Cross-View Spatial Pyramid (MSCVSP)

## One-sentence hypothesis

Adding a multi-scale spatial pyramid inside the per-frame cross-view encoder—so that cross-view attention operates jointly over fine-grained single-joint features and coarse-grained limb/torso-scale features—will improve multi-view 3D pose fusion, physical calibration alignment, and per-view robustness without changing the temporal transformer or triangulation stages.

## Related existing files/modules

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py` — current iter14 anchor (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`).
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py` — parent with the spatio-temporal (time+view) transformer and residual refinement head.
- `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py` — base model providing the per-frame encoder (`_extract_frame_features`) with view-level and joint-level attention.
- `motionflow_mv/fusion/principal_point_correction.py` — learned principal-point / focal-length correction layer used by the anchor.
- `motionflow_mv/fusion/multiscale_temporal_conv_model.py` — prior art for multi-scale *temporal* modelling; the new work is orthogonal (multi-scale *spatial* / cross-view).
- `motionflow_mv/fusion/graph_joint_relation.py` — skeleton-graph module already exploring anatomy-aware joint relations.
- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` — main training entry point.
- `experiments/ablate_spatial_pyramid.py` — smoke-test template for spatial-pyramid ablations.

## Proposed code changes

1. **New module** `motionflow_mv/fusion/cross_view_spatial_pyramid.py`
   - `CrossViewSpatialPyramid(nn.Module)`
     - Args: `d, n_views, scales=(1,2,4), n_heads=1`
     - For each spatial scale `s`, downsample the joint dimension `J` to `J // s` via adaptive average pooling.
     - Apply a lightweight per-scale cross-view attention block (`MultiheadAttention`, `LayerNorm`, `MLP`) on tokens shaped `(N*(J//s), V, d)`.
     - Upsample each scale back to `(N, V, J, d)` via 1-D linear interpolation along the joint axis.
     - Fuse all scales with a learnable weighted projection back to `d` and a residual connection.

2. **New model** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_spatial_pyramid_model.py`
   - Class: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSpatialPyramid`
   - Inherits from `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
   - Adds a `CrossViewSpatialPyramid` instance in `__init__`.
   - Overrides `_extract_frame_features` to call the base encoder, then passes features through the pyramid.
   - All other components (PP correction, spatio-temporal transformer, weight head, residual MLP, signatures) remain unchanged.

3. **New FusionModule wrapper** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_spatial_pyramid_module.py`
   - Class: `RayAttentionTemporalCrossviewResidualPrincipalPointSpatialPyramidFusionModule`
   - Wraps the model for the `FUSION_REGISTRY` pipeline.
   - Registers under name `ray_attention_temporal_crossview_residual_principal_point_spatial_pyramid`.

4. **Update** `motionflow_mv/fusion/__init__.py`
   - Import and register the new FusionModule.

5. **New smoke/ablation script** `experiments/ablate_multiscale_crossview_spatial_pyramid.py`
   - CPU/GPU smoke test: instantiates the new model, runs a few forward/backward steps on a small MPI-INF-3DHP smoke `.npz`, and writes a short report to `docs/swarm_iter_next/`.

6. **(Optional training integration)** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
   - Add `spatial_pyramid` to the `model_type` choices and instantiate the new model. Not required for the proposal skeleton, but kept as a one-line follow-up once smoke passes.

## Training / smoke plan (≤5 epochs, RTX 4090)

- **Smoke**: Run `experiments/ablate_multiscale_crossview_spatial_pyramid.py` on the MPI-INF-3DHP smoke `.npz`.
  - `d=32`, `n_st_layers=2`, `scales=(1,2,4)`, `clip_len=9`, `batch_size=4`, `max_batches=20`, `epochs=2`.
  - Expected runtime on RTX 4090: ~1–2 minutes.
- **Short train**: If smoke succeeds, run the main training script with `model_type=spatial_pyramid` for ≤5 epochs on the full MPI-INF-3DHP S2/Seq1 train split.
  - `d=64`, `n_st_layers=2`, `clip_len=13`, `batch_size=8`, `epochs=5`.
  - Estimated runtime on RTX 4090: ~30–45 minutes for 5 epochs (~6–9 min/epoch).
- **A800 read-only validation (cross-dataset)**: Re-use the existing eval script on MPI-INF-3DHP S2/Seq1; do not run new jobs, only read existing outputs if available.

## Success metrics

- **Primary**: MPI-INF-3DHP S2/Seq1 clean MPJPE ≤ 9.00 mm (improvement over anchor 9.32 mm).
- **Robustness axis**: Relative MPJPE degradation ≤ 5% when one view is dropped at test time (uses existing view-dropout eval).
- **Calibration axis**: Mean predicted principal-point offset on the validation set stays bounded (< 15 px) and reprojection error on refined 3D pose does not regress.
- **Training stability**: Smoke test completes without NaNs/Inf and validation loss decreases monotonically over the 5-epoch run.

## Risk and fallback

- **Risk 1 — Joint-scale pooling loses fine detail.** Adaptive pooling over joints may blur local joint positions, hurting MPJPE.
  - *Fallback*: Replace pooling with strided 1-D convolutions learned end-to-end, or keep only `scales=(1,2)`.
- **Risk 2 — Extra parameters make smoke/unstable training.** The pyramid adds ~1–2× `d²` parameters per scale.
  - *Fallback*: Share the cross-view attention weights across scales, or freeze the base model and train only the pyramid for the first 2 epochs.
- **Risk 3 — No measurable gain on S2/Seq1.**
  - *Fallback*: Treat the module as an ablation-only component; the change is a single file subclass, so reverting means deleting the new model file and using the anchor checkpoint unchanged.
- **Risk 4 — Runtime regression on RTX 4090.**
  - *Fallback*: Reduce `scales` to `(1,2)` or downsample only the coarsest branch to keep the per-epoch time within the anchor's budget.
