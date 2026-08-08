# v25 Multi-View Geometry Fusion Integration Plan

## Goal
Insert the `MultiViewGeometryFusionV25` module into `OmniMultiViewFusionV5` as an
optional block after v18 deformable cross-view attention and before the
spatio-temporal (ST) transformer, and wire it into the training loss.

## Files to touch

| File | Change |
|------|--------|
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Add `use_multiview_geometry_fusion_v25` and sub-toggles; instantiate module; call it in forward. |
| `motionflow_mv/training/trainer_v2.py` | No change required; geometry auxiliary losses are returned as part of the model output and consumed by `compute_loss`. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add CLI flags for v25 toggles and loss weight. |
| `tests/test_omniview_fusion_v5.py` | Add a toggle-on test. |

## Steps

1. **Add constructor flags**
   - `use_multiview_geometry_fusion_v25: bool = False`
   - `v25_use_geometry_attention: bool = True`
   - `v25_use_learned_depth_triangulation: bool = True`
   - `v25_use_geometry_bundle_adjustment: bool = True`
   - `v25_use_camera_joint_graph: bool = False`
   - `v25_geom_loss_weight: float = 0.1`

2. **Instantiate the module** after the v18 block and KAP block:
   ```python
   if self.use_multiview_geometry_fusion_v25:
       self.multiview_geometry_fusion_v25 = MultiViewGeometryFusionV25(
           d=d,
           n_heads=n_heads,
           n_views=n_views,
           use_geometry_attention=v25_use_geometry_attention,
           use_learned_depth_triangulation=v25_use_learned_depth_triangulation,
           use_geometry_bundle_adjustment=v25_use_geometry_bundle_adjustment,
           use_camera_joint_graph=v25_use_camera_joint_graph,
       )
   ```

3. **Forward pass hook**
   - After v18 cross-view attention, compute an initial triangulated pose (the
     same `pred_3d` that the existing Gauss-Newton refinement uses).
   - Call `MultiViewGeometryFusionV25.forward` with `feat`, `points_2d`,
     `K_corrected`, `R`, `t`, `pred_3d_init`, and `view_mask`.
   - Replace `K_corrected, R, t` with the refined camera parameters.
   - Add `v25_geom_loss_weight * geom_loss` to the auxiliary losses.

4. **Loss integration**
   - In `OmniMultiViewTrainer` / `build_compute_loss`, add the v25 geometry loss
     to the total loss if `args.use_multiview_geometry_fusion_v25` is true.
   - The geometry loss should be small relative to MPJPE; start with
     `v25_geom_loss_weight=0.1`.

5. **Testing**
   - Forward shape test with `use_multiview_geometry_fusion_v25=True`.
   - Identity-at-init test: with zero weights, refined pose ≈ initial triangulation.
   - Gradient flow test.

## Training recipe

Start from a v23 checkpoint (v18 + KAP, no neural BA). Freeze the v18 weights
for the first epoch and train only the v25 module with a higher learning rate on
the geometry loss. Then unfreeze and train end-to-end with the standard v23
loss mixture plus the v25 geometry loss.

## Expected outcome

- v25 should preserve the strong v18/v23 baseline at init.
- Over training, the geometry attention + learned depth triangulation should
  improve cross-view fusion, especially for variable-view and outlier-view
  cases.
- GeoBA provides a bounded, analytic camera refinement that supersedes the v21
  neural BA and avoids the 128.27 mm regression.
