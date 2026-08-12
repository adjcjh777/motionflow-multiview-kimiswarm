# Camera-Parameter-Conditioned Fusion — iter15 Proposal

## One-sentence hypothesis

By explicitly injecting calibrated camera intrinsics and extrinsics into the **view-selection weight head** and the **3D residual refinement head**, the model can learn to down-weight geometrically inconsistent views and bias its residual correction toward the physical metric scale of the camera rig, improving multi-view video → 3D skeleton fusion, calibration alignment, and cross-view robustness while leaving the differentiable DLT triangulation backbone untouched.

## Related existing files/modules

- Anchor model and training pipeline:
  - `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`
  - `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py`
  - `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py`
  - `motionflow_mv/fusion/principal_point_correction.py`
  - `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
- Geometry helpers already in the repo:
  - `motionflow_mv/fusion/epipolar_attention_bias.py`
  - `motionflow_mv/fusion/camera_positional_encoding.py`
- Losses:
  - `motionflow_mv/losses/reprojection.py`
  - `motionflow_mv/losses/bone_length.py`
  - `motionflow_mv/losses/velocity.py`

## Proposed code changes

### 1. New model: camera-conditioned weight & residual heads

Create `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_camera_conditioned_model.py` with:

- `CameraConditionedWeightHead(nn.Module)`
  - Input: per-view per-joint feature `feat (N, J, V, d)`, camera matrices `K, R, t`.
  - Encodes each view’s camera parameters into a low-dimensional condition vector.
  - Concatenates condition to per-joint features and predicts view weight logits.
- `CameraConditionedResidualRefiner(nn.Module)`
  - Input: pooled feature `feat (N, J, d)`, raw triangulated 3D `raw3d (N, J, 3)`, and camera matrices.
  - Pools camera parameters into a global rig condition and concatenates it to the residual MLP input.
- `RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned`
  - Subclasses `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
  - Replaces `self.weight_head` and `self.residual_mlp` with the camera-conditioned variants above.
  - Forward pass is otherwise identical to the anchor; all original flags (`return_pp_delta`, `return_raw`, etc.) are preserved.

No changes to the spatio-temporal transformer, principal-point correction, or DLT triangulation path.

### 2. New loss: camera-conditioned reprojection + scale consistency

Create `motionflow_mv/losses/camera_conditioned_loss.py` with:

- `camera_conditioned_reprojection_loss`: robust Charbonnier reprojection error weighted by predicted view weights and normalized by view baseline / ray angle to emphasize physically difficult views.
- `camera_conditioned_scale_loss`: temporal bone-length consistency loss that encourages stable skeleton scale across frames, using the same parent-index skeleton definition as `bone_length_loss`.

Both losses are optional auxiliaries and can be dropped in without modifying existing loss logic.

### 3. Training integration

- Add a new `model_type="camera_conditioned"` branch in `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`.
- Wire the new loss terms via command-line flags `--camera_conditioned_reproj_weight` and `--camera_conditioned_scale_weight`.
- Keep all existing camera augmentation, principal-point loss, and velocity loss paths unchanged.

## Training / smoke plan (≤5 epochs, RTX 4090)

Smoke test on MPI-INF-3DHP S2/Seq1 (held-out validation):

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --val   data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --model_type camera_conditioned \
  --clip_len 9 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --epochs 5 --batch_size 8 --lr 1e-3 \
  --pp_loss_weight 0.1 --cam_aug_pp 5.0 --cam_aug_focal 0.01
```

- Estimated runtime: ~25–35 min per epoch on RTX 4090.
- Total smoke: ~2–3 hours for 5 epochs.
- Validation MPJPE is printed after every epoch; checkpoint saved on best val.

## Success metrics

1. **Clean MPJPE**: ≤ 9.10 mm on MPI-INF-3DHP S2/Seq1 (anchor = 9.32 mm).
2. **Calibration robustness**: Under camera perturbation (rot std 2.0°, trans std 0.02 m, pp std 10 px, focal std 5%), MPJPE degradation relative to clean should be ≤ 15%.
3. **Cross-view robustness**: With random view dropout to 2 views during evaluation, MPJPE increase vs. 4-view clean ≤ 20%.
4. **Physical alignment**: Reprojection error on validation reduced by ≥ 5% compared to the anchor at the same epoch.

## Risk and fallback

- **Risk**: Camera conditioning adds parameters and may overfit the small MPI-INF-3DHP training split; it could also make the weight head harder to initialize, temporarily hurting convergence.
- **Fallback**: If the smoke test does not improve over the anchor after 5 epochs, disable the camera-conditioned weight head and keep only the camera-conditioned residual refiner (or vice versa). If still no gain, fall back to the unmodified anchor and treat the new modules as a negative ablation in the paper.
