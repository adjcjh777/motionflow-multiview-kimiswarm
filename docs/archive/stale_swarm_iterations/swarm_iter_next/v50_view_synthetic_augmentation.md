# v50 View Synthetic Augmentation: VirtualCameraViewSynthesisV50

## Summary

`VirtualCameraViewSynthesisV50` is a training-only augmentation block that synthesizes extra virtual camera views from the current 3-D pose estimate and injects their 2-D projections into the multi-view fusion stream. During training, the module takes the triangulated/estimated 3-D joints and a set of sampled virtual camera parameters, projects the joints into those cameras, adds light pixel noise and visibility masks, and treats the resulting keypoints as if they came from real sensors. The goal is to increase effective view diversity without capturing more data, hardening the model against sparse or dropped real views in v46/v47.

## Architecture

The module lives between the triangulated 3-D pose (or current lifted pose) and the sparse-view fusion head. It samples `v50_synth_num_virtual_views` virtual cameras on a spherical shell around the subject (azimuth/elevation jittered within configurable ranges), projects the 3-D joints using the same pinhole model, and appends the synthetic 2-D keypoints + cameras to the real batch. A visibility mask zeroes-out occluded or behind-camera joints. The virtual cameras are only used in training; at inference the module is bypassed so runtime latency is unchanged. Suggested implementation: a lightweight function in `motionflow_mv/data/view_synthetic_augmentation_v50.py` called inside the training loader or as a forward hook in `omniview_fusion_v5.py` when `self.training` is True.

## New config flags

| Flag | Default | Description |
|---|---|---|
| `use_v50_view_synthetic_augmentation` | `False` | Enable the module. |
| `v50_synth_num_virtual_views` | `2` | Number of virtual cameras to add per sample. |
| `v50_synth_camera_radius_range` | `[2.0, 5.0]` | Distance range (m) of virtual cameras from the subject. |
| `v50_synth_azimuth_range_deg` | `[-60, 60]` | Azimuth jitter relative to the mean real-camera direction. |
| `v50_synth_elevation_range_deg` | `[-20, 20]` | Elevation jitter. |
| `v50_synth_keypoint_noise_px` | `2.0` | Gaussian 2-D keypoint noise (pixels). |
| `v50_synth_apply_prob` | `0.5` | Probability of applying augmentation to a training batch. |
| `v50_synth_loss_weight` | `0.1` | Weight for the optional synthetic-view reprojection consistency loss. |

## Loss term

Optional auxiliary loss: after the final 3-D pose is predicted from the mixed real+synthetic views, we reproject it back into the *virtual* cameras and compute a 2-D reprojection error on the synthetic keypoints. This loss is added as `v50_synth_loss_weight * L2_reproj`. Default weight `0.1` keeps the main MPJCE/MSE loss dominant. The synthetic pixels themselves do not carry ground-trheat labels, so they are only used to regularize the lifted pose against geometrically inconsistent virtual views.

## Evaluation metric

Primary: `MPJPE@k` for `k = 2, 3, 4` on the canonical H36M / MPI / 3DPW actual-mode protocol, plus full-view `val_MPJPE`. We track whether sparse-view accuracy improves while full-view accuracy stays within 1 mm of the v46 baseline.

## Expected MPJPE impact

On the local RTX 4090 v46 smoke baseline (`val_MPJPE ≈ 32.97 mm`), we expect virtual-view augmentation to improve the sparse-view regime most: `MPJPE@2` −3 to −5 mm, `MPJPE@3` −2 to −3 mm, and full-view `val_MPJPE` within ±1 mm. The gain should be larger for rare/dropped-view subsets because the model sees many more 2–3 view combinations during training.

## Main risk / mitigations

*Geometric inconsistency*: virtual cameras can place joints behind the body or create impossible viewing angles, producing noisy 2-D points. **Mitigation**: enforce a minimum depth test, mask occluded joints using a simple bounding-cylinder visibility check, and reject virtual cameras with any invalid projection. *Augmentation instability*: too many or too noisy synthetic views could dominate real data early in training. **Mitigation**: start with `v50_synth_num_virtual_views=1` and `v50_synth_keypoint_noise_px=1.0`, freeze the base model for the first epoch, and ramp the auxiliary loss weight linearly from 0 to `0.1` over the first 500 steps. *Compute overhead*: online projection adds per-sample work. **Mitigation**: implement as a NumPy/PyTorch vectorized transform in the loader; cache virtual camera matrices per epoch if needed.