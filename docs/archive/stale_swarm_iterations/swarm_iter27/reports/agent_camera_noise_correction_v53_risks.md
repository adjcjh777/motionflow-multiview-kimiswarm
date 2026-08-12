# v53 Camera Noise Correction — Risk Register

## Risk 1: Module overfits to training-set camera-specific biases

**Likelihood:** Medium  
**Impact:** High  

Because the correction is per-view (or even per-camera-cluster), it can learn a dataset-specific shortcut (e.g., "Camera 3 is always 3 px to the left") rather than a generalizable noise model. This would hurt cross-dataset transfer to 3DPW actual mode and in-the-wild WebBridge sequences.

**Mitigations:**
- Keep the correction magnitude small via `v53_cnc_max_shift_px` (default 32 px) and an L1/L2 penalty in `cnc_loss`.
- Add a random-view permutation/data augmentation branch during training so the module cannot simply memorize view indices.
- Evaluate on a held-out camera-split validation set before trusting the metric.

## Risk 2: Unstable ray correction near degenerate camera configurations

**Likelihood:** Medium  
**Impact:** Medium  

Applying an affine transform to a normalized ray and reprojecting can amplify errors when the keypoint is near the image border, when focal length is very small, or when views have narrow baselines. The corrected rays may diverge and produce triangulation outliers.

**Mitigations:**
- Apply the correction in normalized image coordinates (divided by focal length), not raw pixels, to make the transform dimensionless.
- Clamp the resulting shift to `v53_cnc_max_shift_px` in pixel space after reprojection.
- Add an identity skip connection (already in the design) so that unstable gradients can be bypassed at init and early epochs.

## Risk 3: Double counting with v52 uncertainty-weighted triangulation

**Likelihood:** Medium  
**Impact:** Medium  

v52 already learns per-view/joint precision weights. If v53 also learns to down-weight noisy views by correcting them away, the two modules can fight each other: v52 might increase uncertainty on a view that v53 has already corrected, leading to collapsed weights or slower convergence.

**Mitigations:**
- Use the proposed `reprojection-change` auxiliary loss so v53 is rewarded for reducing the pre-v52 residual, not for post-v52 metrics.
- Initialize v53 as identity and use `v53_cnc_warmup_epochs` > 0 so v52 has already settled before v53 is allowed to change observations.
- Monitor the correlation between v52 weights and v53 correction magnitudes; add a decorrelation penalty if necessary.

## Risk 4: Extra compute and memory at high resolution

**Likelihood:** Low  
**Impact:** Low  

The per-view MLP is tiny compared to the transformer stack, but the batched matrix-vector products for ray correction and reprojection add a small overhead. On full A800 runs with `clip_len=13` and `batch_size=16`, this could add ~1-2% wall time.

**Mitigations:**
- Fuse the per-view affine computation across joints with a single einsum.
- Cache `K^{-1}` when `K` is shared across the batch/time dimension.
- Gate the correction with `use_v53_camera_noise_correction` so the overhead is zero when disabled.

## Risk 5: Domain shift if intrinsics differ between train and test

**Likelihood:** Medium  
**Impact:** High  

The affine is defined in normalized image coordinates, but the meaning of "normalized" depends on `K`. If the test camera has very different focal lengths or principal points, a correction learned on H36M/WebBridge may transfer poorly.

**Mitigations:**
- Always represent the correction relative to each view's own `K` (done by design).
- Make the correction head additionally condition on `K` and `R` through a small camera embedding (reuse existing `CameraConditionedViewEmbedding` if available) so the module is aware of the camera geometry.
- Add domain-balanced sampling and, if possible, synthetic calibration-noise augmentation during training.
