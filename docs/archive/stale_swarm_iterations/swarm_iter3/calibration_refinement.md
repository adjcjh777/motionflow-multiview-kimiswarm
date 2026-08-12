# Calibration Refinement for Multi-View Human Pose Fusion

> Scope: investigate how camera calibration errors propagate through the current MotionFlow multi-view pipeline, survey practical refinement methods, and recommend a minimal next step toward CVPR/ICRA 2027 publication quality.

## 1. Problem statement

The MotionFlow multi-view branch currently assumes a **static, perfectly calibrated rig**. The `Camera` class stores a pinhole projection matrix `P = K[R | t]`, and every fusion plugin—DLT, robust triangulation, attention, residual/temporal refinement—triangulates or predicts 3D points under that fixed camera model. In practice this assumption is fragile:

1. **Calibration drift / initial calibration error.** Lens parameters from a one-time checkerboard capture or a downloaded dataset (Shelf/Campus) contain small but non-negligible errors in `K`, `R`, and `t`. These errors translate directly into biased 3D joint positions and large reprojection outliers.
2. **No distortion model.** The current `Camera` stores only `K, R, t` and assumes zero distortion. Wide-angle or consumer lenses introduce radial/tangential distortion that cannot be absorbed by a pinhole model.
3. **Dataset-specific calibration brittleness.** Design notes (`docs/design_v3.md`) report that learned plugins trained on Shelf fail on Campus, partly because the network memorizes scale/camera geometry rather than learning calibration-invariant fusion.
4. **Missing in-the-wild path.** There is no mechanism to refine or even estimate camera parameters when only raw synchronized videos are available.

**Calibration refinement** therefore means: *given noisy or unknown camera parameters and multi-view 2D keypoint observations, jointly estimate improved intrinsics/extrinsics (and optionally distortion) while preserving or improving the accuracy of the fused 3D human pose.*

## 2. Key related work and methods

### 2.1 Bundle adjustment (BA) — classical baseline

Bundle adjustment minimizes the reprojection error over camera parameters and 3D points:
