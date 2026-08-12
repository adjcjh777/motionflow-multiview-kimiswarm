# Camera-geometry embedding v4

## Task

Implement a normalized camera embedding (focal length / principal point / 6D rotation / scaled translation) and add it to a new `motionflow_mv/fusion/ray_attention_v4_model.py`.

## Delivered

* `motionflow_mv/fusion/ray_attention_v4_model.py`
  * New `RayAttentionFusionModelV4` keeps the v3 view/joint attention + differentiable weighted DLT pipeline.
  * Replaces the raw flattened `K/R/t` camera token with `_normalized_camera_embedding()`.
  * Camera token is 13-D:
    * focal length (fx, fy) normalized by mean focal length,
    * principal point (cx, cy) normalized by mean focal length,
    * 6-D rotation = first two columns of `R`,
    * translation scaled by the mean camera distance.
  * The normalized embedding is invariant to absolute scene scale/resolution, which should improve cross-dataset generalization.

## Design notes

* Focal length / principal point are normalized by the mean focal length because image dimensions are not part of the `Camera` dataclass. This gives dimensionless intrinsics consistent across rigs.
* 6-D rotation is the continuous representation from Zhou et al. (CVPR 2019), avoiding Euler/axis-angle discontinuities.
* Translation is divided by the mean camera distance per sample, making the embedding scale-invariant while preserving relative camera layout.
* The rest of the v3 architecture is untouched to keep the change minimal and ablatable.

## Verification

A small forward-shape test with a synthetic 4-view rig confirms the model runs and produces `(B, J, 3)` predictions and `(B, V, J)` weights.

## Next steps

1. Train `RayAttentionFusionModelV4` on the 62k-frame H36M subset and compare MPJPE against the v3 checkpoint.
2. Ablate the normalized embedding vs. raw `K/R/t` vs. no camera embedding.
3. If scale invariance proves useful, apply the same normalization to the ray/camera-center inputs as well.
