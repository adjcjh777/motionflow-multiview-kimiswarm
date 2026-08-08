# v31 variant: `camera_view_embedding_upgrade`

## Problem statement

`OmniMultiViewFusionV5` currently conditions each view with a two-layer MLP that
flattens `K`, `R`, and `t` into a 21-D vector. This discards the explicit
geometric structure of the multi-view rig:

* Intrinsics and extrinsics are concatenated before any non-linearity, so focal
  length, principal point, camera center, and optical axis are not given
  separate, meaningful channels.
* Pairwise relationships between cameras—baseline length, relative rotation, and
  optical-axis alignment—are never exposed directly to the model. These are
  exactly the cues that the downstream epipolar-biased transformer and v25
  geometry-fusion modules reason about.
* The embedding is only used as a residual additive bias before the
  spatio-temporal transformer. It therefore helps little when view order or
  camera count varies, which is increasingly important for variable-view
  training and WebBridge-style mixed datasets.

This may contribute to the rapid overfitting seen in v29a and to the
brittleness of variable-view subsets.

## Concrete proposed change

Add a richer, geometry-aware camera view embedding
(`CameraConditionedViewEmbeddingV31` in
`motionflow_mv/fusion/camera_view_embedding_v31.py`) and swap it in for the
existing `CameraConditionedViewEmbedding`.

The new embedding has two branches:

1. **Local camera descriptor.** Encode structured quantities derived from `K`
   and `(R, t)`: normalized focal lengths/principal point, camera center `C`,
   world-space optical axis, and the vector/distance from `C` to the scene
   centroid. These are fed through a small MLP with LayerNorm.
2. **Pairwise view-geometry descriptor.** For every view pair compute baseline
   length, relative rotation angle, and optical-axis cosine similarity. A small
   MLP lifts these to a pairwise feature space, and a lightweight
   self-attention aggregates the pairwise context for each view. This gives the
   model direct, permutation-equivariant access to multi-view rig geometry.

The two branches are concatenated and projected back to dimension `d`. The
output projection is zero-initialised, so the block is identity at init and the
model falls back to the existing learned `view_pos_embed`.

Integration is a single one-line swap in `OmniMultiViewFusionV5`: use the v31
class when `use_camera_view_embedding` is enabled. No other source files need
changes for this ablation.

## Expected impact

* **val_MPJPE:** A modest but consistent improvement (1–3 mm) on H36M and MPI
  mixed validation, because richer geometry conditioning should reduce the
  burden on the spatio-temporal transformer and let v30’s hierarchical encoder
  operate on more meaningful per-view tokens.
* **Overfitting:** Slower overfitting after epoch 1 compared with v29a. The
  embedding encodes hard geometric invariances (view permutation, camera
  distances, angles) rather than memorising dataset-specific camera indices.
* **Variable-view robustness:** Better generalisation to random view subsets and
  permutations, since the embedding is strictly camera-conditioned and
  permutation-equivariant.

## Main risk

The pairwise attention is `O(V²)` in memory. With `V = 14` (MPI-padded mixed
loader) this is small, but it can become noticeable for large batch sizes or
when running many concurrent A800-D jobs. The second risk is calibration
sensitivity: if camera perturbations are large, encoding baseline/angle
explicitly can amplify noise. The existing camera-augmentation curriculum and
`v30` stochastic depth should mitigate this, but the first smoke test should
monitor training stability and gradient norms.

## Files created for this variant

* `motionflow_mv/fusion/camera_view_embedding_v31.py` – implementation.
* `experiments/train_omniview_fusion_v5_webbridge_v31_camera_view_embedding_upgrade.py`
  – monkey-patch wrapper that dispatches to the standard v5 trainer.
* `scripts/launch_v31_camera_view_embedding_upgrade_local4090.sh` – local RTX
  4090 smoke launcher.
* `configs/v31_camera_view_embedding_upgrade.yaml` – flag record / configuration
  snapshot.
