# v31: Multiview Temporal Perceiver

## Problem

v29a (hierarchical-only multi-scale view encoder) achieves a strong first epoch
(28.12 mm) but overfits sharply by epoch 3 (81.08 mm). v30 hardened the encoder
with stochastic depth, gated residual paths, and cross-scale fusion, but it
still processes each frame largely independently after view fusion. Meanwhile,
v19 introduced a Temporal Perceiver that refines per-frame 3D poses using both
pose and pooled spatiotemporal features, yet it was never systematically
combined with the v30 hardened hierarchical encoder. The result is a gap: strong
per-frame multiview fusion exists, and a separate temporal refinement module
exists, but they have not been validated together as a single variant. v31
closes this by stacking the v19 temporal perceiver on top of v30's hardened
hierarchical multiview encoder, with the v29 physical-space temporal loss in
warmup mode.

## Proposed Change

Create a new training variant `v31_multiview_temporal_perceiver` that enables
three existing flags together:

- `--use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1`
- `--use_temporal_perceiver_v19`
- `--use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001 --v29_physical_loss_warmup_epochs 1`

The rest of the base stack (v25 geometry fusion, v18 deformable cross-view
attention, variable view training, camera view embedding, set view aggregator,
and the standard auxiliary losses) stays identical to v30a. No new source files
are introduced. The temporal perceiver receives the concatenation of the
per-frame 3D pose `(B,T,J,3)` and the view-pooled ST feature `(B,T,J,d)`,
compresses the clip to 32 latent tokens, and decodes per-frame pose residuals
that are added to the baseline pose. TTE is explicitly disabled.

## Expected Impact

- **val_MPJPE:** small improvement over v30a (target -2 to -5 mm) by exploiting
  temporal continuity in the final pose sequence.
- **Overfitting:** lower than v29a. v30's stochastic depth and gated residual
  keep the multiview path regularized, and the perceiver's latent bottleneck
  (32 latents for the whole clip) limits capacity in the temporal path.
- **Physical plausibility:** the v29 physical loss with one-epoch warmup should
  encourage foot-floor contact and smooth bone lengths without destabilizing
  early training.

## Main Risk

The v19 temporal perceiver was validated in isolation but not with v30's
hierarchical encoder. The feature/pose concatenation may magnify any drift in
the view-pooled ST features, especially when variable-view training drops
views. If the perceiver over-smooths fast motions or underfits, val_MPJPE
could stagnate or rise after epoch 1, similar to v29a. A short smoke test on
RTX 4090 should confirm that the module trains without NaNs and that the first
validation lands below ~35 mm before committing A800-D resources.
