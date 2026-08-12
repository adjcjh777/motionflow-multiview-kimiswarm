# v52 Geometry-Aware Attention Pooling — Risk Report

## Risk 1: Warm-start regression when loading older checkpoints

**Description:** GAAP is intended to be identity-at-init so that a v45/v46/v47/v48/v50/v51
checkpoint remains usable. If the residual gate is not initialized to exactly
zero, or if the ray embedding MLP has a non-zero bias, the first forward pass of
a loaded checkpoint will shift features and degrade `val_MPJPE`.

**Mitigation:**
- Hard-code `v52_gaap_residual_gate_init = 0.0` in the constructor default.
- Initialize all MLP output layers to zero (using `.zero_()` on the final bias
  and small weights) and document this in the module.
- Smoke-test checkpoint loading: load the strongest existing checkpoint,
  enable GAAP, and verify that the first-batch MPJPE is within 0.1 mm of the
  baseline before any training step.

## Risk 2: Epipolar bias overpowers learned feature attention

**Description:** The geometry-aware attention score adds an epipolar Sampson
term. If `v52_gaap_epipolar_weight` is too large, the model may ignore views that
are geometrically distant but actually useful (e.g. views with strong 2D
confidence or complementary occlusion patterns).

**Mitigation:**
- Start with `v52_gaap_epipolar_weight = 0.0` and increase slowly during a
  dedicated ablation.
- Clamp the epipolar term to a bounded range (e.g. `[-5, 5]`) before adding it
  to the logits so that it can never dominate the scaled dot product.
- Include an ablation in the smoke test that compares `with` and `without`
  epipolar pooling; do not enable epipolar pooling if it degrades val_MPJPE.

## Risk 3: Variable and low view counts break the attention mask

**Description:** `OmniMultiViewFusionV5` already supports variable view counts
via `view_mask`. GAAP introduces pairwise view attention; if the mask is not
broadcast correctly to the attention logits, missing views will pollute the pooled
features or cause NaNs in softmax.

**Mitigation:**
- Reuse the existing `_build_view_attention_mask` helper (or its logic) to
  build the additive mask for GAAP.
- Add an assertion test with `V=2, 3, 4` and random masks to ensure the output
  shape is always `(B, T, V, J, d)` and that masked views contribute zero.
- Unit-test the module with all views masked out for one clip; the output must
  equal the input when `v52_gaap_residual_gate_init = 0.0`.

## Risk 4: Memory and latency increase from extra `V x V` attention

**Description:** GAAP computes `O(V^2)` pairwise attention for every joint and
timestep. With the existing ST transformer already processing `T * V` tokens,
adding GAAP for `J` joints can increase memory noticeably, especially on A800
full runs with large batch size or long clips.

**Mitigation:**
- Default to `v52_gaap_n_layers = 1` and `v52_gaap_n_heads = 4`; do not stack
  many layers unless the smoke test shows headroom.
- Only apply GAAP to a subset of joints if memory becomes an issue (e.g. limbs
  that benefit most from cross-view ray consistency), but keep the first
  version full-body for simplicity.
- Profile peak VRAM on a single RTX 4090 smoke batch before queuing the A800
  full run; if it exceeds the v51 baseline by more than 15 %, scale back.

## Risk 5: Geometry inputs are noisy or uncalibrated for some datasets

**Description:** WebBridge and other mixed datasets may contain approximate
intrinsics/extrinsics. GAAP’s ray embedding directly consumes `K, R, t`; if the
camera parameters are noisy, the geometry-aware attention will spread that
noise into the feature stream.

**Mitigation:**
- Make the ray embedding robust: normalize rays, detach the geometry branch from
the gradient by default (`stop_grad_geometry = True`), and only allow gradients
through the geometry term after the smoke test confirms stability.
- Add a small learned affine transform on the ray embedding so the model can
  down-weight inaccurate geometry if needed.
- Evaluate per-domain MPJPE in the smoke test; if WebBridge degrades while
  Human3.6M improves, make `v52_gaap_use_epipolar_pooling` domain-conditional or
  lower the epipolar weight for mixed training.
