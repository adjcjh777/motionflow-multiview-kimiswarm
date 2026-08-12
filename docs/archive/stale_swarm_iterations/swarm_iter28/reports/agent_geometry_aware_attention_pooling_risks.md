# v54 Geometry-Aware Attention Pooling — Risk Report

Module: `geometry_aware_attention_pooling_v54`  
Base: `v53 Physical-Space Calibration`  
Date: 2026-08-09

---

## Risk 1: Attention weights collapse to a single view or become uniform

**Description:** The geometry bias `e_geom(i, k)` in Eq. (1) is unbounded in magnitude if the ray-sigma `v54_gaap_ray_sigma` is too small or if the 2-D detections are noisy. A very large positive bias can make the softmax concentrate all attention on one view; a near-zero sigma can make it ignore geometry entirely.

**Impact:** Re-triangulation by v52 becomes effectively single-view, raising `MPJPE` by 5–20 mm on occluded joints.

**Mitigation:**
- Clamp `e_geom` to `[-5, 5]` before adding it to the content scores.
- Initialise `v54_gaap_geometry_bias_weight = 0.0` and learn it through a softplus gate so it warms up from 0 to 1 over the first epoch.
- Add a per-head attention entropy loss (`v54_gaap_loss_weight * H(A)`) with weight 0.001, reusing the existing `attention_entropy_weight` infrastructure.
- Monitor the max/entropy of `geometry_attention_map` in smoke tests; abort if entropy drops below 0.1 nats before epoch 1 ends.

---

## Risk 2: O(N²) ray-pair computation for variable views

**Description:** Computing the closest-approach distance for every pair of rays costs `O(V²)` per `(B, T, J)`. With v46/v51 variable views (up to `V=8`) and long clips, this is non-trivial and may dominate the forward pass on the local RTX 4090.

**Impact:** Training throughput drops by 20–40 %; smoke test may exceed the 15-minute budget.

**Mitigation:**
- Cache ray directions and camera centers once per forward pass; they are constant across joints.
- Use the cheap approximation `d_ik  || (C_i - C_k) × r_k || / ||r_k||` evaluated at `pred_3d_init` instead of solving the full skew-line problem when `v54_gaap_pool_type == "cross_view_joint"`.
- Gate the computation with `view_mask` and early-out when a view is masked, keeping complexity `O(V_visible²)`.
- Benchmark the forward pass in the smoke script; if a single step takes > 250 ms, fall back to the cheaper `mean` pooling mode.

---

## Risk 3: Identity-at-init property breaks due to LayerNorm or residual gates

**Description:** Although the attention output projection is zero-initialised, the residual path passes through `LayerNorm` and a learned gate `γ`. If any of these are initialised non-identity, the module output at step 0 is not exactly equal to the input, breaking the warm-start guarantee from v52/v53.

**Impact:** Loading a v53 checkpoint with v54 enabled changes `val_MPJPE` by > 0.1 mm, causing regression or forcing a full re-train from scratch.

**Mitigation:**
- Use `nn.Identity()` for the post-attention normalisation when `v54_gaap_identity_init=True` at initialisation, or use `LayerNorm` with zero-initialised affine parameters.
- Set `v54_gaap_residual_gate_init = 0.0` so the residual branch is zero-weighted.
- Add a unit test `test_v54_identity_at_init.py` that asserts `allclose(pooled_features, features, atol=1e-5)` before any training step.
- In the smoke script, compare the first validation against the v53 baseline; reject if `ΔMPJPE > 0.1 mm`.

---

## Risk 4: Gradient instability from the ray-similarity bias

**Description:** The ray-similarity term contains `1 / b` or `log(b)` where `b` depends on inverse camera intrinsics and normalised ray directions. Back-propagating through `inv(K)` or through `pred_3d_init` can produce exploding gradients, especially when the camera matrix is close to singular or the 2-D keypoint lies exactly on a principal point.

**Impact:** Training diverges within the first 200 steps with `loss = NaN/Inf`.

**Mitigation:**
- Stop gradient from `pred_3d_init` to the geometry bias: compute the bias with `pred_3d_init.detach()`; only the attention output is back-propagated through the content branch.
- Use `torch.linalg.pinv(K)` with `rcond=1e-3` instead of a hard inverse.
- Clip `v54_gaap_geometry_bias_weight` to `[-2.0, 2.0]` via `torch.clamp` in the forward.
- Apply global gradient clipping (`max_grad_norm=1.0`) already present in the trainer.

---

## Risk 5: Negative interaction with v46 sparse-view dropout and v51 cross-domain reliability

**Description:** v46 drops random views during training and v51 predicts per-domain reliability. The v54 attention may learn to depend on a view that is suddenly missing during evaluation, or it may override the v51 reliability weights by re-normalising attention over a different set of views.

**Impact:** Sparse-view evaluation `MPJPE@2` or `MPJPE@3` increases relative to v53; the module overfits to the full 4-view training distribution.

**Mitigation:**
- Apply the same `view_mask` used by v46/v51 inside the softmax so dropped views are excluded from both keys and queries.
- Add a compatibility flag `v54_gaap_use_uwt_weights` that injects v52 precision weights as an additive bias to the attention scores, ensuring the module respects the v52/v51 uncertainty estimates.
- Evaluate on `MPJPE@2/3/4` in the smoke script, not only the full-view metric, and gate the A800 queue behind a `< 2 %` regression on any sparse-view metric.
