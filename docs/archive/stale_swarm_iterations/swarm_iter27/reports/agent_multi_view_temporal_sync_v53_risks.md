# v53 Multi-View Temporal Synchronization — Risk Report

## 1. Temporal warping breaks causality and motion blur

**Risk:** Predicted fractional offsets require interpolating the 3-D pose sequence in time. If offsets are large, the interpolation blends poses from different true frames and can blur fast motions, hurting rather than helping distal joints.

**Mitigation:**
- Bound offsets with a small `max_shift` (default 2.0 frames) and a `tanh` output.
- Provide a causal-only mode (`v53_mvts_causal_warp`) that warps using only past frames for streaming inference.
- Add an offset-magnitude regularizer (`v53_mvts_offset_reg_weight`) to keep offsets near zero unless the data forces a shift.

## 2. v52 uncertainty weights may be noisy early in training

**Risk:** The v53 fusion step relies on v52 precision weights `w^U`. During the first epochs v52 weights may still be near-uniform or noisy, so the sync fusion can down-weight the wrong views and amplify misalignment instead of correcting it.

**Mitigation:**
- Use a loss-weight warmup: set `v53_mvts_sync_loss_weight = 0` until `epoch >= v52_uwt_warmup_epochs`.
- Fallback to uniform view weighting (`v53_mvts_use_uwt_weights=false`) in early epochs.
- Identity-initialize the residual gate so the module is effectively bypassed until v52 has stabilized.

## 3. Interaction with v47/v49 temporal aggregation

**Risk:** v53 warps and smooths the 3-D pose sequence before v47/v49 temporal aggregation runs. Stacking both modules may over-smooth trajectories and degrade fast-motion accuracy.

**Mitigation:**
- Default v53 to run only when v47 and v49-Lite are disabled, or explicitly tune the combination.
- Add an ablation config that tests v53 alone, v47 alone, and v53+v47.
- Monitor temporal-jerk metrics (e.g., 2nd derivative of MPJPE over time) in smoke to detect over-smoothing.

## 4. Identity-at-init can hide integration bugs

**Risk:** With the residual gate initialized to zero, the module returns the v52 output unchanged at startup. A wiring error (e.g., passing the wrong tensor to the warper) may therefore go undetected until training fails to improve.

**Mitigation:**
- Add a unit test in `tests/test_multi_view_temporal_sync_v53.py` that disables identity init, runs a forward pass, and checks that gradients flow and `X^out != X^U`.
- Verify in smoke that after a few training steps the predicted offsets have non-zero variance and `L_sync` decreases.
- Compare against a hand-computed two-view toy example with a known one-frame offset.

## 5. Differentiable temporal interpolation is memory-heavy

**Risk:** Warping the full `(B, T, J, 3)` tensor for each view stores intermediate indices and gradients for every frame, which can OOM on the RTX 4090 smoke when `T` is large (e.g., `clip_len=27`).

**Mitigation:**
- Implement the warp with `grid_sample` or `F.interpolate` on a compact `(B, V, J, T)` tensor, then permute, rather than materialising a `(B, T, V, J, 3)` warped tensor.
- Cap `v53_mvts_max_shift` to a small integer and pre-compute interpolation weights.
- Benchmark peak GPU memory in smoke and, if needed, restrict the module to a local temporal window around each frame.
