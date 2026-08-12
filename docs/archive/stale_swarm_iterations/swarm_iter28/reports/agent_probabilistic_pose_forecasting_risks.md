# Agent v54 Risk Analysis: Probabilistic Pose Forecasting

**Module:** `probabilistic_pose_forecasting_v54`  
**Date:** 2026-08-09  
**Analyst:** design-swarm agent  
**Scope:** `motionflow_mv/fusion/probabilistic_pose_forecasting_v54.py`, `OmniMultiViewFusionV5`

## Executive summary

v54 introduces a causal probabilistic forecasting head on top of the v52/v53 calibrated poses. It is intentionally lightweight and identity-initialized, so the main risks are not catastrophic failure but **temporal leakage**, **over-smoothing**, and **loss-of-balance** with existing auxiliary losses. Below are five concrete risks with mitigations that can be actioned during implementation.

## Risk 1: Temporal leakage / non-causal lookahead

**Description:** If the temporal encoder uses symmetric padding or full self-attention, the forecast for frame `t+1` can peek at frame `t+1` or later. This would produce an unrealistic MPJPE improvement at training time that does not transfer to online inference.

**Impact:** Smoke and full-run metrics look good, but real-time / causal evaluation degrades; the module is not usable in the final pipeline.

**Mitigation:**

1. Enforce a strictly causal architecture: causal Conv1D with `(kernel_size - 1)` left padding only, or masked causal self-attention with `mask[i, j] = (j <= i)`.
2. Add a unit test that constructs a sequence where `X_{t+1}` is radically different from earlier frames and verifies that the forecast at time `t` does not depend on `X_{t+1}`.
3. Mark the module explicitly as "causal" in docstrings and smoke-test scripts; never allow future padding.

## Risk 2: Over-smoothing / motion blur

**Description:** The correction MLP can learn to pull every frame toward a temporally averaged pose, erasing fast but real motion (e.g., rapid hand movements, jumps). The `L_forecast` term may dominate and penalize legitimate high-velocity changes.

**Impact:** MPJPE on slow sequences improves while fast-motion sequences degrade; overall paper metrics become biased.

**Mitigation:**

1. Keep the correction gate small at initialization (`residual_gate_init=-6.0`) and only train it after a `warmup_epochs` period.
2. Use a small `λ_cons` (default `0.01`) to regularize the correction magnitude.
3. Add an ablation on a high-motion subset (e.g., jumping / running actions) and reject the module if MPJPE on that subset rises by more than 1 mm.

## Risk 3: Identity-at-init is not strict

**Description:** Even though the correction path is gated near zero, BatchNorm, LayerNorm, or non-zero biases inside the forecast MLP can shift the output before any training step. This breaks the warm-start guarantee from v52/v53 checkpoints.

**Impact:** The first smoke validation after enabling v54 shows a small but measurable MPJPE drift (e.g., 0.3–0.8 mm), making it hard to tell whether v54 is helping.

**Mitigation:**

1. Avoid normalization layers on the output side; use them only inside the MLP trunk if at all.
2. Add a `tests/test_v54_identity_at_init.py` test that asserts `||pred_3d_ref - pred_3d_init||_2 < 1e-4` before the optimizer step, using the default flag values.
3. Provide a `v54_ppf_identity_init` flag that zero-initializes every learnable layer and disables all non-linearities in the forecast path at init.

## Risk 4: Imbalance with v50 / v52 / v53 auxiliary losses

**Description:** v50 (SEFH), v52 (UWT), v53 (PSC), and v54 (PPF) all add auxiliary losses. If `v54_ppf_loss_weight` is too large, the gradient from `L_ppf` can drown the geometry losses, especially during early epochs when the forecast head is still random.

**Impact:** Triangulation quality degrades because the model optimizes temporal smoothness at the expense of reprojection / physical-space terms.

**Mitigation:**

1. Default `v54_ppf_loss_weight` to `0.01` and enable it only after `v54_ppf_warmup_epochs >= 1`.
2. If `v54_ppf_detach_input=True` (default), the new loss only trains the forecasting head, preserving the v52/v53 gradient paths.
3. Run a smoke ablation sweeping `v54_ppf_loss_weight in [0.001, 0.01, 0.1]` and pick the value that does not change v53-only MPJPE by more than 0.1 mm.

## Risk 5: Sequence-length and memory overhead

**Description:** v54 operates on the full temporal sequence `(B, T, J, 3)`. For long clips (`T=243`) and full `d=64`, an inefficient temporal encoder could add noticeable memory and compute overhead, especially on the RTX 4090 smoke runs.

**Impact:** Smoke OOM or reduced throughput; A800 runs need a smaller batch size.

**Mitigation:**

1. Use depthwise-separable causal Conv1D (not a full transformer) so complexity is `O(T * J * hidden^2)` and memory is `O(T * J * hidden)`.
2. Default `v54_ppf_hidden=32` for smoke configs and `64` for full A800 runs.
3. Profile the module with `torch.cuda.memory_summary()` on a single forward pass before adding it to the training queue; reject if overhead exceeds 5% of the v53 baseline.

## Recommendation

Proceed with implementation. Prioritize Risks 1 and 4 in the smoke phase, then verify Risk 2 on a high-motion validation subset before committing to a full A800 run.
