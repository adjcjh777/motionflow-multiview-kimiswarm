# v53 Probabilistic Pose Forecasting — Risk Report

This document lists the concrete risks for the proposed `v53_probabilistic_pose_forecasting` module and the mitigations that should be built into the smoke test and full-run queue.

## Risk 1: Over-smoothing of fast motion

**Description.** The uncertainty-gated smoothing residual `δ_t = γ · gate · (μ_{t+1} - X_t)` can suppress rapid joint accelerations (e.g., a tennis serve or a jump).  If the gate learns a value close to 1.0, the output becomes a heavily low-pass-filtered trajectory, raising MPJPE on dynamic actions.

**Evidence / likelihood.** Medium.  Existing `v47` temporal aggregation already showed a similar trade-off: it helped static/clipped sequences but slightly degraded the fastest Human3.6M actions (e.g., `Directions`, `Phoning`).

**Mitigations.**
- Initialize the scalar `γ = 0.0` and ramp the auxiliary loss weight linearly over `v53_ppfc_warmup_epochs`.
- Cap the gate output with a temperature-scaled sigmoid `sigmoid(g / 0.5)` so the effective gate rarely exceeds ~0.7.
- Add a smoke-test metric `MPJPE_dynamic` computed on the top-20% highest-velocity frames; reject the module if it degrades by >1.0 mm.

## Risk 2: Variance collapse and numerical instability

**Description.** The NLL loss strongly penalizes under-estimated variance.  With limited training data the network may push predicted `σ` toward `v53_ppfc_min_std`, causing exploding gradients in `log σ` and unstable training.

**Evidence / likelihood.** Medium-high.  Heteroscedastic variance heads in prior versions (`v33` uncertainty-aware triangulation, `v52` log-precision) required clamping and a warmup phase to avoid NaNs.

**Mitigations.**
- Use `softplus(log σ) + min_std` and clamp `log σ` to `[-2, 2]` inside the loss to keep variance in the 0.5–50 mm range.
- Initialize the final layer of `MLP_σ` to predict `log(min_std)` so the initial variance is large enough.
- Monitor the smoke-test tensorboard for `ppfc_std_mean`; stop if it drops below `0.003` within the first 200 steps.

## Risk 3: Redundancy or negative interaction with v47 / v49 temporal modules

**Description.** `v47_temporal_aggregation` and `v49_lite_temporal_aggregation` already fuse evidence across time.  Stacking `v53` on top may duplicate regularization, over-constrain the trajectory, and increase MPJPE.

**Evidence / likelihood.** Medium.  The `v49-Lite` smoke already targets causal temporal aggregation; adding a second causal head without ablation risks double-counting.

**Mitigations.**
- Treat `v53` as **exclusive** with `v47`/`v49` in the first smoke: add an guard in `omniview_fusion_v5.py` that raises `ValueError` if more than one temporal head is enabled simultaneously.
- Ablate four configs: baseline `v52`, `v52+v47`, `v52+v49`, and `v52+v53`.
- Only stack `v53` with `v47`/`v49` after single-head validation shows each improves independently.

## Risk 4: Memory and latency overhead from the temporal window

**Description.** The causal window `v53_ppfc_window` requires materializing a `(B, T, w, J, 3)` tensor and running an extra per-joint MLP.  On the largest A800 batch (`batch_size=16`, `T=27`, `J=17`, `V=4`), this adds ~1–2 MB of activation memory and a small forward latency.  Combined with `v52` and the self-evolution heads, it may push the GPU near the OOM limit.

**Evidence / likelihood.** Low-medium.  Previous stack `v50+v51+v52` runs near 30 GB on A800; another head increases the margin.

**Mitigations.**
- Default `v53_ppfc_window` to `3` for smoke and `5` for full runs; make it a YAML flag so it can be reduced without code changes.
- Use an in-place `unfold` with `nn.Unfold` or tensor slicing without copying the full window; profile peak memory with `torch.cuda.max_memory_allocated()` in the smoke.
- Queue the first A800 run with `batch_size=12` if the smoke memory delta is >5%.

## Risk 5: Identity-at-init warm start fails to engage during short smoke

**Description.** Because the final projection layers and gate are zero-initialized, the module starts as an identity mapping.  With only 1–2 smoke epochs, the NLL loss may not overcome the regularizers, leaving MPJPE unchanged and falsely suggesting the module is ineffective.

**Evidence / likelihood.** Low-medium.  `v52` UWT required ~1 epoch before its auxiliary loss produced a measurable gain; shorter tests looked flat.

**Mitigations.**
- Run the smoke for at least 2 epochs and report both epoch-1 and best-epoch MPJPE.
- Use a larger initial auxiliary weight (`v53_ppfc_loss_weight=2.0`) during the smoke to accelerate engagement, then lower it to `1.0` for the full A800 run.
- Add a diagnostic metric `ppfc_gate_mean`; if it remains <0.05 after epoch 1, increase the learning-rate multiplier for the new parameters.
