# Agent v53 Risk Analysis: Adaptive Learning Rate Per Domain with v52 Uncertainty Feedback

**Module:** `adaptive_lr_per_domain_v53`  
**Date:** 2026-08-09  
**Analyst:** design-swarm agent  
**Scope:** `motionflow_mv/training/adaptive_lr_per_domain_v53.py`, trainer `experiments/train_omniview_fusion_v5_webbridge_multi.py`, `OmniMultiViewFusionV5` auxiliary outputs

## Executive summary

v53 is a trainer-side learning-rate scaler that uses v52 uncertainty/precision as a per-domain difficulty signal. Because it does not change the model architecture, the main risks are not model-capacity risks but *optimization stability*, *interaction with existing domain-loss mechanisms*, and *reliance on v52 statistics*. Below are five concrete risks with actionable mitigations.

## Risk 1: α_d oscillates and destabilizes training

**Description:** The per-domain LR scale `α_d` depends on the EMA of v52 uncertainty and on per-domain gradient norms. With small mixed batches, the gradient-norm estimate is noisy, and the exponential term can swing `α_d` between `1/γ` and `γ` from step to step. This can amplify noisy gradients instead of smoothing them.

**Impact:** Loss spikes, divergence, or slow convergence; smoke tests fail or show large MPJPE variance.

**Mitigation:**

1. Require `v53_dulr_min_samples` steps before any scaling is applied.
2. Use a long EMA (`beta ≥ 0.99`) for both uncertainty and gradient norms.
3. Clamp `α_d ∈ [1/γ, γ]` with a conservative `γ` (default 2.0, smoke 1.5).
4. Optionally smooth `α_d` with a per-domain momentum buffer: `α_d_smooth = 0.9 α_d_prev + 0.1 α_d`.

## Risk 2: Mixed-batch domain labels are ambiguous

**Description:** Mixed batches in `webbridge_mixed_dataset.py` may contain samples from multiple domains. v53 assumes a single domain label per batch to scale the optimizer step. If the batch contains a mix, the chosen `d_batch` (e.g. majority label) may not match all samples, causing some domains to be over- or under-scaled.

**Impact:** Domain-specific metrics drift; under-represented domains get the wrong LR.

**Mitigation:**

1. Default to scaling only when all samples in the batch share the same domain label; otherwise set `α_d = 1` for that step.
2. If mixed batches are required, compute per-sample `α_d` and apply a weighted step proportional to sample count per domain.
3. Log the fraction of mixed-batch steps where scaling is skipped.

## Risk 3: v52 uncertainty statistics are themselves noisy early in training

**Description:** v52 precision weights are random immediately after initialization and only become meaningful after several hundred steps. If v53 activates too early, it will scale the LR based on noise, amplifying the wrong domains.

**Impact:** Early training diverges or becomes biased toward domains that happen to have higher initial v52 weights.

**Mitigation:**

1. Set `v53_dulr_warmup_steps` to at least the first full epoch of training.
2. Verify that v52 weights have stabilized by checking that the mean per-domain weight entropy is below a threshold before enabling scaling.
3. Provide a `strict_warmup` mode that keeps `α_d = 1` until the v52 auxiliary loss has decreased for two consecutive epochs.

## Risk 4: Double interaction with v41 weighted domain loss and v48 domain conditioning

**Description:** v41 already reweights the MSE per domain, and v48 applies domain-conditional FiLM/GRL. v53 adds a third domain-specific mechanism. If a domain is down-weighted by v41 and simultaneously receives a large LR from v53, the net effect is unclear and may over-correct.

**Impact:** Domain metrics improve in isolation but the full-stack evaluation regresses; tuning becomes a three-way trade-off.

**Mitigation:**

1. Run ablations on v41-only, v48-only, v53-only, and all combinations to isolate effects.
2. Couple v53 with the existing v41 domain weights: use `effective_α_d = α_d / (v41_domain_weight_d + ε)` to avoid compounding.
3. Log per-domain loss, gradient norm, and `α_d` together so interactions are visible.

## Risk 5: Added trainer complexity and reproducibility

**Description:** v53 requires the trainer to collect v52 outputs, maintain per-domain EMA state, and modify the optimizer step. This adds state that must be saved/restored with checkpoints and complicates reproducibility if experiments are resumed from a checkpoint taken before v53 was enabled.

**Impact:** Checkpoint bloat, resume bugs, or silently different behaviour on restart.

**Mitigation:**

1. Store the `AdaptiveLRPerDomainV53` state dict in the trainer checkpoint under `"adaptive_lr_v53"`.
2. On resume, load the state dict only if `use_v53_adaptive_lr_per_domain` is True; otherwise ignore it.
3. Add a smoke test that resumes a checkpoint and asserts `α_d` continuity.

## Recommendation

Proceed with implementation, but prioritize Risk 1 (oscillation) and Risk 3 (v52 warmup) in the smoke phase. A stable v53 is a prerequisite for any cross-domain A800 run that builds on v52.
