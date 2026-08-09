# Agent v54 Risk Analysis: Adaptive Low-Rank Per-Domain Fusion

**Module:** `adaptive_lr_per_domain_v54`  
**Date:** 2026-08-09  
**Analyst:** design-swarm agent  
**Scope:** `motionflow_mv/fusion/adaptive_lr_per_domain_v54.py`, `OmniMultiViewFusionV5` ST-transformer output, auxiliary loss integration

## Executive summary

v54 ALR is a model-side low-rank adapter bank that refines fused multi-view tokens per domain. The main risks are not raw capacity but *interaction with existing domain mechanisms*, *mixed-batch routing ambiguity*, *unseen-domain behavior*, and *low-rank under-fitting*. Below are five concrete risks with actionable mitigations.

## Risk 1: v54 adapters conflict with v48 domain conditioning

**Description:** v48 already applies domain-conditional FiLM/GRL to the feature tokens. Adding another per-domain residual on top of v48 may create redundant or opposing domain-specific transformations, causing instability or cancellation.

**Impact:** Full-stack training diverges or underperforms compared with v48-only or v54-only; tuning the two mechanisms jointly becomes difficult.

**Mitigation:**

1. Keep v54 adapters **identity-at-init** so the v48 baseline is preserved until v54 has learned.
2. Start ablations with `v54_alr_loss_weight = 0.0` and `v54_alr_warmup_epochs = 1` to isolate the adapter effect from the auxiliary loss.
3. Log per-domain adapter gradient norms alongside v48 FiLM gradients; if they are strongly anti-correlated, reduce `v54_alr_rank` or disable v48 FiLM in the v54 experiment.

## Risk 2: Mixed batches make per-domain routing ambiguous

**Description:** `webbridge_mixed_dataset.py` may produce batches containing samples from multiple domains. v54 assumes a single integer `domain_id` per sample. If a batch mixes domains, routing each sample correctly is straightforward, but aggregating per-domain statistics (e.g. for the auxiliary loss) and maintaining per-domain EMAs becomes fragile.

**Impact:** Under-represented domains in a mixed batch receive inconsistent adapter gradients; the auxiliary loss is computed on a domain-mixed residual and sends conflicting signals.

**Mitigation:**

1. Route at the per-sample level using `torch.index_select` over the batch dimension; do not assume a single domain per batch.
2. Compute the auxiliary loss only when `domain_id` is uniform across the batch, or weight the loss by per-sample domain membership.
3. Log the fraction of mixed-batch steps and assert it is below a tunable threshold in smoke tests.

## Risk 3: Unseen domains fall back poorly

**Description:** At inference, a domain ID may not have a dedicated adapter (e.g. a new capture setup). If `v54_alr_share_unseen=True`, the shared adapter must generalize; if it is under-trained, the model may regress relative to the v53 baseline.

**Impact:** Cross-domain evaluation shows a sudden MPJPE jump on unseen domain IDs; the module hurts generalization.

**Mitigation:**

1. Train the shared adapter on all domains jointly, not only on leftovers.
2. Optionally add a soft-mixture fallback that predicts adapter weights from the mean pooled token using a tiny domain-classifier-style routing network.
3. In smoke tests, evaluate with a held-out domain label and require `Δ MPJPE ≤ 0.5 mm` compared with v53.

## Risk 4: Low-rank adapters are too small for complex domain shifts

**Description:** With `v54_alr_rank=8`, each domain adapter has only ~1k effective parameters. If a domain shift requires a large change in the fusion manifold, the adapter may underfit and the gate will stay near zero, yielding no improvement.

**Impact:** v54 has no measurable impact on MPJPE; the engineering cost is wasted.

**Mitigation:**

1. Provide a smoke-to-medium sweep over `v54_alr_rank ∈ {4, 8, 16, 32}`.
2. Monitor the mean absolute value of `ΔF` after warm-up; if it remains below 1e-3, increase rank or remove the gate's negative bias.
3. Allow adapters to operate on a subset of feature channels (e.g. the first `d/2`) to concentrate capacity.

## Risk 5: Gate collapse or over-activation

**Description:** The uncertainty/physics gate `γ` may collapse to near-zero for all tokens (module does nothing) or saturate near-one for all tokens (module ignores v52/v53 signals). Both extremes remove the intended conditional refinement.

**Impact:** No gain in the zero case; potential instability or regression in the saturated case.

**Mitigation:**

1. Initialize the gate MLP final bias to `v54_alr_gate_init=-6.0` for a soft warm start.
2. Add a small entropy bonus on `γ` during the first epoch to encourage it to spread.
3. Log histograms of `γ` per domain; if the inter-quartile range is below 0.05 after warm-up, halve `v54_alr_loss_weight` and re-run smoke.

## Recommendation

Proceed with implementation, but prioritize Risk 1 (v48 interaction) and Risk 3 (unseen domain fallback) in the smoke phase. A clean warm-start identity check and a held-out-domain test are prerequisites for any A800 full run.
