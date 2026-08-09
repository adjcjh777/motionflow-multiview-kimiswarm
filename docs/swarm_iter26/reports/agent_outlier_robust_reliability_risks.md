# v52 Outlier-Robust Reliability — Risk Register

**Module:** `outlier_robust_reliability_v52`
**Date:** 2026-08-09
**Status:** Design / pre-implementation review

---

## Risk 1: Gradient Instability Through M-Estimator Clipping

**Description:** The Tukey bisquare / Huber weights have near-zero gradient when residuals exceed the clipping threshold. During early training, an outlier view can be hard-clipped to zero, causing its reliability to collapse and never recover even if the view later becomes accurate.

**Impact:** Medium — can stall learning for some views, especially with variable view counts.

**Mitigation:**
- Use a **softened** M-estimator with a small floor on the weight (`w = max(w_c(x), 0.05)`).
- Make the clipping percentile a learnable or batch-statistic-driven value rather than a fixed hyperparameter.
- Detach the residual inputs (as v50 SEFH already does) so the robust weights do not back-propagate unstable gradients into the pose estimator.

---

## Risk 2: Identity Initialization Never Leaves Baseline

**Description:** The module is zero-initialized so that `r_v52 ≈ r_v51` and `σ_v52 ≈ σ_v51` at startup. If the auxiliary loss weight is too small or the pose estimator already saturates the metric, the new gates can stay frozen at identity and v52 adds no value.

**Impact:** Medium — the module becomes dead weight rather than an active improvement.

**Mitigation:**
- Initialize the reliability offset with a small positive bias (e.g. `+0.1`) so the model starts slightly away from identity.
- Use a curriculum that linearly increases the v52 auxiliary loss weight from 0.0 to `v52_orr_loss_weight` over the first epoch.
- Monitor the mean absolute reliability offset in tensorboard; if it stays below `0.01` for more than 500 steps, raise the loss weight.

---

## Risk 3: Physical-Cue Branch Overfits to H36M Floor Plane

**Description:** The floor-penetration signal is derived from the floor-plane assumption in v28 physical-space alignment. Datasets without a calibrated floor (e.g. some MPI-INF-3DHP sequences or in-the-wild 3DPW) can produce noisy or biased floor cues, causing the physical-cue encoder to learn H36M-specific patterns.

**Impact:** High — cross-dataset generalization may degrade.

**Mitigation:**
- Make floor-cue conditioning **optional** via `v52_orr_use_physical_cues` and disable it for 3DPW/unknown-floor domains.
- Normalize physical cues by the dominant bone length of the current skeleton so the signal is scale-invariant.
- Use the domain embedding (v48 / v51 CDSVR) to modulate physical-cue gain: `gain = sigmoid(W_domain · domain_emb)`.

---

## Risk 4: Added Latency From Cross-View Attention

**Description:** The `ReliabilityRefiner` uses cross-view attention over `V` views and `J` joints. With `T` frames, this is `O(T · V² · J²)`, which may become a bottleneck when `V=8` and `T=243` clips are used on A800.

**Impact:** Medium — training throughput drops and memory increases.

**Mitigation:**
- Default to **per-frame** operation; do not let the attention span time unless `v47` temporal aggregation is already enabled.
- Use low-rank attention or a small number of attention heads (`v52_orr_num_heads=4`, hidden=64) to keep FLOPs modest.
- Benchmark the forward pass with `torch.utils.benchmark` before committing to a full A800 run; if >10% slower, fall back to a lightweight MLP refiner.

---

## Risk 5: Interaction With v48 Domain Generalization Causes Domain-Specific Reliability Collapse

**Description:** v48 domain generalization already applies FiLM/conditional-BN and a GRL discriminator. Adding v52 domain-conditioned reliability rescaling could push the reliability distribution for a minority domain toward 0 or 1, amplifying domain shift instead of reducing it.

**Impact:** Medium — per-domain MPJPE may diverge, especially for rare domains.

**Mitigation:**
- Share the same domain embedding path used by v51 CDSVR instead of introducing a new domain encoder.
- Clamp the reliability offset to `[-2, 2]` so it cannot collapse to 0 regardless of domain.
- Report per-domain reliability histograms in smoke tests and abort if any domain mean reliability drops below `0.2`.

---

## Summary

| Risk | Probability | Severity | Owner | Mitigation Priority |
|---|---|---|---|---|
| Gradient instability from M-estimator | Medium | Medium | v52 implementer | High |
| Identity-init deadlock | Medium | Medium | v52 implementer | High |
| Physical-cue overfit to H36M floor | Medium | High | v52 + v48 owners | High |
| Cross-view attention latency | Medium | Medium | v52 implementer | Medium |
| v48 interaction / domain collapse | Low-Medium | Medium | v52 + v48 owners | Medium |

All mitigations should be verified in the RTX 4090 smoke before an A800 full run is queued.
