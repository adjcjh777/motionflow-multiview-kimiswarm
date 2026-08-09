# v53 Outlier-Robust Reliability — Risk Register

**Module:** `outlier_robust_reliability_v53`  
**Date:** 2026-08-09  
**Status:** Design / pre-implementation review  

---

## Risk 1: Robust-Kernel Scale Collapses to Near Zero

**Description:** The learned robust scale `σ = Softplus(σ_log)` controls how quickly weights decay for large residuals. If `σ_log` drifts strongly negative, the kernel becomes a hard threshold and gradients vanish for outlier views; if it drifts strongly positive, all views receive nearly uniform weights and the module stops refining v52.

**Impact:** Medium — training instability or loss of outlier rejection.

**Mitigation:**
- Clamp `σ` to `[0.1, 10.0]` so it cannot collapse or explode.
- Initialize `σ_log` so the initial scale is comparable to the median residual observed in the first batch.
- Detach residual inputs (following v50 SEFH) so only the reliability factor is trained from robust gradients, not the pose estimator.

---

## Risk 2: v53 Overwrites Useful v52 Weights

**Description:** Because v53 refines v52 weights multiplicatively, an overly aggressive reliability factor `γ` can nullify the carefully learned v52 weight distribution, especially if `v53_orr_loss_weight` is too high or the robust loss dominates.

**Impact:** Medium — regression on already-clean multi-view sequences.

**Mitigation:**
- Keep the pose-correction gate `g` initialized to `-6.0` (sigmoid  0.002) so the final pose stays near `X_v52` at startup.
- Start with `v53_orr_loss_weight = 0.005` and scale up only after smoke shows no regression.
- Log the mean and std of `log(γ)`; abort training if the absolute mean exceeds 1.0 before the first validation.

---

## Risk 3: Physical-Space Residuals Are Noisy on Domains Without Calibrated Floor

**Description:** The physical residual `e_phys` relies on floor and bone-length signals from v28/v40 physical-space alignment. Datasets without a reliable floor (e.g., 3DPW, in-the-wild sequences) may inject misleading residuals that v53 interprets as outliers, harming cross-domain generalization.

**Impact:** High — per-domain MPJPE may diverge.

**Mitigation:**
- Make physical-cue usage optional via `v53_orr_use_physical` and disable it for domains lacking calibrated floor.
- Normalize `e_phys` by the current skeleton’s median bone length so the signal is scale-invariant.
- Blend physical residual with a learned domain embedding so domains without floor data receive a muted gain.

---

## Risk 4: Cross-View/Joint Attention Adds Latency and Memory

**Description:** The `ReliabilityRefiner` attends over `V` views and `J` joints. With long temporal clips (`T=243`) and `V=8`, materializing `(B, T, V, J, hidden)` attention maps increases memory use and can slow training throughput.

**Impact:** Medium — A800 throughput drop or OOM in full-scale runs.

**Mitigation:**
- Use low-rank multi-head attention with `v53_orr_num_heads=4` and `v53_orr_hidden=64`.
- Apply the refiner per-frame rather than across time; temporal cues are already handled by the broadcast temporal residual.
- Benchmark peak memory on the RTX 4090 smoke; if overhead exceeds 5 %, replace attention with a lightweight 2-layer MLP refiner.

---

## Risk 5: Warm-Start Training Deadlocks at Identity

**Description:** With all final layers zero-initialized and the correction gate near zero, the v53 module may remain silent and never learn to down-weight outliers if the auxiliary loss is too weak or the v52 baseline already saturates the metric.

**Impact:** Low-Medium — module adds no value but does not regress.

**Mitigation:**
- Use a small non-zero bias on the reliability offset head (e.g., `+0.05`) so `γ` starts slightly above 1 for low-residual views and below 1 for high-residual views.
- Ramp `v53_orr_loss_weight` linearly from `0.0` to its target value over the first epoch.
- Monitor per-view weight histograms in smoke tests; if the distribution is unchanged after 500 steps, increase the loss weight or bias.

---

## Summary

| Risk | Probability | Severity | Owner | Mitigation Priority |
|---|---|---|---|---|
| Robust scale collapse | Medium | Medium | v53 implementer | High |
| Overwriting v52 weights | Medium | Medium | v53 implementer | High |
| Physical-cue cross-domain noise | Medium | High | v53 + v48 owners | High |
| Attention latency / memory | Medium | Medium | v53 implementer | Medium |
| Identity-init deadlock | Low-Medium | Medium | v53 implementer | Medium |

All mitigations should be exercised in the RTX 4090 smoke before a full A800 run is queued.
