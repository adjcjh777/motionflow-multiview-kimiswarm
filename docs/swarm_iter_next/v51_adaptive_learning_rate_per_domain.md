# v51 Adaptive Learning Rate Per Domain

**Focus area:** adaptive_learning_rate_per_domain  
**Depends on:** v41 weighted domain loss, v48 domain generalization, v50 SEFH  
**Status:** proposal for v51 design swarm

## 1. Idea

v41 and v48 already separate domains, but the optimizer still uses a single global learning rate. Studio (H36M/MPI) and in-the-wild (3DPW/WebBridge) batches differ in noise, batch-size balance, and gradient scale. v51 proposes a **domain-conditioned adaptive learning-rate scaler (DALR)** that rescales the effective step per domain during training. It has no inference cost and keeps the v49 real-time head unchanged.

## 2. Architecture

DALR is a trainer-side helper, not a model submodule.

1. **Per-domain gradient buffer.** For each domain `d`, maintain an moving average of gradient norm `g_d` and a loss-curvature proxy `ρ_d = |Δloss_d| / (||g_d||² η)` over the last `N` steps.
2. **LR scaler.** `α_d = clamp(τ · g_d / (g_d + g_global) · (1 + κ · (ρ_d − ρ_target)), 1/γ, γ)`. Domains with above-median gradient norm or slower-than-target loss decrease get a reduced LR; under-represented domains get a boost.
3. **SEFH coupling (optional).** If `use_v50_self_evolution_feedback_head=True`, the per-domain mean SEFH reliability is multiplied into `α_d`, so high-reliability domains tolerate a larger step.

The update becomes `θ ← θ − η · α_d · ∇L_d` for the batch's domain label.

## 3. Config Flags and Defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_adaptive_lr_per_domain` | bool | `False` |
| `v51_dalr_buffer_steps` | int | `100` |
| `v51_dalr_tau` | float | `1.0` |
| `v51_dalr_kappa` | float | `0.5` |
| `v51_dalr_rho_target` | float | `0.1` |
| `v51_dalr_gamma` | float | `2.0` |
| `v51_dalr_min_samples` | int | `10` |
| `v51_dalr_sefh_coupling` | bool | `True` |
| `v51_dalr_apply_to_bn_bias` | bool | `False` |

## 4. Loss Term

None. DALR only rescales the optimizer step; it does not add a loss term.

## 5. Evaluation Metric

- Primary: `val_MPJPE@full` and per-domain `MPJPE@k` (k=2,3,4,full) from `experiments/eval_variable_views.py`.
- Diagnostics: per-domain `α_d`, per-domain validation gap `Δ_MPJPE_d`, and epochs-to-convergence.

## 6. Expected MPJPE Impact

- H36M/MPI (studio): ±0 mm full-view; small sparse-view gain from balanced gradients.
- 3DPW actual / WebBridge (wild): `MPJPE@2` −2 to −4 mm, `MPJPE@3` −1 to −2 mm, full-view −0.5 to −1.5 mm.
- DALR is identity-at-init: `α_d = 1` until enough statistics are gathered, so it cannot collapse the v50 baseline.

## 7. Main Risk

**Risk:** `α_d` can oscillate and diverge when a domain has very few batches (e.g., small 3DPW actual smoke set).  
**Mitigation:** Require `v51_dalr_min_samples` before scaling, clamp `α_d ∈ [1/γ, γ]`, smooth updates with momentum, and start smoke with `v51_dalr_gamma=1.5`.

## 8. Integration

- Trainer: `experiments/train_omniview_fusion_v5_webbridge_multi.py` — add `AdaptiveDomainLR` helper, collect per-domain gradient norms after `loss.backward()`, and apply `α_d` before `optimizer.step()`.
- Data: reuse v41/v48 domain labels from `webbridge_mixed_dataset.py`.
- Smoke: base on v48-domain or v50 SEFH smoke; accept if `val_MPJPE@full` is within 1 mm of base and 3DPW actual `MPJPE@2` improves ≥1 mm.

## 9. Paper-Story Fit

DALR extends the self-evolution narrative from *what the model believes* (SEFH reliability) to *how the model learns*. Reliability tells the model which views to trust at inference; DALR tells the optimizer which domains need more attention during training, forming a coherent self-evolving training–inference loop across views, time, and domains.
