# v51 Paper-Gap Analysis: Cross-Domain Sparse-View Reliability (CDSVR)

## Identified paper gap

The v49 paper story closes with self-evolution (v37/v39/v43, consolidated in v50 SEFH) and sparse-view / temporal / domain robustness (v46–v48). Yet the narrative still lacks a direct bridge between the two: a reliability mechanism that is *itself* domain-aware. Paper-gap analysis shows that v50 SEFH learns per-view reliability from reprojection, temporal, and epipolar residuals, but those residuals are distribution-dependent. On 3DPW actual the residual-to-reliability mapping can differ from H36M/MPI, so the same learned reliability head may under-trust good views or over-trust noisy ones when only two or three views are available. This cross-domain sparse-view reliability gap is the next concrete hole to fill.

## Proposed v51 module: Cross-Domain Sparse-View Reliability (CDSVR)

**Architecture.** `CrossDomainSparseViewReliabilityV51` is a lightweight 2-layer cross-attention block that consumes (1) the v50 SEFH per-view reliability vector `r ∈ R^V`, (2) the per-joint log-variance `σ ∈ R^J`, and (3) a domain embedding `d` (from v48's domain adapter or a one-hot domain label). It outputs a domain-conditioned reliability offset `Δr ∈ R^V` and a per-joint uncertainty rescale `α ∈ R^J`. These are applied residual/multiplicatively: `r'_v = r_v + Δr_v`, `σ'_j = σ_j / α_j`. The module is identity-at-init: `Δr_v` is zero-initialized and `α_j` is initialized to one, so enabling the flag leaves the v50 baseline unchanged at startup.

**New config flags / defaults.**

| Flag | Type | Default |
|---|---|---|
| `use_v51_cross_domain_sparse_view_reliability` | bool | `False` |
| `v51_cdsvr_hidden` | int | `64` |
| `v51_cdsvr_num_heads` | int | `4` |
| `v51_cdsvr_dropout` | float | `0.1` |
| `v51_cdsvr_offset_min` | float | `0.05` |
| `v51_cdsvr_use_domain_label` | bool | `True` |
| `v51_cdsvr_uncertainty_temperature` | float | `1.0` |
| `v51_cdsvr_identity_init_gate` | bool | `True` |
| `loss.v51_cdsvr_loss_weight` | float | `0.01` |

**Loss term.** `L_cdsvr = λ · [ (1/V) Σ_v w'_v · Huber(||ε_v||, δ) − (1/J) Σ_j log α_j + γ · Var(w') ]`, where `w'_v = sigmoid(r'_v / τ)`, `ε_v` is the reprojection residual of the refined pose, and `τ` is a temperature. The first term couples domain-conditioned reliability to geometric residuals; the second prevents collapsed uncertainty; the third preserves view diversity. `λ = loss.v51_cdsvr_loss_weight`, `δ = 0.1`, `γ = 0.01`.

**Evaluation metric.** Standard `MPJPE@k` for `k = 2,3,4,full`; per-domain `MPJPE@k`; 3DPW actual `MPJPE@2/3`; and diagnostic `Spearman(r'_v, ||ε_v||) > 0.35` plus an ECE-style calibration score for `exp(−σ')`.

**Expected MPJPE impact.** On 3DPW actual we expect `MPJPE@2` −5 to −7 mm and `MPJPE@3` −3 to −4 mm, with full-view H36M/MPI within ±0.5 mm. The sparse-view cross-domain gap should shrink more than the in-domain gap.

**Main risk.** The module may conflate domain shift with view dropout, causing reliability to collapse to a single mode on small domains. Mitigation: identity-at-init, clamp `Δr_v` to `[-2, 2]`, freeze base v50 weights for the first epoch, and require `use_v50_self_evolution_feedback_head=True` when this flag is set.

## Integration sketch

- Module: `motionflow_mv/fusion/cross_domain_sparse_view_reliability_v51.py`
- Wiring: call after `SelfEvolutionFeedbackHeadV50` in `omniview_fusion_v5.py`; consume v48 domain embedding if available, otherwise use one-hot domain label.
- Smoke config: `configs/benchmark_v51_cdsvr_smoke.yaml`, warm-started from the best v50 checkpoint.
