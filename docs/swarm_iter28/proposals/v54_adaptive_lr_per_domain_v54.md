# v54 Adaptive Low-Rank Per-Domain Fusion (ALR)

## 1. Motivation

The current MotionFlow-MultiView stack combines v48 domain-conditioned feature adaptation, v52 uncertainty-weighted triangulation, and v53 physical-space calibration. Yet once the spatio-temporal (ST) transformer has fused multi-view tokens, all domains share the exact same feature manifold. Studio captures (H36M/MPI) and in-the-wild sequences (3DPW/WebBridge) differ systematically in camera distribution, occlusion patterns, motion blur, and physical validity; a single shared manifold forces the model to compromise.

v54 introduces the **Adaptive Low-Rank Per-Domain Fusion (ALR)** module. It adds a bank of lightweight, domain-specific low-rank adapters on the fused feature tokens after the ST transformer. Each domain obtains a small, learnable residual subspace, while the shared backbone remains unchanged. The adapter outputs are gated by v52 uncertainty and v53 physical-space residuals, so the module is **identity-at-init** and only refines multi-view fusion where the current domain/uncertainty/physics signal justifies it.

## 2. Architecture

ALR is placed **after** the ST transformer and **before** the covariance/weight heads in `OmniMultiViewFusionV5`. It receives the fused per-view per-joint feature tokens and returns refined tokens of the same shape.

### 2.1 Low-rank per-domain adapters

For each domain `d` in a fixed domain list, learn two projection matrices:

```
A_d ∈ R^(d × r)      B_d ∈ R^(r × d)     with r << d
```

The per-domain residual is:

```
ΔF_d = B_d (A_d F)
```

`B_d` is zero-initialized, so `ΔF_d = 0` at initialization and the module is identity.

### 2.2 Domain routing and soft mixture

For a batch with per-sample domain labels `domain_id ∈ {0, ..., D-1}^B`, route each sample to its adapter via one-hot weights `m_d ∈ {0,1}^B`. For unseen domains or domain-agnostic inference, fall back to a learned shared adapter `ΔF_shared` (always trained) or a soft mixture over all adapters.

The mixed residual is:

```
ΔF = Σ_d m_d · ΔF_d   +   (1 - Σ_d m_d) · ΔF_shared
```

where the second term activates only when a sample has no matching domain adapter.

### 2.3 Uncertainty- and physics-aware gate

From v52, use the per-view triangulation weights `w_{b,t,v,j}`. From v53, use the per-joint physical residual magnitude `ρ_{b,t,j}` (reprojection + bone + floor residual). Pool these to a per-token scalar:

```
u_{b,t,v,j} = (1 / V) · Σ_v w_{b,t,v,j}          # high when v52 is confident
p_{b,t,v,j} = ρ_{b,t,j}                          # high when v53 disagrees with physics
```

A small MLP maps the concatenation of `u`, `p`, the token `F`, and the view-mean token `F̄` to a residual gate:

```
γ = σ( MLP_gate( [u, p, F, F̄] ) )
```

The final refined feature is:

```
F' = F + γ · ΔF
```

The gate MLP final layer is zero-initialized and its bias is set to `v54_alr_gate_init = -6.0`, so `γ ≈ 0.0025` at init.

### 2.4 Optional auxiliary loss

To discourage the adapters from drifting away from physically plausible features, add a tiny penalty on large gates when v53 reports low physical residual:

```
L_alr = (1 / (B T V J)) Σ γ · max(0, 1 - ρ / ρ_thr)
```

This loss is weighted by `v54_alr_loss_weight` and gated by `v54_alr_warmup_epochs`.

## 3. Inputs / Outputs (tensor shapes)

| Symbol | Shape | Description |
|--------|-------|-------------|
| `F` | `(B, T, V, J, d)` | Fused feature tokens after ST transformer |
| `domain_id` | `(B,)` | Integer domain label per sample |
| `v52_weights` | `(B, T, V, J)` | v52 UWT normalized weights |
| `v53_residual` | `(B, T, J)` | v53 physical-space residual magnitude per joint |
| `view_mask` | `(B, T, V)` | Optional visibility mask for variable views |
| **Output** `F'` | `(B, T, V, J, d)` | Domain-refined feature tokens |
| **Output** `alr_loss` | `()` | Optional scalar auxiliary loss |

## 4. Config flags

```
use_v54_adaptive_lr_per_domain: bool = False
v54_alr_num_domains: int = 6
v54_alr_rank: int = 8
v54_alr_hidden: int = 64
v54_alr_n_layers: int = 2
v54_alr_identity_init: bool = True
v54_alr_gate_init: float = -6.0
v54_alr_use_v52_gate: bool = True
v54_alr_use_v53_gate: bool = True
v54_alr_loss_weight: float = 0.001
v54_alr_dropout: float = 0.1
v54_alr_warmup_epochs: int = 0
v54_alr_share_unseen: bool = True
```

- `v54_alr_rank`: low-rank adapter dimension `r`. With `d=64` and `r=8`, each domain adds only `2·d·r = 1024` parameters, plus a small shared gate MLP.
- `v54_alr_share_unseen`: if True, use a learned shared adapter for domain IDs not in `0..num_domains-1`; if False, fall back to a uniform mixture of all domain adapters.

## 5. Expected MPJPE impact

- **Full-view inference:** modest gain of ~0.3–0.8 mm on H36M/MPI/WebBridge by allowing each domain a tailored fusion manifold.
- **Sparse/variable-view inference (`MPJPE@2`, `MPJPE@3`):** larger gain of 1.0–2.5 mm, because in-the-wild domains benefit from a dedicated adapter when views are scarce.
- **Warm-start verification:** loading a v53 checkpoint with v54 enabled should change `val_MPJPE@full` by ≤ 0.1 mm before training.
- **Cross-domain generalization:** when evaluated on an unseen domain, the shared/unseen adapter path keeps metrics within 0.5 mm of the v53 baseline.

## 6. Risks

See `docs/swarm_iter28/reports/agent_adaptive_lr_per_domain_v54_risks.md` for the full risk analysis. The main concerns are interaction with v48 domain conditioning, mixed-batch domain routing, unseen-domain behavior, and low-rank capacity.

## 7. 5-step implementation plan

1. **Module:** create `motionflow_mv/fusion/adaptive_lr_per_domain_v54.py` with `AdaptiveLowRankPerDomainV54` implementing the adapter bank, domain routing, gate MLP, and auxiliary loss.
2. **Wiring:** in `OmniMultiViewFusionV5.__init__` add the v54 flags and instantiate the module; in `forward` call it after the ST transformer and before the covariance/weight heads.
3. **Expose v52/v53 signals:** ensure the forward pass passes `v52_weights` and `v53_residual` to the module and stores them in the auxiliary `outputs` dict for logging.
4. **Smoke test:** create `configs/benchmark_v54_alr_smoke.yaml` and `scripts/run_v54_alr_smoke_local_4090.sh`; run on RTX 4090 and verify identity-at-init (Δ MPJPE ≤ 0.1 mm) and stable 1-epoch training.
5. **Ablate:** compare `v53` vs `v53 + v54` on the v50/v51/v52/v53 baseline; report `MPJPE@full` and `MPJPE@2/3/4` on H36M, MPI, and WebBridge domains separately.
