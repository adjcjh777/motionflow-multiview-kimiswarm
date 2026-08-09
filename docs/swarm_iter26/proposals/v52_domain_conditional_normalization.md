# v52: Domain-Conditional Normalization for Multi-View Feature Tokens

**Status:** Proposal (design-only)
**Labels:** `experiment`, `P1-next`
**Depends on:** v48 domain generalization (#164), v51 cross-domain sparse-view reliability (#181)

## Motivation

v48 adds a domain-conditional FiLM / GRL adapter at the *feature* level, and v51 refines per-view reliability from the v50 Self-Evolution Feedback Head. Neither module explicitly normalizes the internal feature distribution per domain before the spatial-temporal (ST) transformer. In mixed training (H36M, MPI-INF-3DHP, AIST++, 3DPW), the per-view token statistics—mean, variance, and higher-order moments—differ systematically: studio rigs have tight calibration and low noise, while in-the-wild monocular inputs are noisier and camera-geometry agnostic. A standard LayerNorm after the v48 adapter erases these domain-specific moments, which is useful for domain invariance but may discard domain-specific signal that could improve per-domain accuracy. v52 introduces a **Domain-Conditional Normalization (DCN)** layer that learns affine rescaling conditioned on the domain label, keeping the model warm-startable/identity-at-init and slotting directly into `OmniMultiViewFusionV5`.

## Proposed Architecture

### Module placement

`motionflow_mv/fusion/domain_conditional_normalization_v52.py` exposes `DomainConditionalNormalizationV52`. It is inserted into `OmniMultiViewFusionV5.forward` **after** the optional v48 domain adapter and **before** the ST transformer:

```text
feat (B, T, V, J, d)
    |
    ▼
[optional v48 domain adapter]
    |
    ▼
[ v52 Domain-Conditional Normalization ]
    |   domain_id + view_mask
    ▼
[ ST transformer ]
```

### Forward logic

Inputs:
- `feat`: `(B, T, V, J, d)` multi-view feature tokens produced by v25/v45 geometry fusion, hierarchical encoders, graph networks, and v48 adapter.
- `domain_id`: `(B,)` integer domain labels (0=H36M, 1=MPI, 2=AIST, ..., 5=3DPW).
- `view_mask`: `(B, T, V)` binary mask, used to compute per-sample active view count.

Outputs:
- `feat'`: `(B, T, V, J, d)` normalized tokens with domain-conditional affine parameters.

### Equations

Given a token tensor `f ∈ R^(B×T×V×J×d)`:

```
μ_b, σ_b = LayerNormStatistics(f_b)               # per-sample mean/std, (B, 1, 1, 1, d)
f̂_b      = (f_b - μ_b) / (σ_b + ε)                # standard LayerNorm output

z_d      = Embed(domain_id)                       # (B, d_emb)
if use_view_count:
    n_views_active = view_mask.sum(dim=-1)        # (B, T)
    v_emb          = MLP_view_count(n_views_active)  # (B, T, d_emb)
    z_d            = z_d.unsqueeze(1) + v_emb        # (B, T, d_emb)

h        = MLP_dcn(z_d)                           # (B, T, 2·g)
γ, β     = split(h)                               # each (B, T, g)
γ        = γ.unsqueeze(-1)                        # (B, T, 1, 1, g)
β        = β.unsqueeze(-1)

# Group-wise broadcast over channels
f_grouped = f̂_b.view(B, T, V, J, g, c/g)
f'_grouped = f̂_grouped * (1 + γ) + β
feat'    = f'_grouped.view(B, T, V, J, d)
```

At initialization, the final projection layers of `MLP_dcn` are zero-initialized, so `γ = 0` and `β = 0`, giving `feat' = f̂_b`. This guarantees the module is **identity at init** and can be safely enabled on top of any existing checkpoint (v25, v45, v46, v47, v48, v50, or v51).

### Implementation API

```python
class DomainConditionalNormalizationV52(nn.Module):
    def __init__(
        self,
        d: int = 64,
        num_domains: int = 6,
        num_groups: int = 4,
        hidden: int = 64,
        dropout: float = 0.1,
        use_view_count_conditioning: bool = True,
        identity_init: bool = True,
    ):
        ...

    def forward(
        self,
        feat: torch.Tensor,        # (B, T, V, J, d)
        domain_id: torch.Tensor,   # (B,)
        view_mask: torch.Tensor,   # (B, T, V)
    ) -> torch.Tensor:
        ...
```

## Inputs / Outputs Summary

| Tensor | Shape | Description |
|--------|-------|-------------|
| `feat` | `(B, T, V, J, d)` | Input feature tokens from upstream v48 adapter |
| `domain_id` | `(B,)` | Integer domain label per clip |
| `view_mask` | `(B, T, V)` | Binary mask of active views |
| Output `feat'` | `(B, T, V, J, d)` | Domain-conditionally normalized tokens |

## Config Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v52_domain_conditional_normalization` | `bool` | `False` | Enable the v52 DCN module |
| `v52_dcn_num_domains` | `int` | `6` | Number of distinct domain labels |
| `v52_dcn_num_groups` | `int` | `4` | Channel groups for group-wise affine (must divide `d`) |
| `v52_dcn_hidden` | `int` | `64` | Hidden dimension of the domain MLP |
| `v52_dcn_dropout` | `float` | `0.1` | Dropout in the domain MLP |
| `v52_dcn_use_view_count_conditioning` | `bool` | `True` | Append active-view-count embedding for sparse-view robustness |
| `v52_dcn_identity_init` | `bool` | `True` | Zero-initialize output projections so module is identity at startup |

## Expected MPJPE Impact

- **Primary:** Reduces cross-domain gap by 3–8% relative (≈0.5–1.5 mm on studio full-view MPJPE, larger on 3DPW actual).
- **Sparse views:** The view-count-conditioned affine is expected to improve `MPJPE@2` and `MPJPE@3` on mixed-domain val, because it compensates for the shift in feature statistics when only a subset of views is available.
- **Baseline:** Because the module is identity at init, a smoke test with `use_v52_domain_conditional_normalization=True` should reproduce the parent variant's numbers (v48 or v51) to within random-seed noise.

## Risks

See `docs/swarm_iter26/reports/agent_domain_conditional_normalization_risks.md` for full details. Top risks:

1. **Redundancy with v48 FiLM** — v52 also predicts per-domain affine terms, potentially duplicating v48's adaptation.
2. **Instability from conditional normalization** — If `v52_dcn_num_groups` is too small or the MLP is too deep, the module can overfit to small domains.
3. **View-count conditioning noise** — For very small `T`, the active-view-count embedding may be noisy.
4. **Interaction with v50/v51 auxiliary heads** — v52 changes the feature distribution fed into the ST transformer, which may shift the learned behavior of the residual MLP and v50/v51 heads.

## 5-Step Implementation Plan

1. **Create module.** Implement `motionflow_mv/fusion/domain_conditional_normalization_v52.py` with the API above, unit tests for identity-at-init, and smoke tests for shape correctness.
2. **Wire into `OmniMultiViewFusionV5`.** Add v52 flags to `__init__`, instantiate `DomainConditionalNormalizationV52`, and call it in `forward` immediately after the v48 adapter block (around line 1476) and before the ST transformer permutation.
3. **Update trainer and config.** In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, ensure `domain_id` is passed to the model. Add `configs/benchmark_v52_domain_conditional_normalization_smoke.yaml` and `scripts/run_v52_domain_conditional_normalization_smoke_local_4090.sh`.
4. **Smoke test.** Run on RTX 4090 with `use_v52_domain_conditional_normalization=True` against the v48 or v51 baseline. Verify no NaN/OOM and that epoch-1 val_MPJPE is within ±1 mm of the parent run.
5. **Ablation and full run.** Ablate `v52_dcn_use_view_count_conditioning` and `v52_dcn_num_groups` on the smoke config. If positive, queue an A800 run on top of the best v48/v51 checkpoint and update `docs/swarm_iter26/status.md`.

## Paper Story Fit

v52 reinforces the paper narrative of an *optimized multi-view motionflow pipeline*: after multi-view video, pose extraction, and multi-view fusion/calibration, the feature representation must be normalized in a domain-aware manner before physical-space alignment. By conditioning normalization on both the dataset domain and the instantaneous view count, the model explicitly acknowledges that different recording conditions produce different internal statistics, while remaining warm-startable and minimally invasive.
