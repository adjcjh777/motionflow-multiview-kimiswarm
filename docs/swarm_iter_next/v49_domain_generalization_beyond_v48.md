# v49: Self-Evolving Domain Generalization Beyond v48

**Status:** Design / candidate direction  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #166 (proposed)  
**Depends on:** v48-domain (#164)

## 1. Problem statement

v48 makes the model generalize across *known* domains (H36M, MPI-INF-3DHP, AIST++, 3DPW) by using domain labels, FiLM/conditional-BN adapters, and dynamic domain-weighted loss (DDWL). In practice, however, we cannot assume:

1. **Domain labels are available at test time.** Real-world deployment may receive streams whose domain is unknown or mixed.
2. **The target domain was seen during training.** New camera rigs, lighting conditions, or in-the-wild environments are not in the six-domain list.
3. **Static domain parameters are sufficient.** Domain shift can happen within a sequence (e.g., moving from indoor studio to outdoor sunlight).

v49 therefore goes **beyond v48** by removing the dependence on known domain labels and enabling the model to adapt to new, unseen domains **at test time** through a self-evolution feedback loop driven by uncertainty, view reliability, and reprojection residuals.

## 2. Proposed approach

v49 introduces a **Self-Evolving Domain Adapter (SEDA)** that replaces (or augments) v48's fixed per-domain FiLM with a small set of **domain basis vectors** and an online mechanism to blend them. The blending weights are inferred on-the-fly from self-supervised signals, so no domain label is required.

### 2.1 High-level idea

Instead of `domain_id -> FiLM params`, we use:

```text
input clip / frame
    |
    v
[v25/v45/v46 geometry fusion] --> per-frame 3D pose P_t
    |
    v
[v37 self-critique reliability + triangulation uncertainty] --> confidence map
    |
    v
[v49 SEDA] --> blend K domain basis modulations
    |
    v
refined pose P'_t
```

The domain basis modulations are learned during training, but the **mixing weights** are inferred at test time from the model's own feedback loop.

### 2.2 Core components

**Component A: Domain basis FiLM**

Replace the single per-domain FiLM in `DomainAdapterV48` with `K` learnable basis FiLM modules (shared across all domains). During training, a small gating network predicts a soft blend over the `K` bases from a domain embedding. During test-time self-evolution, the same gating network is updated from self-supervised signals instead of from domain labels.

**Component B: Self-evolution feedback loop (SEFL)**

For each test clip (or streaming window), the model runs a short iterative loop:

1. **Forward pass:** produce pose `P_t`, per-view reliability `r_vt`, and triangulation uncertainty `u_jt`.
2. **Reprojection residual:** project `P_t` back to each view and compute `e_vjt = ||x_hat - x_2d||`.
3. **Domain-consistency score:** for each candidate domain basis `k`, compute the mean reprojection residual after applying basis `k`. Lower residual = better fit.
4. **Update gate:** move the mixing weights toward the basis that yields lower reprojection error and higher reliability.
5. **Iterate 2–3 steps or until convergence.**

This is directly inspired by the self-evolution / self-improvement loop in Qwen: the model uses its own predictions and uncertainty to refine its internal state, without external labels.

**Component C: Uncertainty-gated blending**

The update step is gated by the v37/v46 reliability and triangulation uncertainty so that:

- Low-reliability views contribute less to the domain-consistency score.
- High-uncertainty frames are smoothed more aggressively across the temporal window.

### 2.3 How it fits with v46–v48 and the overall pipeline

- **v46 Sparse-View Generalization:** provides the per-view reliability head that tells SEDA which views to trust when scoring domain bases. View dropout during training also exposes the model to partial observations, making the domain bases more robust.
- **v47 Temporal Aggregation:** gives a temporal context. SEDA can run on short temporal windows, using v47's temporal smoothing as a prior for the self-evolution update.
- **v48 Domain Generalization:** provides the base architecture (FiLM, GRL, DDWL). v49 can be implemented as a **drop-in replacement** for the v48 `DomainAdapterV48` module, or as an optional `use_v49_self_evolving_domain` branch. v49 reuses the same hook points and flags.
- **Overall pipeline:** v49 sits at the same place as v48 (after geometry fusion, before final pose output). It adds a test-time loop that is active only at inference, so training cost is unchanged.

## 3. Concrete code-level changes

### New module

`motionflow_mv/fusion/self_evolving_domain_v49.py`:

```python
class SelfEvolvingDomainV49(nn.Module):
    """Test-time self-evolving domain adapter.

    Parameters
    ----------
    in_channels:
        Channel dimension of the feature tensor (last axis).
    num_bases:
        Number of domain basis FiLM modules (default 8).
    hidden:
        Hidden dimension of the gate MLP.
    n_iters:
        Number of self-evolution steps at test time.
    sigma_reproj:
        Cauchy scale for reprojection residuals (pixels).
    """

    def __init__(
        self,
        in_channels: int,
        num_bases: int = 8,
        hidden: int = 64,
        n_iters: int = 3,
        sigma_reproj: float = 5.0,
    ):
        ...

    def forward(
        self,
        feat: torch.Tensor,              # (B, T, V, J, C)
        points_2d: torch.Tensor,           # (B, T, V, J, 2)
        camera_params: dict,               # K, R, t
        view_mask: torch.Tensor,           # (B, T, V)
        reliability: torch.Tensor,         # (B, T, V, J)
        domain_id: Optional[torch.Tensor] = None,  # optional training label
    ) -> torch.Tensor:
        """Return domain-adapted features."""
        ...
```

### Modified files

| File | Change |
|------|--------|
| `motionflow_mv/fusion/self_evolving_domain_v49.py` | New SEDA module with domain-basis FiLM and self-evolution loop. |
| `motionflow_mv/fusion/domain_adapter_v48.py` | Optional: add `SelfEvolvingDomainV49` as alternative backend when `use_v49_self_evolving_domain=True`. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Add `use_v49_self_evolving_domain` flag; instantiate SEDA; branch in forward pass after v48 adapter. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add CLI flags; pass `reliability` and `points_2d` to SEDA; no extra training loss (SEDA is test-time only). |
| `experiments/eval_variable_views.py` | Report `MPJPE@k` with and without v49 SEDA; add `seda_n_iters` sweep. |
| `configs/benchmark_v49_domain_smoke.yaml` | Smoke config. |
| `scripts/run_v49_domain_smoke_local_4090.sh` | Smoke script. |

### New training/evaluation flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_self_evolving_domain` | bool | `False` | Master switch. |
| `v49_seda_num_bases` | int | `8` | Number of domain-basis FiLM modules. |
| `v49_seda_hidden` | int | `64` | Gate MLP hidden size. |
| `v49_seda_n_iters` | int | `3` | Self-evolution steps at test time. |
| `v49_seda_sigma_reproj` | float | `5.0` | Cauchy scale for reprojection residual. |
| `v49_seda_use_v48_film_init` | bool | `True` | Initialize bases from v48 per-domain FiLM weights if available. |

## 4. Risks / failure modes

| Risk | How it manifests | Mitigation |
|------|------------------|------------|
| **Self-evolution diverges** | Test-time loop oscillates or produces NaN. | Bound mixing weights with softmax temperature; early-stop when pose change < threshold. |
| **Bases collapse to one average domain** | All bases learn similar FiLM parameters; no specialization. | Add diversity loss on basis weights during training; use `num_bases=8` minimum. |
| **Noisy reprojection residuals mislead adaptation** | Occluded views dominate the domain score. | Use v37/v46 reliability as a soft mask; ignore residuals where reliability < 0.3. |
| **Latency increase at test time** | `n_iters x` forward passes. | Default `n_iters=3`; cache triangulation head outputs; skip SEDA when all views are high-reliability. |
| **Regression on known domains** | v49 performs worse than v48 on H36M/MPI because labels are no longer used. | Keep v48 branch as fallback; use v49 only when `domain_id is None`. |

## 5. Success metrics and experiments

### Evaluation metrics

Extend `experiments/eval_variable_views.py` to report:

| Metric | Description |
|--------|-------------|
| `MPJPE@k` (v48 baseline) | Standard v48 with known domain labels. |
| `MPJPE@k` (v49 SEDA) | v49 self-evolving adapter, no domain labels. |
| `basis_usage` | Entropy of mixing weights; high entropy = bases specialize. |
| `seda_convergence` | Fraction of clips that early-stop before `n_iters`. |

### Recommended smoke test

- **Hardware:** RTX 4090
- **Config:** `configs/benchmark_v49_domain_smoke.yaml`
- **Goal:**
  - No NaN/OOM.
  - `MPJPE@full` within 1 mm of v48 baseline on H36M/MPI val.
  - On a held-out *unseen* domain subset (e.g., a different camera rig or 3DPW actual), v49 improves over v48-with-missing-label by ≥5%.

### Recommended full experiment

- **Hardware:** A800-D
- **Config:** v48 full config + `use_v49_self_evolving_domain=True`
- **Data:** mixed manifest (H36M / MPI / AIST / 3DPW pseudo) + 3DPW actual val
- **Goal:**
  - No regression on H36M/MPI/AIST full-view `MPJPE@k`.
  - 3DPW actual `MPJPE@1` improves over v48 by ≥5%.
  - Domain-blind evaluation (drop `domain_id`) shows smaller cross-domain gap than v48 run with `domain_id`.

## 6. Self-evolution feedback loop

The v49 self-evolution loop is the central mechanism that differentiates it from v48:

```text
for step in 1..v49_seda_n_iters:
    # 1. Predict pose and per-view reliability
    P, r = model(points_2d, cameras, view_mask)

    # 2. Compute reprojection residuals
    e = reproject(P) - points_2d   # (B, T, V, J)

    # 3. Score each domain basis
    score_k = -mean(reliability * log(1 + (e / sigma)^2)) for basis k

    # 4. Update mixing weights
    w = softmax(score / temperature)

    # 5. Apply blended FiLM
    feat = sum_k w[k] * FiLM_k(feat)
```

This loop uses the model's own **reprojection consistency** as the learning signal, guided by **uncertainty/reliability** weights, exactly matching the Qwen-inspired self-evolution / self-improvement principle. It lets the model adapt to new domains without labels, closing the gap between v48's labeled multi-domain training and real-world deployment.

## 7. Relation to other variants

- **v37 self-critique reliability:** v49 reuses the learned per-view reliability to weight reprojection residuals in the self-evolution score.
- **v46 sparse-view generalization:** v46's view dropout and reliability head make the domain bases robust to missing views.
- **v47 temporal aggregation:** v49 can operate on v47-smoothed poses or use v47's temporal window as the self-evolution context.
- **v48 domain generalization:** v49 is a label-free, test-time extension of v48. The two can coexist: v48 handles known domains, v49 handles unknown ones.

## 8. Next steps

1. Wait for v48-domain smoke results (#164).
2. Implement `SelfEvolvingDomainV49` and unit tests.
3. Warm-initialize the `K` domain bases from v48 per-domain FiLM weights when available.
4. Wire the v49 branch into `OmniMultiViewFusionV5` and the trainer.
5. Run smoke on RTX 4090; compare v49-without-labels vs v48-with-labels on H36M/MPI/AIST/3DPW actual.
6. Queue full A800 run once smoke shows no regression and visible cross-domain gain.
