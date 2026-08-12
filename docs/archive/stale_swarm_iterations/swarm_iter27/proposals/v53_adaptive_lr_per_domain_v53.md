# v53: Adaptive Learning Rate Per Domain with v52 Uncertainty Feedback (DU-LR)

**Task identifier:** `design_v53_adaptive_lr_per_domain`  
**Status:** Proposal (no code yet)  
**Depends on:** v41 weighted domain loss, v48 domain generalization, v50 Self-Evolution Feedback Head, v51 Cross-Domain Sparse-View Reliability, v52 Uncertainty-Weighted Triangulation

## 1. Motivation

v41 and v48 already separate domains, but the optimizer still applies a single global learning rate. Studio captures (H36M/MPI) and in-the-wild sequences (3DPW/WebBridge) differ in noise, occlusion, and batch-size balance, so their gradients have different scales and convergence speeds. v51 proposed a domain-level LR scaler, but it ignored the per-sample, per-view uncertainty that the model itself computes.

**v53** closes the loop: it uses the per-view, per-joint precision weights produced by v52 Uncertainty-Weighted Triangulation as a live domain-difficulty signal to rescale the optimizer step per domain. Domains that v52 consistently labels as uncertain receive a larger effective step; domains that are already well-modelled have their step reduced. Because the scaling starts at identity and only activates after a short warmup, it cannot collapse a trained v52 checkpoint.

## 2. Module overview

**File (proposed):** `motionflow_mv/training/adaptive_lr_per_domain_v53.py`

```text
AdaptiveLRPerDomainV53(
    domains=[0, 1, 2, 5],
    warmup_steps=500,
    beta=0.99,
    tau=0.5,
    lambda_unc=1.0,
    s_target=0.5,
    gamma=2.0,
    min_samples=50,
    uncertainty_source="weights",
    sefh_coupling=True,
)
```

### 2.1 Inputs / outputs

The helper is called once per training step inside the trainer loop.

**Inputs**

* `domain_ids`: `(B,)` integer domain labels from the batch.
* `v52_weights`: `(B, T, V, J)` normalized triangulation weights output by `UncertaintyWeightedTriangulationV52`.
* `v52_log_precision`: `(B, T, V, J)` or `(B, T, V)` log-precision values from v52 (optional).
* `sefh_reliability`: `(B, T, V)` per-view reliability from v50 Self-Evolution Feedback Head (optional).
* `grad_dict`: dict mapping parameter group name to per-domain gradient norm.

**Outputs**

* `scales`: dict mapping `domain_id -> float` (the LR multiplier `α_d`).
* `diagnostics`: dict with `mean_uncertainty_d`, `grad_norm_d`, and `scale_d` for logging.

### 2.2 Architecture and equations

For each domain `d` in the batch, compute an uncertainty proxy from v52 weights:

```
if uncertainty_source == "weights":
    u_b = mean_{v,j} (1 - w_{b,v,j})                # high when weights are low
else:  # "precision"
    u_b = mean_{v,j} exp(-log_precision_{b,v,j})    # high when precision is low
```

Update an exponential moving average of domain uncertainty:

```
s_d^(t) = beta * s_d^(t-1) + (1 - beta) * mean_{b in d} u_b
```

Compute per-domain gradient norm and global gradient norm:

```
g_d     = mean_{b in d} ||∇θ L_b||
g_global = median_d g_d
```

The final per-domain LR scale is:

```
α_d = clamp( (g_global / (g_d + ε))^τ * exp(-λ * (s_d - s_target)), 1/γ, γ )
```

The scale is identity-at-warmup: `α_d = 1` until every domain in the current batch has been seen at least `min_samples` times. If `sefh_coupling=True`, the scale is further multiplied by the mean v50 reliability of the domain, clamped to `[0.8, 1.25]`.

The optimizer update becomes:

```
θ ← θ - η * α_{d_batch} * ∇θ L
```

where `d_batch` is the domain label of the current batch (or the majority domain for mixed batches).

### 2.3 Integration into the pipeline

v53 lives in the trainer, not inside `OmniMultiViewFusionV5`. However, the model must expose the v52 outputs so the trainer can read them:

```python
aux_losses, outputs = model(...)
# outputs contains "v52_weights" and "v52_log_precision"
v52_weights = outputs["v52_weights"]
v52_log_precision = outputs["v52_log_precision"]

scales = adaptive_lr.update(
    domain_ids=batch["domain_id"],
    v52_weights=v52_weights,
    v52_log_precision=v52_log_precision,
    sefh_reliability=outputs.get("sefh_reliability"),
    grad_dict=trainer.grad_norms,
)
```

### 2.4 Warm-start / identity-at-init

The module is warm-startable by construction. The model parameters are unchanged; only the optimizer step is rescaled. With `warmup_steps` and `min_samples` active, `α_d ≡ 1` until enough v52 statistics are gathered, so loading a v52 checkpoint and enabling v53 leaves inference metrics unchanged.

## 3. Config flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v53_adaptive_lr_per_domain` | bool | `False` | Enable the v53 DU-LR helper. |
| `v53_dulr_domains` | list[int] | `[0,1,2,5]` | Domain ids expected in mixed training. |
| `v53_dulr_warmup_steps` | int | `500` | Steps before any scaling is applied. |
| `v53_dulr_beta` | float | `0.99` | EMA decay for domain uncertainty. |
| `v53_dulr_tau` | float | `0.5` | Exponent for gradient-norm rebalancing. |
| `v53_dulr_lambda_unc` | float | `1.0` | Strength of uncertainty correction. |
| `v53_dulr_s_target` | float | `0.5` | Target mean uncertainty; higher values reduce LR. |
| `v53_dulr_gamma` | float | `2.0` | Max LR multiplier, clamps `α_d ∈ [1/γ, γ]`. |
| `v53_dulr_min_samples` | int | `50` | Minimum per-domain samples before scaling. |
| `v53_dulr_uncertainty_source` | str | `"weights"` | `"weights"` or `"precision"`. |
| `v53_dulr_sefh_coupling` | bool | `True` | Multiply by v50 SEFH domain reliability. |

## 4. Expected MPJPE impact

* **Smoke (RTX 4090, 50–100 samples):** ≤ 0.5 mm change; identity warmup preserves the v52 baseline.
* **Medium (500–2k samples):** 0.5–2 mm improvement on 3DPW actual and WebBridge wild domains, mainly on `MPJPE@2` and `MPJPE@3`.
* **Full (mixed, 10k+ samples):** 1–3 mm improvement over the strongest v50/v51/v52 stack on cross-domain evaluation, with faster convergence of under-represented domains.

## 5. Risks

See `docs/swarm_iter27/reports/agent_adaptive_lr_per_domain_v53_risks.md` for detailed risks and mitigations. The main concerns are unstable LR oscillations, interaction with mixed-batch sampling, dependence on v52 noise, and extra trainer complexity.

## 6. Implementation plan

1. **Trainer helper:** Implement `AdaptiveLRPerDomainV53` in `motionflow_mv/training/adaptive_lr_per_domain_v53.py` with EMA buffers, per-domain gradient-norm tracking, and warmup logic.
2. **Expose v52 outputs:** Modify `OmniMultiViewFusionV5.forward` to return `v52_weights` and `v52_log_precision` in the auxiliary `outputs` dict when `use_v53_adaptive_lr_per_domain=True`.
3. **Trainer wiring:** In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, collect v52 outputs after `model.forward`, call `adaptive_lr.update(...)`, and apply `α_d` to each parameter group before `optimizer.step()`.
4. **Smoke config:** Create `configs/benchmark_v53_adaptive_lr_per_domain_smoke.yaml` and `scripts/run_v53_adaptive_lr_per_domain_smoke_local_4090.sh`; verify identity during warmup and finite α_d values.
5. **Unit tests + ablation:** Add `tests/test_adaptive_lr_per_domain_v53.py` covering warmup identity, clamping, EMA behaviour, and an ablation that confirms no regression against the v52-only baseline.
