# Agent-03: DomainAdapterV48 API and Integration Notes

**Owner:** Agent-03  
**Branch:** `v48-domain`  
**Tracking issue:** #164  
**Date:** 2026-08-09

## 1. Scope

Finalize the public API for the v48 domain-generalization adapter and document exactly how it plugs into:

- `motionflow_mv/fusion/domain_adapter_v48.py`
- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- `experiments/eval_variable_views.py`
- `tests/test_domain_adapter_v48.py`

This report does **not** modify source files; it resolves the remaining naming/ABI inconsistencies so the IMPLEMENT agents can commit a consistent stack.

## 2. Design decisions

| Decision | Rationale |
|----------|-----------|
| Module name | `DomainAdapterV48` in `motionflow_mv/fusion/domain_adapter_v48.py` (already implemented). |
| Model flag | `use_v48_domain_generalization` — aligns with the v48 proposal, the smoke script, and the test that checks for this flag. The current draft uses `use_v48_domain_adapter`; this should be renamed. |
| Prefix | `v48_dg_*` for all model/trainer flags (`dg` = domain generalization). Avoid `v48_da_*` which collides with the existing proposal and smoke script. |
| Integration point | Feature level, **after** the domain embedding and **before** the spatio-temporal (time+view) transformer. The adapter transforms multi-view feature tokens; triangulation stays unchanged. |
| Warm-start | FiLM and conditional BN are initialized to identity. The adapter is a no-op at init, so a v47 checkpoint can be loaded first. |
| Domain loss | The model computes `F.cross_entropy(domain_logits, dataset_id)` internally and returns the scalar as `out[7]`. This avoids exposing raw logits to the trainer and keeps the trainer loss code small. |
| DDWL | Implemented as a stateful closure in the trainer, not inside the adapter. It up-weights harder domains after a configurable warmup. |
| Per-domain dropout | A JSON dict keyed by `dataset_id` (e.g. `{"0": 0.30, "5": 0.15}`). 3DPW actual (`V=1`) can be given a gentler rate. |

## 3. Final `DomainAdapterV48` API

```python
class DomainAdapterV48(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_domains: int = 6,
        hidden: int = 64,
        dropout: float = 0.1,
        use_film: bool = True,
        use_conditional_bn: bool = False,
        use_grl_discriminator: bool = True,
        grl_lambda: float = 0.1,
    ):
        """
        Parameters
        ----------
        in_channels:
            Channel dimension of the input feature tensor (last dimension).
        num_domains:
            Number of distinct datasets/domains. Default 6 covers
            h36m, mpi, aist, shelf, campus, 3dpw.
        hidden:
            Hidden dimension of the domain-embedding MLP.
        dropout:
            Dropout probability inside the FiLM and discriminator MLPs.
        use_film:
            Apply domain-conditional FiLM modulation. Recommended default: True.
        use_conditional_bn:
            Apply per-domain conditional batch normalization. Optional; default False.
        use_grl_discriminator:
            Attach a gradient-reversal domain discriminator. Recommended default: True.
        grl_lambda:
            Scaling factor for the gradient reversal layer.
        """
        ...

    def forward(
        self,
        feat: torch.Tensor,                 # (B, T, V, J, C)
        dataset_id: torch.Tensor,            # (B,)
        view_mask: Optional[torch.Tensor] = None,  # (B, T, V) — reserved, currently unused
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (adapted_features, domain_logits)."""
        ...
```

**Outputs**

- `adapted_features`: same shape as `feat`.
- `domain_logits`: `(B, num_domains)` logits, or `None` if `use_grl_discriminator=False`.

**Identity-at-init behavior**

When `use_film=True` and `use_conditional_bn=False`, the FiLM MLP is zero-initialized, so `adapted_features ≈ feat` at the start of training.

## 4. `OmniMultiViewFusionV5` integration

### Constructor additions

```python
# v48 domain generalization
use_v48_domain_generalization: bool = False,
v48_dg_hidden: int = 64,
v48_dg_num_domains: int = 6,
v48_dg_dropout: float = 0.1,
v48_dg_use_film: bool = True,
v48_dg_use_conditional_bn: bool = False,
v48_dg_use_grl_discriminator: bool = True,
v48_dg_grl_lambda: float = 0.1,
```

Instantiation (after the v47 temporal head):

```python
self.use_v48_domain_generalization = use_v48_domain_generalization
self.v48_dg_hidden = v48_dg_hidden
self.v48_dg_num_domains = v48_dg_num_domains

if self.use_v48_domain_generalization:
    from motionflow_mv.fusion.domain_adapter_v48 import DomainAdapterV48

    self.domain_adapter_v48 = DomainAdapterV48(
        in_channels=self.d,
        num_domains=v48_dg_num_domains,
        hidden=v48_dg_hidden,
        dropout=v48_dg_dropout,
        use_film=v48_dg_use_film,
        use_conditional_bn=v48_dg_use_conditional_bn,
        use_grl_discriminator=v48_dg_use_grl_discriminator,
        grl_lambda=v48_dg_grl_lambda,
    )
else:
    self.domain_adapter_v48 = None
```

### Forward additions

Apply the adapter right after the optional domain embedding and before the ST transformer:

```python
# Optional domain embedding (additive, pre-existing).
if self.use_domain_embedding and domain_id is not None:
    domain_emb = self.domain_embedding(domain_id)  # (B, d)
    feat = feat + domain_emb.view(B, 1, 1, 1, self.d)

# v48 domain-conditional feature adaptation.
v48_domain_loss = None
if (
    self.use_v48_domain_generalization
    and self.domain_adapter_v48 is not None
    and domain_id is not None
):
    feat, domain_logits = self.domain_adapter_v48(
        feat,
        dataset_id=domain_id,
        view_mask=view_mask_flat.view(B, T, V),
    )
    if domain_logits is not None:
        v48_domain_loss = F.cross_entropy(domain_logits, domain_id)

# Spatio-temporal (time + view) attention with optional epipolar bias.
...
```

### Return tuple

The model currently returns:

```python
out = (pred_3d, weights, visibility, L, epi_loss)
```

Extend it to a length-8 tuple so the trainer can access `v48_domain_loss` at `out[7]`:

```python
out = (pred_3d, weights, visibility, L, epi_loss, None, None, v48_domain_loss)
```

`None` placeholders keep `out[5]` and `out[6]` reserved for the trainer's legacy entropy/budget loss slots.

## 5. Trainer / CLI integration

The trainer draft already adds the following flags:

```python
parser.add_argument("--use_v48_domain_generalization", action="store_true", default=False)
parser.add_argument("--v48_dg_hidden", type=int, default=64)
parser.add_argument("--v48_dg_grl_lambda", type=float, default=0.1)
parser.add_argument("--v48_dg_use_domain_film", action="store_true", default=True)
parser.add_argument("--no_v48_dg_use_domain_film", dest="v48_dg_use_domain_film", action="store_false")
parser.add_argument("--v48_dg_use_ddwl", action="store_true", default=True)
parser.add_argument("--no_v48_dg_use_ddwl", dest="v48_dg_use_ddwl", action="store_false")
parser.add_argument("--v48_dg_ddwl_temperature", type=float, default=2.0)
parser.add_argument("--v48_dg_ddwl_warmup_epochs", type=int, default=1)
parser.add_argument("--v48_3dpw_actual_val_paths", type=str, default=None)
parser.add_argument("--v48_dropout_per_domain", type=str, default=None)
```

### DDWL state (trainer)

Stateful per-domain EMA maintained in `build_compute_loss`:

```python
v48_loss_ema: Dict[int, float] = {d: 0.0 for d in range(v48_num_domains_ddwl)}
```

Per-batch update:

```python
for d in range(v48_num_domains_ddwl):
    mask = (dataset_id.squeeze(-1) == d)
    if mask.any():
        mean_loss_d = mse_per_sample[mask].mean().item()
        v48_loss_ema[d] = (
            v48_ddwl_beta * v48_loss_ema[d]
            + (1.0 - v48_ddwl_beta) * mean_loss_d
        )
```

Adaptive weights after warmup:

```python
ema_t = torch.tensor(
    [max(v48_loss_ema[d], 1e-8) for d in range(v48_num_domains_ddwl)],
    device=device, dtype=torch.float32,
)
w_d_raw = (ema_t / ema_t.max()) ** (-1.0 / v48_ddwl_temperature)
w_d = w_d_raw / w_d_raw.sum() * v48_num_domains_ddwl
w_d = w_d.clamp(0.5, 2.0)
```

The model receives the v48 domain-loss scalar at `out[7]`:

```python
v48_domain_loss = out[7] if len(out) > 7 else None
...
if v48_domain_loss is not None:
    loss = loss + v48_domain_loss
    metrics["v48_domain_loss"] = v48_domain_loss.item()
```

### Per-domain view dropout

A helper `apply_per_domain_view_dropout` takes the JSON string, parses it to a `Dict[str, float]`, and zeros out the confidence channel per sample according to the sample's `dataset_id`. The existing v46 view-dropout probability is set to `0.0` when per-domain dropout is active.

## 6. Data loader and evaluation

### Domain IDs

| ID | Domain |
|----|--------|
| 0 | H36M |
| 1 | MPI-INF-3DHP |
| 2 | AIST++ |
| 3 | Shelf |
| 4 | Campus |
| 5 | 3DPW (pseudo + actual) |

3DPW `actual` mode uses `V=1` with per-frame camera arrays; the loader returns tensors of shape `(T, MAX_VIEWS, ...)` so the model can consume them with the standard `view_mask`.

### Per-dataset evaluation

`experiments/eval_variable_views.py` now supports:

```bash
python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/v48_domain_smoke_local_4090.pth \
    --config outputs/v48_domain_smoke_local_4090.config.json \
    --dataset_manifest docs/swarm_iter25/v48_eval_manifest.txt \
    --output_csv outputs/v48_per_dataset.csv
```

The manifest format is one `<name> <path>` per line. Output includes per-dataset `MPJPE@k` and the cross-dataset `domain_gap`.

## 7. Smoke config / script

The smoke script `scripts/run_v48_domain_smoke_local_4090.sh` should:

1. Enable v46 sparse-view generalization.
2. Enable v47 temporal aggregation.
3. Enable v48 domain generalization with the flags in Section 5.
4. Use a mixed manifest that includes at least H36M, MPI, and 3DPW pseudo.
5. Run 1–2 epochs with `train_samples <= 500` for a quick RTX 4090 sanity check.

Success criteria:

- `val_MPJPE` finite for all domains.
- No `NaN` or `Inf` in losses.
- Domain discriminator accuracy stays within `[0.45, 0.55]`.
- v48 domain loss is non-zero and produces gradients.

## 8. Tests

`tests/test_domain_adapter_v48.py` should verify:

- Output shape matches input shape.
- Identity-like behaviour at init when only FiLM is enabled.
- Invalid `dataset_id` values raise `ValueError`.
- Gradients flow through all enabled sub-modules.
- `OmniMultiViewFusionV5` with `use_v48_domain_generalization=True` can run a forward pass.

The existing unit-level tests already cover the first four items; the integration test is gated on the flag name `use_v48_domain_generalization`.

## 9. Open inconsistencies to resolve before merge

1. **Flag naming in `omniview_fusion_v5.py`:**
   - Current draft uses `use_v48_domain_adapter` and `v48_da_*`.
   - Change to `use_v48_domain_generalization` and `v48_dg_*` to match the trainer, tests, and proposal.

2. **Adapter arguments not yet wired:**
   - `v48_dg_use_domain_film` should map to `use_film`.
   - `v48_dg_grl_lambda` should map to `grl_lambda`.
   - Optionally support `v48_dg_use_conditional_bn` and `v48_dg_use_grl_discriminator`.

3. **Domain loss not returned:**
   - `omniview_fusion_v5.py` currently applies the adapter but discards `domain_logits`.
   - Compute `F.cross_entropy(domain_logits, domain_id)` and append it as `out[7]`.

4. **`eval_variable_views.py` n_views handling:**
   - The current draft passes `n_views=0` to `_build_omniview_v5_model`, relying on the loaded config. Ensure the model constructor accepts `n_views=0` or pass the real view count from the dataset.

## 10. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| GRL destabilizes a warm-started v47 checkpoint. | Start with `v48_dg_grl_lambda=0.01`; freeze backbone for 1 epoch; ramp lambda if discriminator accuracy stays near chance. |
| `dataset_id` and `domain_id` are conflated. | Use `dataset_id` in the loader/trainer and `domain_id` only inside `OmniMultiViewFusionV5.forward`; keep the names consistent. |
| DDWL over-weights a single hard domain. | Clamp weights to `[0.5, 2.0]` and use temperature `>= 2.0` (already implemented). |
| 3DPW actual (`V=1`) is over-dropped. | Use `v48_dropout_per_domain["5"] = 0.15` and ensure `min_views=1` for actual-mode sequences. |
| Return-tuple ordering confuses future modules. | Always keep the 8-tuple layout: `(pred, weights, visibility, L, epi_loss, entropy, budget, v48_domain_loss)`. |

## 11. Files expected to be touched downstream

| File | Agent | Change |
|------|-------|--------|
| `motionflow_mv/fusion/domain_adapter_v48.py` | Agent-05 | Keep module; no API changes required. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Agent-06 | Rename flag, pass adapter args, return domain loss. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Agent-07 | Already drafted; verify DDWL + per-domain dropout + CLI flags. |
| `experiments/eval_variable_views.py` | Agent-11 | Already drafted; verify per-dataset output and `n_views` handling. |
| `configs/benchmark_v48_domain_smoke.yaml` | Agent-08 | Add v48 flags. |
| `scripts/run_v48_domain_smoke_local_4090.sh` | Agent-09 | Already drafted; use `use_v48_domain_generalization`. |
| `tests/test_domain_adapter_v48.py` | Agent-10 | Verify integration once flag is renamed. |
| `docs/proposals/v48_domain_generalization.md` | Agent-13 | User guide already drafted. |
