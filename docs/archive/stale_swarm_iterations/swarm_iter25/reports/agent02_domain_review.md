# Agent-02: v41 Domain Code Review and v48 Adapter Design

**Owner:** Agent-02  
**Branch:** `v48-domain`  
**Tracking issue:** #164  
**Date:** 2026-08-09

## Executive summary

This report reviews the existing domain-aware machinery that v48 can build on:

1. A learnable **domain embedding** in `OmniMultiViewFusionV5` (simple additive embedding, currently only used when `--use_domain_embedding` is set).
2. A **static per-domain MSE weight** in the trainer (`--domain_loss_weights`), used only for H36M (domain 0) and MPI-INF-3DHP (domain 1).
3. A **`DomainBalancedSampler`** that rebalances mixed training epochs across datasets.
4. Two earlier **domain-adaptation wrappers** (`motionflow_mv/models/domain_adaptation_wrapper.py` and `motionflow_mv/fusion/domain_adaptation_wrapper.py`) that implement GRL-based domain discriminators and per-domain FiLM modulation, but are wired to older backbones (`RayAttentionFusionModelTemporalResidual` / `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`) and are not currently used by v5/v46/v47.
5. The **v41 DDWL redesign document** (`docs/v41_domain_loss_redesign.md`), which proposes an adaptive, per-domain, per-joint, per-term loss but has not yet been implemented in the trainer.

**Key finding:** v48 should not invent a new domain module from scratch. The most incremental and reviewable design is to reuse the existing GRL+FiLM skeleton from the legacy wrappers, port the v41 DDWL proposal from design doc to trainer code, and insert the combination as a lightweight `DomainAdapterV48` module after the v46 sparse-view and v47 temporal heads. This keeps v48 as a thin, warm-startable layer on top of v47.

---

## 1. Current v41 domain code inventory

### 1.1 Domain embedding in `OmniMultiViewFusionV5`

**File:** `motionflow_mv/fusion/omniview_fusion_v5.py` (lines 334, 426-428, 1274-1277)

```python
# __init__
self.use_domain_embedding = use_domain_embedding
if self.use_domain_embedding:
    self.domain_embedding = nn.Embedding(num_domains, d)

# forward
if self.use_domain_embedding and domain_id is not None:
    domain_emb = self.domain_embedding(domain_id)  # (B, d)
    feat = feat + domain_emb.view(B, 1, 1, 1, self.d)
```

- The embedding is **additive** to the multi-view feature tokens.
- It is only used when the trainer passes `domain_id` and the flag `use_domain_embedding` is set.
- It is **not combined** with v46 reliability weights or v47 temporal conditioning.

### 1.2 Static per-domain loss weights in trainer

**File:** `experiments/train_omniview_fusion_v5_webbridge_multi.py` (lines 973-978, 1145-1151, 1754-1756)

```python
# setup
domain_loss_weights: Optional[torch.Tensor] = None
if getattr(args, "domain_loss_weights", None) is not None:
    weights = [float(w.strip()) for w in args.domain_loss_weights.split(",")]
    domain_loss_weights = torch.tensor(weights, dtype=torch.float32)

# loss
mse_per_sample = F.mse_loss(pred_3d, y, reduction="none").mean(dim=(1, 2, 3))
if dataset_id is not None and domain_loss_weights is not None:
    domain_sample_weights = domain_loss_weights.to(device)[dataset_id.squeeze(-1).long()]
    loss = (mse_per_sample * domain_sample_weights).mean()
else:
    loss = mse_per_sample.mean()
```

- Weights are **static** for the whole run.
- They apply only to the 3-D MSE term; reprojection, physical, bone-length, and other losses are unweighted.
- They only support the two domains hard-coded in the trainer (H36M=0, MPI-INF-3DHP=1).
- Adding a third domain requires guessing a new scalar.

### 1.3 `DomainBalancedSampler`

**File:** `motionflow_mv/data/domain_balanced_sampler.py`

- Rebalances a `ConcatDataset` so every domain contributes equally to each epoch.
- Assumes inner datasets expose `dataset_name` (e.g. `"h36m"`, `"mpi"`, `"3dpw"`).
- Works independently of the loss weighting; combining the two can over-correct if both are active.

### 1.4 Legacy `DomainAdaptationWrapper`s

**Files:**
- `motionflow_mv/models/domain_adaptation_wrapper.py`
- `motionflow_mv/fusion/domain_adaptation_wrapper.py`

Both implement the same core ideas:

1. **Gradient-reversal layer (GRL):** adversarially train a binary domain classifier on pooled spatio-temporal features.
2. **Per-domain FiLM modulation:** one affine adapter per domain (`"0"` and `"1"`) applied to feature tokens.
3. A **maximum-mean-discrepancy (MMD)** helper in the fusion wrapper.

**Key limitations for v48:**
- They wrap **older backbones** (`RayAttentionFusionModelTemporalResidual` / `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`), not `OmniMultiViewFusionV5`.
- They are **binary** (source vs. target) and hard-coded for two domains.
- They do not use the v46 reliability features or v47 temporal features.
- They are not referenced by the current trainer.

### 1.5 v41 DDWL design doc

**File:** `docs/v41_domain_loss_redesign.md`

Proposes a **domain-difficulty-weighted loss (DDWL)** with:
- Per-domain running EMA of unweighted MSE.
- Adaptive weights: `w_d_raw = (loss_ema / loss_ema.max()) ** (-1 / T)`.
- Optional per-joint importance map.
- Application to multiple loss terms (MSE, reprojection, physical).
- Warmup epoch with uniform weights.

This has **not been implemented** in the trainer.

---

## 2. Gaps that v48 must address

| # | Gap | Why it matters for v48 |
|---|-----|------------------------|
| 1 | **Static weights** cannot adapt to per-domain difficulty. | 3DPW actual/pseudo and studio domains have very different loss scales; a fixed scalar will either under-weight or over-weight them. |
| 2 | **Loss weighting is MSE-only.** | Reprojection and physical priors are also affected by domain noise; they should be reweighted too. |
| 3 | **Only two domains supported.** | v48 must handle at least H36M, MPI, AIST++, 3DPW pseudo, and 3DPW actual (and possibly Shelf/Campus). |
| 4 | **Domain embedding is unconditioned.** | Adding a scalar embedding to features does not produce domain-invariant representations; it just gives the model a domain tag. |
| 5 | **GRL+FiLM wrappers are disconnected from v5/v46/v47.** | They cannot be reused without re-wiring; v48 needs a wrapper that fits `OmniMultiViewFusionV5`. |
| 6 | **No per-dataset view-dropout schedule.** | v46 uses a single global dropout probability; 3DPW actual (V=1) cannot tolerate the same dropout as H36M (V=4). |
| 7 | **No per-domain evaluation metric.** | The current `eval_variable_views.py` reports `MPJPE@k` globally, not per domain. |

---

## 3. Proposed v48 adapter design

### 3.1 Design principles

1. **Build on v46/v47.** v48 is a thin adapter, not a replacement.
2. **Reuse existing code.** Port the v41 DDWL proposal and reuse the GRL+FiLM skeleton from the legacy wrappers.
3. **Keep the API minimal.** The trainer passes `domain_id` to the model; the model returns `domain_logits` only when a flag is set.
4. **Warm-start friendly.** The adapter should be a no-op at initialization so a v47 checkpoint can be loaded first.
5. **Do not break H36M/MPI manifests.** Existing data paths and flags remain unchanged unless v48 is explicitly enabled.

### 3.2 Module: `DomainAdapterV48`

**New file:** `motionflow_mv/fusion/domain_adapter_v48.py`

Insert after v47 temporal aggregation. It takes the v46/v47 output features and returns both refined features and optional domain logits.

```python
class DomainAdapterV48(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_views: int,
        num_domains: int = 6,
        hidden: int = 64,
        dropout: float = 0.1,
        use_grl: bool = True,
        use_film: bool = True,
        grl_lambda: float = 0.1,
    ):
        ...

    def forward(
        self,
        feat: torch.Tensor,        # (B, T, V, J, C)
        dataset_id: torch.Tensor,  # (B,)
        view_mask: torch.Tensor,   # (B, T, V)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return domain-invariant features and optional domain logits."""
        ...
```

**Responsibilities:**
- **Instance normalization:** normalize reliability features per domain to reduce covariate shift.
- **GRL domain discriminator:** adversarially encourage domain-invariant features.
- **Domain-conditional FiLM:** allow the model to adapt to each domain without storing per-domain parameters beyond the FiLM affine.
- **No triangulation logic:** it only transforms features; the existing v25/v45 triangulation path remains unchanged.

### 3.3 Integration into `OmniMultiViewFusionV5`

**File:** `motionflow_mv/fusion/omniview_fusion_v5.py`

Add constructor flags:

```python
use_v48_domain_adapter: bool = False,
v48_domain_adapter_hidden: int = 64,
v48_domain_adapter_grl_lambda: float = 0.1,
v48_domain_adapter_use_film: bool = True,
v48_domain_adapter_num_domains: int = 6,
```

Instantiation (around the v46/v47 blocks):

```python
if self.use_v48_domain_adapter:
    self.domain_adapter_v48 = DomainAdapterV48(
        in_channels=self.d,
        n_views=n_views,
        num_domains=v48_domain_adapter_num_domains,
        hidden=v48_domain_adapter_hidden,
        grl_lambda=v48_domain_adapter_grl_lambda,
        use_film=v48_domain_adapter_use_film,
    )
else:
    self.domain_adapter_v48 = None
```

Forward pass:
- After v47 temporal aggregation (or after the triangulation if v47 is off), optionally apply `domain_adapter_v48` to the feature tokens.
- Return `domain_logits` only when `domain_id` is provided and `use_v48_domain_adapter=True`.

### 3.4 Trainer changes

**File:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`

1. **Implement v41 DDWL** as the default v48 loss weighting:
   - Maintain `loss_ema: Dict[int, float]` and `joint_error_ema: Tensor[num_domains, n_joints]`.
   - Update EMA after burn-in (1 epoch warmup by default).
   - Compute adaptive per-domain weights with temperature `v48_ddwl_temperature` (default 2.0).
   - Apply weights to MSE, reprojection, and physical losses.

2. **Add CLI flags:**
   ```python
   parser.add_argument("--use_v48_domain_adapter", action="store_true")
   parser.add_argument("--v48_domain_adapter_hidden", type=int, default=64)
   parser.add_argument("--v48_domain_adapter_grl_lambda", type=float, default=0.1)
   parser.add_argument("--v48_ddwl_temperature", type=float, default=2.0)
   parser.add_argument("--v48_ddwl_warmup_epochs", type=int, default=1)
   parser.add_argument("--v48_ddwl_apply_to_reproj", action="store_true")
   parser.add_argument("--v48_ddwl_apply_to_physical", action="store_true")
   parser.add_argument("--v48_dropout_per_domain", type=str, default=None)
   ```

3. **Pass `domain_id` to the model** when `use_v48_domain_adapter` is enabled.

4. **Per-dataset view dropout:** when `v48_dropout_per_domain` is provided, override the global v46 dropout probability per `dataset_id`.

### 3.5 Data loader changes

**File:** `motionflow_mv/data/webbridge_mixed_dataset.py`

- Ensure `dataset_id` mapping supports 3DPW actual (domain 5 already reserved).
- The loader already returns `dataset_id`; no new fields are needed.
- Agent-01 is responsible for adding the `actual` mode to the loader.

### 3.6 Evaluation changes

**File:** `experiments/eval_variable_views.py` or `experiments/eval_omniview_fusion_v5_webbridge_multi.py`

- Aggregate `MPJPE@k` per `dataset_id`.
- Report `domain_gap = max(MPJPE@k) - min(MPJPE@k)`.
- Track domain discriminator accuracy; target is within `[0.45, 0.55]`.

### 3.7 Suggested domain IDs

| ID | Domain |
|----|--------|
| 0 | H36M |
| 1 | MPI-INF-3DHP |
| 2 | AIST++ |
| 3 | Shelf |
| 4 | Campus |
| 5 | 3DPW (pseudo + actual) |

If 3DPW pseudo and actual need separate treatment, reserve 6 for `3DPW_actual`.

---

## 4. Recommended scope for the first v48 smoke

1. **Module:** implement `DomainAdapterV48` with GRL+FiLM only (no DDWL) and verify it loads.
2. **Trainer:** add `--use_v48_domain_adapter` and pass `domain_id`.
3. **Loss:** implement DDWL on MSE only for the smoke; extend to reprojection/physical for full run.
4. **Smoke config:** 1-2 epochs, 50-100 samples, mixed H36M/MPI/3DPW pseudo.
5. **Success check:** finite per-domain val_MPJPE, no NaN, and domain discriminator accuracy stays near chance.

---

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| v48 adapter destabilizes v47 warm-start. | Initialize FiLM to identity and GRL lambda to 0.01; freeze v47 weights for 1 epoch. |
| DDWL over-weights a single hard domain. | Clamp domain weights to `[0.5, 2.0]` and use temperature `T >= 2.0`. |
| Domain embedding and FiLM fight. | Make `use_domain_embedding` and `use_v48_domain_adapter` mutually exclusive in the trainer parser. |
| Per-dataset dropout breaks 3DPW actual (V=1). | Cap dropout so `min_views` is always 1 for 3DPW actual. |
| GRL makes features too domain-invariant and hurts studio accuracy. | Start with `grl_lambda=0.01` and increase only if discriminator accuracy stays > 0.55. |

---

## 6. Files expected to be touched by downstream agents

| File | Agent | Change |
|------|-------|--------|
| `motionflow_mv/fusion/domain_adapter_v48.py` | Agent-05 | New module |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Agent-06 | Wire v48 flags |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Agent-07 | DDWL + CLI flags |
| `configs/benchmark_v48_domain_smoke.yaml` | Agent-08 | Smoke config |
| `scripts/run_v48_domain_smoke_local_4090.sh` | Agent-09 | Smoke script |
| `tests/test_domain_adapter_v48.py` | Agent-10 | Unit/integration tests |
| `experiments/eval_variable_views.py` | Agent-11 | Per-domain MPJPE |
| `scripts/launch_v33_a800_queue.py` | Agent-12 | A800 queue entry |
| `docs/proposals/v48_domain_generalization.md` | Agent-13 | Update user guide |
| `AGENTS.md` | Agent-14 | Add v48 conventions |

---

## 7. Open questions

1. Should `DomainAdapterV48` operate on the v46 feature tokens `(B, T, V, J, C)` or on the v47-refined poses `(B, T, J, 3)`? The proposal shows it after v47; operating on features is more expressive, but operating on poses is simpler and warmer-start friendly.
2. Should 3DPW pseudo and 3DPW actual share domain ID 5 or be split into 5 and 6? Splitting makes DDWL more precise but doubles the DDWL state.
3. Should the GRL+FiLM path be enabled together or as separate flags? Separating them lets ablations isolate the contribution of each component.
