# v52 Adaptive Sparse-View Dropout (ASVD)

**Status:** Proposal  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #186  
**Depends on:** #181 (v51 Cross-Domain Sparse-View Reliability), #175 (v49-Lite causal temporal)  

---

## 1. Summary

v52 Adaptive Sparse-View Dropout (ASVD) replaces the fixed-rate v46 view dropout with a **learned, per-(view, joint) dropout gate**. It decides which cameras to keep based on per-joint feature quality, upstream reliability from v51, and an adaptive view budget. At initialization it keeps every available view, so it is warm-startable and identity-like. During training it learns to discard noisy or redundant views; at inference it supports any user-specified camera budget.

---

## 2. Motivation

v46 drops views uniformly at random, ignoring that real captures have **structured redundancy**: some cameras are occluded for specific joints, some baselines are bad for certain poses, and studio vs. in-the-wild domains tolerate different view counts. ASVD makes the multi-view fusion/calibration stage of the paper pipeline input-adaptive: after per-view 2-D pose extraction and before triangulation, the model selects the most informative views per joint.

---

## 3. Architecture

```text
Input feature tokens  (B, T, V, J, C)
        |
[ v51 CDSVR reliability  r ∈ (0,1)^(B,T,V,J) ]
        |
        ▼
[ ASVD Score Network ]
        |
        ├── Per-(view,joint) keep score  s_vj  (MLP over feat + reliability)
        ├── Per-(time,joint) view budget   K_tj  (pooled MLP → Gumbel top-K)
        └── Differentiable keep mask       m_vj ∈ {0,1}  (straight-through)
        |
        ▼
Triangulation weights  w_vj ← w_vj * m_vj
```

### 3.1 Score network

For each token at `(b, t, v, j)`:

```
z_vj = MLP_score( concat( f_vj, r_vj ) )       # (B,T,V,J,H)
s_vj = Linear_score( z_vj )                    # (B,T,V,J)
p_vj = sigmoid( s_vj )                         # per-view keep probability
```

### 3.2 Budget network

```
h_t    = mean_{v,j}( z_vj )                    # (B,T,H)
K_logits_t = MLP_budget( h_t )                 # (B,T, K_max)
K_t    = GumbelSoftMax( K_logits_t, τ )          # one-hot over {k_min, ..., V}
```

At init, `MLP_score` and `MLP_budget` are zeroed so the budget selects all views and `p_vj = 0.5`, making the keep mask the identity.

### 3.3 Differentiable top-K selection

For budget `K_t`, set the `K_t` largest `p_vj` to `1` and the rest to `0`:

```
m_vj = STE_topK( p_vj ; K_t )
w'_vj = m_vj * w_vj
```

---

## 4. Module API

```python
class AdaptiveSparseViewDropoutV52(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64,
                 min_views: int = 2, temperature: float = 0.5):
        ...

    def forward(
        self,
        feat: torch.Tensor,          # (B, T, V, J, C)
        reliability: torch.Tensor,     # (B, T, V, J) from v51/v46; default 1.0
        view_mask: torch.Tensor,     # (B, T, V) bool
    ) -> torch.Tensor:
        """Return differentiable keep mask (B, T, V, J) in {0,1}."""
        ...
```

---

## 5. Training Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v52_adaptive_sparse_view_dropout` | bool | `False` | Master switch |
| `v52_asvd_hidden` | int | `64` | Hidden dimension of score/budget MLPs |
| `v52_asvd_min_views` | int | `2` | Minimum retained views |
| `v52_asvd_temperature` | float | `0.5` | Gumbel-SoftMax temperature |
| `v52_asvd_warmup_epochs` | int | `1` | Epochs with identity keep mask |
| `v52_asvd_input_reliability` | str | `"v51"` | Source of reliability (`"v51"`, `"v46"`, `"v37"`, `"none"`) |

---

## 6. Expected MPJPE Impact

- **Full views**: no regression by construction; identity initialization preserves the v51 baseline.
- **Sparse views (2–4)**: 5–10% lower `MPJPE@k` than v46 by keeping informative views and discarding occluded/noisy ones.
- **Cross-domain**: domain-agnostic budget features should reduce the 3DPW↔studio gap when paired with v48/v51.

---

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Straight-through top-K gradient bias | Anneal `v52_asvd_temperature`; add entropy regularizer on `p_vj` |
| Budget collapses to `min_views` | Initialize budget to all views; freeze ASVD for `v52_asvd_warmup_epochs` |
| Double masking with v51 reliability | Treat v51 as soft weights; ASVD is the hard selector |
| Inference budget mismatch | Expose a runtime `max_views` override in trainer/eval |
| Domain-specific overfitting | Apply ASVD after domain-agnostic normalization |

---

## 8. Five-Step Implementation Plan

1. Add `motionflow_mv/fusion/adaptive_sparse_view_dropout_v52.py`.
2. Wire it into `OmniMultiViewFusionV5` after the v51 reliability head and before triangulation weights, guarded by `use_v52_adaptive_sparse_view_dropout`.
3. Add the six flags to `experiments/train_omniview_fusion_v5_webbridge_multi.py`; pass the chosen upstream reliability tensor into ASVD.
4. Create `configs/benchmark_v52_adaptive_sparse_view_dropout_smoke.yaml` and `scripts/run_v52_adaptive_sparse_view_dropout_smoke_local_4090.sh`.
5. Extend `experiments/eval_variable_views.py` to report `MPJPE@k` for `k = 2,3,4,full`.

---

## 9. Paper Story Fit

ASVD strengthens the paper narrative *multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized MotionFlow pipeline*. The fusion/calibration stage now contains an adaptive, reliability-aware camera selector, showing that the optimized pipeline reasons about which views are worth fusing.
