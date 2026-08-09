# v52 — Learned View Selection Policy

**Author:** design-swarm v52  
**Status:** proposal  
**Tracking:** issue #184 (proposed)  
**Depends on:** v25/v45 geometry fusion, v46 sparse-view generalization, v51 cross-domain sparse-view reliability

---

## 1. Motivation

The MotionFlow-MultiView pipeline already extracts per-view 2-D joints, fuses geometry (v25/v45), reasons about sparse-view reliability (v46, v51), and refines poses in physical space (v28/v40). What is still missing is an **explicit learned policy** that decides, for every joint and frame, which subset of cameras should actually contribute to triangulation.

Current fusion modules produce continuous per-view weights, but they do not form a compact, interpretable selection. As a result:

- Noisy/occluded views are down-weighted but still consume attention and DLT compute.
- The model has no principled way to trade accuracy for runtime (e.g., use only 3 of 4 cameras when the fourth is unreliable).
- Existing v46/v51 reliability heads operate independently; a policy head can combine geometry, appearance, and reliability cues into a single decision boundary.

The v52 module introduces a **differentiable learned view-selection policy** that learns to mask views before triangulation. It is warm-startable/identity-at-init, so it can be dropped into `OmniMultiViewFusionV5` without changing existing checkpoints.

---

## 2. Architecture

Module: `motionflow_mv/fusion/learned_view_selection_policy_v52.py`  
Class: `LearnedViewSelectionPolicyV52`

### 2.1 Inputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `feat` | `(B, T, V, J, C)` | Multi-view feature tokens produced by the ST transformer |
| `K` | `(B, T, V, 3, 3)` | Calibrated intrinsics |
| `R` | `(B, T, V, 3, 3)` | Camera rotations |
| `t` | `(B, T, V, 3)` | Camera translations |
| `view_mask` | `(B, T, V)` | Binary mask of valid views |
| `reliability` (optional) | `(B, T, V, J)` | Reliability from v46 and/or v51 CDSVR |

### 2.2 Per-view geometry embedding

Flatten intrinsics/extrinsics and encode them with a small MLP:

```
g_v = MLP([vec(K_{b,t,v}), vec(R_{b,t,v}), t_{b,t,v}])   # (B, T, V, d_geo)
```

Default `d_geo = 32`.

### 2.3 Token assembly and cross-view reasoning

For each `(b, t, v, j)` we build a feature token:

```
z_{b,t,v,j} = LayerNorm( Linear([ feat_{b,t,v,j} ; g_{b,t,v} ; rel_{b,t,v,j} ]) )
```

where `rel` is zero-filled when no reliability head is active. We then apply a lightweight multi-head self-attention block across the view dimension to let each view attend to the others while respecting `view_mask`:

```
h = MHSA(z, view_mask=view_mask)            # (B, T, V, J, d_model)
h = LayerNorm(h + z)
```

Default `d_model = C` (matches feature dimension) with `v52_lvsp_n_heads = 4` and `v52_lvsp_n_layers = 2`.

### 2.4 Selection mask

A per-view/joint MLP predicts a logit per token:

```
α_{b,t,v,j} = MLP(h_{b,t,v,j})             # scalar
m_{b,t,v,j} = 2 · sigmoid( α_{b,t,v,j} / τ )  # (B, T, V, J)
```

**Warm-start / identity-at-init:** the final linear layer of the MLP is initialized with zeros, so `α = 0` at the start of training and `m ≈ 1.0`. The existing triangulation is therefore unchanged at initialization.

Optionally, a hard straight-through selection can be enabled with `v52_lvsp_hard=True`:

```
π_v = softmax( (α_v + gumbel_noise) / τ )      # over V views
k   = floor(v52_lvsp_budget · V)
m_hard = one_hot(argmax_v π_v)                 # forward
m      = m_hard.detach() + (m_soft - m_soft.detach())   # STE backward
```

The default is the continuous sigmoid mask; the straight-through path is kept for ablation.

---

## 3. Integration into `OmniMultiViewFusionV5`

The module is inserted **after** the v46 sparse-view reliability and v51 CDSVR re-weighting and **before** the DLT triangulation.

Current pseudo-code at the triangulation boundary:

```python
if v46_reliability is not None:
    weights = weights * v46_reliability.view(B*T, V, J)
if cdsvr_reliability is not None:
    weights = weights * cdsvr_reliability.view(B*T, V, J)

# DLT
pred_3d_raw = triangulate_dlt_batched_lstsq(points_2d, P, weights)
```

With v52:

```python
selection_mask = self.learned_view_selection_policy_v52(
    feat=feat, K=K_corrected, R=R, t=t,
    view_mask=view_mask, reliability=combined_reliability
)  # (B, T, V, J)
weights = weights * selection_mask.view(B*T, V, J)
```

`combined_reliability` is the element-wise product of available v46/v51 reliability maps, or `None` if neither is active.

---

## 4. Auxiliary losses

- **Budget loss:** encourage the average selected fraction to approach a target `v52_lvsp_target_budget` (default `1.0`, i.e., no sparsity):

```
L_budget = λ_budget · ( mean(m) - b )²
```

A budget warmup schedule keeps `λ_budget = 0` for the first `v52_lvsp_budget_warmup_epochs` epochs so the identity initialization is preserved while the rest of the model warms up.

- **Minimum-view penalty:** if a hard top-k path is used, an auxiliary penalty enforces at least `v52_lvsp_min_views` active cameras.

---

## 5. Config flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v52_learned_view_selection_policy` | bool | `False` | Master toggle |
| `v52_lvsp_hidden` | int | `64` | Hidden dim of the per-token MLP |
| `v52_lvsp_n_layers` | int | `2` | Cross-view transformer layers |
| `v52_lvsp_n_heads` | int | `4` | Attention heads |
| `v52_lvsp_dropout` | float | `0.1` | Dropout in transformer/MLP |
| `v52_lvsp_temperature` | float | `1.0` | Gumbel/sigmoid temperature |
| `v52_lvsp_use_reliability_input` | bool | `True` | Feed v46/v51 reliability as input |
| `v52_lvsp_hard` | bool | `False` | Use straight-through hard selection |
| `v52_lvsp_min_views` | int | `2` | Minimum active views |
| `v52_lvsp_target_budget` | float | `1.0` | Target active-view fraction |
| `v52_lvsp_budget_weight` | float | `0.01` | Weight of budget loss |
| `v52_lvsp_budget_warmup_epochs` | int | `0` | Epochs before budget loss turns on |
| `v52_lvsp_identity_init` | bool | `True` | Zero-initialize final layer so `m≈1` |

---

## 6. Expected MPJPE impact

- **Full-view setting:** suppressing occasional occluded/noisy views is expected to improve H36M val MPJPE by **1–3 mm** relative to the v46 baseline.
- **Sparse-view setting:** on `MPJPE@2`/`MPJPE@3` from v46, a learned policy should close the gap to full-view fusion by **5–10 %**, because it explicitly trains the model to pick the most informative subset.
- **Smoke target:** `< 80 mm` on the standard local smoke config when stacked on v46 + v45.

---

## 7. Risks

See `docs/swarm_iter26/reports/agent_learned_view_selection_policy_risks.md` for a concrete risk/mitigation table.

---

## 8. 5-step implementation plan

1. **Implement the module.** Create `motionflow_mv/fusion/learned_view_selection_policy_v52.py` with the camera-geometry embedding, cross-view transformer, and continuous/hard mask outputs. Ensure zero-initialization for identity-at-init behavior.
2. **Wire into `OmniMultiViewFusionV5`.** Add the config flags in `__init__`, instantiate the module, and call it in `forward` right before DLT triangulation. Gate the budget loss with `budget_warmup_epochs`.
3. **Add smoke config/script.** Write `configs/benchmark_v52_learned_view_selection_smoke.yaml` and `scripts/run_v52_learned_view_selection_smoke_local_4090.sh`.
4. **Smoke and identity check.** Run the smoke test on RTX 4090. Verify that with `use_v52_learned_view_selection_policy=True` but at initialization the MPJPE is unchanged compared to the v46 baseline, confirming warm-start behavior.
5. **Ablate and queue.** Vary `v52_lvsp_target_budget` and `v52_lvsp_temperature` on H36M val. If smoke is clean, add the full A800 entry to `scripts/launch_v33_a800_queue.py` and update `docs/swarm_iter26/status.md`.
