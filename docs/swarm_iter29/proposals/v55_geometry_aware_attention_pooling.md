# v55 Geometry-Aware Attention Pooling (GAAP)

**Tracking issue:** #185  
**Base branch:** `v55-gaap`  
**Depends on:** v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT, v53-PSC, v54-PSC-v2

## 1. Module name and one-line purpose

**Module:** `GeometryAwareAttentionPoolingV55` → `motionflow_mv/fusion/geometry_aware_attention_pooling_v55.py`

**One-liner:** Refine the per-view per-joint feature tokens that feed into v52 Uncertainty-Weighted Triangulation via a lightweight, geometry-biased cross-view attention, so that each view can borrow evidence from geometrically consistent neighboring views while preserving the v52 baseline at initialization.

## 2. Where it sits in the `OmniMultiViewFusionV5` forward pass

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → per-view joint tokens/rays (B, V, J, C) and pred_3d_init
    ↓
v46-SVG dropout / v50-SEFH / v51-CDSVR per-view feature/reliability heads
    ↓
**v55 Geometry-Aware Attention Pooling** → refined per-view tokens (B, V, J, C)
    ↓
v52 Uncertainty-Weighted TriangulationV52 → pred_3d_uwt
    ↓
v53/v54 physical-space calibration → final pose
```

GAAP is inserted **after all per-view feature/reliability heads** (v46/v50/v51) and **immediately before v52 UWT**. It does not replace any prior block; it only re-weights the per-view joint tokens that v52 uses to compute uncertainty and triangulate.

## 3. Inputs, outputs, and shapes

### Inputs

| Tensor | Shape | Description |
|---|---|---|
| `view_tokens` | `(B, V, J, C)` | Per-view per-joint feature tokens from the preceding per-view blocks. |
| `pred_3d_init` | `(B, J, 3)` | Rough triangulated 3D pose from v25/v45 geometry fusion; used to compute the geometry bias. |
| `K` | `(B, V, 3, 3)` | Camera intrinsics. |
| `R` | `(B, V, 3, 3)` | Camera rotations. |
| `t` | `(B, V, 3)` | Camera translation. |
| `view_mask` | `(B, V)` | Boolean mask of valid views (already applied by v46 dropout). |
| `reliability` (optional) | `(B, V, J)` | Per-view per-joint reliability from v51 CDSVR, used as an additional attention prior. |

### Outputs

| Tensor | Shape | Description |
|---|---|---|
| `refined_tokens` | `(B, V, J, C)` | Same shape as `view_tokens`; identity-at-init by design. |
| `gaap_weights` | `(B, V, J, V)` | Cross-view attention weights for each target view `v` and joint `j`. Uniform at init. |
| `gaap_loss` | scalar | Optional entropy/sparsity regularizer on the attention weights. |

## 4. Architecture

### Per-joint cross-view attention block

For each joint `j` and target view `v`, define a query from the target token and keys/values from all views:

- **Query:** `q_v = W_q · view_tokens[:, v, j]`
- **Key/Value:** `k_u = W_k · view_tokens[:, u, j]`, `value_u = W_v · view_tokens[:, u, j]`
- Multi-head attention with `n_heads` heads.

The attention logits are

```text
logit(v, u) = (q_v · k_u) / sqrt(d_k) + gamma * geometry_bias(v, u)
```

where `gamma` is a learnable scalar initialized to **0.0**, so the geometry term is disabled at initialization. A missing view is masked to `-inf` before softmax.

### Geometry bias

Two supported modes selected by a config flag:

- **`epipolar`:** `geometry_bias(v, u) = - || epipolar_distance(P_v, P_u, pred_3d_init) ||`
- **`ray_angle`:** `geometry_bias(v, u) = - | π - angle(ray_v, ray_u) |`, computed at each joint using the camera centers and the 3D anchor.

The bias is scaled by a learned temperature initialized to **1.0** and is clamped to avoid extreme values.

### Optional reliability prior

If `v55_gaap_use_reliability=True`, the v51 CDSVR reliability score `r_u` is added as a logit bias: `logit(v, u) += log(r_u + ε)`. At init this has negligible effect because reliability is near uniform; it becomes useful once v51 has trained.

### Residual output and gate

```text
attn_out = MultiHeadAttn(view_tokens)
refined_tokens = view_tokens + sigmoid(gate) * W_o · attn_out
```

- `W_o` is zero-initialized.
- `gate` is initialized to **-6.0**, so `sigmoid(gate) ≈ 0.0025` and the residual is effectively zero at init.
- This guarantees `refined_tokens  view_tokens`, so loading a v54 checkpoint with v55 enabled does not change the v52/v53/v54 pipeline.

### Optional refinement MLP

A single-layer MLP with LayerNorm and dropout (`v55_gaap_dropout=0.1`) on the attended output before the residual; its final layer is also zero-initialized.

### Loss

`gaap_loss` is an optional attention-entropy regularizer:

```text
L_gaap = -v55_gaap_loss_weight * sum_j mean_v entropy(gaap_weights[:, :, j, :])
```

- It is added to the total loss only after `v55_gaap_warmup_epochs`.
- At init the weights are uniform, so the entropy is high and the loss is near zero; as training proceeds, the loss gently encourages peaked, interpretable cross-view attention.

### Identity-at-init summary

| Component | Init strategy |
|---|---|
| Output projection `W_o` | Zero. |
| Residual gate | Logit `-6.0`. |
| Geometry scalar `gamma` | `0.0`. |
| Temperature | `1.0` (soft, uniform effect at init). |
| Reliability logit scale | Zero if used. |
| Result | `refined_tokens == view_tokens` to within numerical error. |

## 5. Expected MPJPE impact and main risks

| View setting | Expected impact |
|---|---|
| Full views | `−0.4` to `−1.2 mm` by suppressing noisy or occluded per-view tokens. |
| Sparse views (`@2`, `@3`) | `−1.0` to `−2.5 mm`; borrowing geometrically consistent evidence is most helpful when few views remain. |

### Main risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| **Geometry bias dominates too early** | Attention collapses to a single view and loses feature diversity. | `gamma` starts at 0 and is softly ramped; temperature prevents sharp logits. |
| **Over-smoothing across views** | Per-view discriminative cues average away; MPJPE rises. | Keep attention heads small (`n_heads=4`, `hidden=64`), single layer, gated residual. |
| **Conflicts with v52 UWT weight learning** | Both modules try to re-weight views; double counting. | GAAP refines *features*, v52 refines *triangulation weights*; separate responsibilities. Ablate `use_v55_gaap` with v52 disabled to check. |
| **OOM from V×V attention** | `V` up to ~16, `J` up to ~17; memory is small but batched over clips. | The attention is `O(V^2 J C)` with small constants; clip length is already limited by v47. |
| **Identity-at-init failure** | v54 checkpoint regresses by `>0.1 mm` when v55 is enabled. | Enforce zero-init output, gate `-6.0`, and `gamma=0`; unit test `||refined_tokens - view_tokens||_∞ < 1e-4`. |

## 6. Smoke acceptance criteria

Run `bash scripts/run_v55_geometry_aware_attention_pooling_smoke_local_4090.sh` on the local RTX 4090.

1. **`val_MPJPE@full` within 1 mm** of the v54-PSC-v2 baseline on the same smoke config.
2. **No NaN, Inf, or OOM** through at least one full epoch.
3. **Identity-at-init:** loading the best v54 checkpoint with `use_v55_geometry_aware_attention_pooling=True` and zero gradient steps changes `val_MPJPE` by `< 0.1 mm`.
4. **Attention uniformity at init:** for enabled views, `max_v,u |gaap_weights - 1/V| < 0.05` before the first optimizer step.
5. **Geometry bias sanity:** all computed biases are finite; masked view pairs have weight `0`.
6. **`MPJPE@2` and `MPJPE@3` are not worse** than the v54 baseline; an improvement of ≥ 0.5 mm is preferred.

## 7. Required new files and files to modify

### New files

- `motionflow_mv/fusion/geometry_aware_attention_pooling_v55.py` — `GeometryAwareAttentionPoolingV55` module.
- `configs/benchmark_v55_geometry_aware_attention_pooling_smoke.yaml` — smoke config copied from v54 smoke with v55 flags enabled.
- `scripts/run_v55_geometry_aware_attention_pooling_smoke_local_4090.sh` — smoke launch script that warm-starts from the best available v54 checkpoint.
- `tests/test_geometry_aware_attention_pooling_v55.py` — unit tests for identity-at-init, attention uniformity, masking, and gradient flow.

### Files to modify

- `motionflow_mv/fusion/omniview_fusion_v5.py` — add constructor flag, instantiate `GeometryAwareAttentionPoolingV55` when enabled, insert the call **after v51/v50 per-view heads and before v52 UWT**, and add `gaap_loss` to `epi_loss`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — aggregate `loss_dict["v55_gaap"]` with `v55_gaap_loss_weight` and `v55_gaap_warmup_epochs`.
- `scripts/launch_v33_a800_queue.py` — add an A800 full-run entry `v55_geometry_aware_attention_pooling_on_v54`.

### Proposed config flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_geometry_aware_attention_pooling` | bool | `False` | Master toggle. |
| `v55_gaap_hidden` | int | `64` | Hidden/attention dimension. |
| `v55_gaap_n_heads` | int | `4` | Number of attention heads. |
| `v55_gaap_n_layers` | int | `1` | Number of stacked attention/MLP layers. |
| `v55_gaap_geometry_bias_type` | str | `"epipolar"` | `"epipolar"` or `"ray_angle"`. |
| `v55_gaap_use_reliability` | bool | `True` | Use v51 CDSVR reliability as an attention prior. |
| `v55_gaap_gate_init` | float | `-6.0` | Residual gate logit at init. |
| `v55_gaap_gamma_init` | float | `0.0` | Geometry bias scalar at init. |
| `v55_gaap_temperature` | float | `1.0` | Softmax temperature for geometry bias. |
| `v55_gaap_dropout` | float | `0.1` | Dropout on the refinement MLP. |
| `v55_gaap_loss_weight` | float | `0.001` | Weight of the attention-entropy regularizer. |
| `v55_gaap_warmup_epochs` | int | `0` | Epochs before `gaap_loss` contributes to total loss. |

## Notes

- Keep the module **strictly optional**: `OmniMultiViewFusionV5` must load and run when `use_v55_geometry_aware_attention_pooling=False`.
- Do not start v55 smoke until the v54-PSC-v2 smoke has produced a stable baseline checkpoint.
- If GAAP conflicts with v52 UWT weight learning, an ablation should disable `v52_uwt_use_geometry_bias` and keep GAAP's geometry term.
