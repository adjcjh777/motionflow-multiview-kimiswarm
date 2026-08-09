# v51 Dynamic View Selection Policy

**Focus area:** `dynamic_view_selection_policy`  
**Tracking issue:** `#177` (proposed)  
**Depends on:** v46 sparse-view generalization, v50 self-evolution feedback head  

## 1. Module name and file path

`DynamicViewSelectionPolicyV51` → `motionflow_mv/fusion/dynamic_view_selection_policy_v51.py`

## 2. Architecture

A lightweight policy head that decides, on-the-fly, which camera subset to keep for the next triangulation step. Instead of applying the static random dropout of v46 or relying only on per-view reliability scores, the v51 policy predicts a **differentiable view-selection mask** from the current pose estimate, geometric residuals, and the v50 reliability vector. This makes view selection an explicit, learnable part of the self-evolution loop: the model proposes a subset, triangulates, measures residual improvement, and refines the policy.

**Inputs (per frame / clip):**

- 2-D keypoints and camera parameters
- Current 3-D pose estimate from v46/v50
- Per-view reprojection residual from the v50 SEFH
- v46 reliability logits
- Domain embedding (from v48, if available)

**Policy head:**

- Two-layer MLP (`hidden=64`) → per-view logit
- Gumbel-softmax straight-through estimator to produce a binary-ish keep mask
- Hard constraint: at least `v51_dvsp_min_views` kept (enforced by top-k + padding)
- Output: final selected view subset, fed back into the triangulation / v50 refinement step

**Modes:**

- **Training:** stochastic Gumbel-softmax sampling with entropy regularization
- **Inference:** deterministic greedy selection or fixed top-k

The module is identity-at-init in the sense that, when disabled, the pipeline falls back to standard v46 dropout. When enabled, it replaces random dropout with a learned policy during training and uses greedy selection at inference.

## 3. Config flags with defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_dynamic_view_selection_policy` | bool | `False` |
| `v51_dvsp_hidden` | int | `64` |
| `v51_dvsp_num_layers` | int | `2` |
| `v51_dvsp_min_views` | int | `2` |
| `v51_dvsp_top_k` | int | `3` |
| `v51_dvsp_gumbel_temperature` | float | `0.5` |
| `v51_dvsp_policy_loss_weight` | float | `0.01` |
| `v51_dvsp_entropy_regularizer` | float | `0.001` |
| `v51_dvsp_use_reliability_input` | bool | `True` |
| `v51_dvsp_inference_mode` | str | `"greedy"` |

## 4. Loss term

```text
L_dvsp = λ_policy · [ MPJPE_subset - MPJPE_full ] + λ_entropy · H(π)
```

where `MPJPE_subset` is the pose error of the selected view subset (differentiable via Gumbel-softmax straight-through) and `H(π)` is the policy entropy. The first term rewards subsets that recover the full-view accuracy; the entropy term prevents collapse to a single preferred subset. A small auxiliary reprojection-residual reduction term can optionally be added to reward views that lower geometric residual.

## 5. Evaluation metric

- `MPJPE@k` for `k = 2, 3, 4, full`
- **View budget curve**: `MPJPE` vs. average kept views at inference
- **Policy efficiency**: average number of views used at inference time
- **Spearman(policy score, residual reduction)** target `> 0.3`

## 6. Expected MPJPE impact

| View setting | Expected change |
|---|---|
| `MPJPE@2` | −2 to −4 mm |
| `MPJPE@3` | −1 to −2 mm |
| `MPJPE@full` | ±0.3 mm |

By selecting better-than-random subsets in the 2–3 view regime, v51 should close the gap between aggressive sparse-view evaluation and the full-view baseline while reducing average inference cost.

## 7. Main risk and mitigation

**Risk:** Policy collapse to a small fixed subset, or Gumbel-softmax straight-through gradients destabilizing training.

**Mitigation:**

- Enforce `min_views=2` with top-k + random padding during training
- Clamp entropy regularizer to prevent over-exploration
- Anneal Gumbel temperature from `1.0` to `0.1` over first epoch
- Start smoke with `v51_dvsp_policy_loss_weight=0.001` before committing to `0.01`
- Identity fallback: when disabled, v46 random dropout is unchanged

## 8. Paper-story fit

v51 extends the self-evolution narrative from *"knowing which views to trust"* (v37/v39/v50) to *"deciding which views to even use"*. It directly supports the v49 deployment story of dynamic view budgets for real-time streaming, and it naturally couples with the sparse-view / cross-domain frontier by learning to adapt the camera set to the current pose, domain, and residual evidence.
