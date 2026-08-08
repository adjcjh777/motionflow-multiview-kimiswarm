# v33 Design Proposal: Domain-adaptive / Curriculum View Selection

**Slug:** `domain_view_curriculum`  
**Scope:** MotionFlow-MultiView v33 (next-iteration prototype)  
**Target downstream:** ICRA/CVPR 2027 multi-view pose pipeline  

## 1. Problem Statement and Motivation

Current v31/v32 variable-view training already improves generalisation, but it treats every domain the same way:

*   H36M has **4 real cameras**, MPI-INF-3DHP has **14 real cameras**, and the mixed loader pads them to the same 14-view tensor.
*   The v32 `--domain_aware_view_curriculum` flag only *clamps* the random subset size to the real camera count (`configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml` → `dataset_id 0/1`). It does not learn how many views each domain actually needs, nor which views are geometrically strongest.
*   With `--use_variable_view_training` the curriculum is purely epoch-based (`variable_view_curriculum_alpha`), ignoring per-domain difficulty and camera geometry.

**Goal for v33:** make the view-selection curriculum **domain-conditional** and **learnable**. The model should decide, per domain and per sample, how many views to keep and which views to down-weight, while remaining robust to variable-view inference (2–14 views).

## 2. Proposed Architecture Changes

### 2.1 New module: `DomainAdaptiveViewCurriculum`

Location: `motionflow_mv/fusion/domain_adaptive_view_curriculum_v33.py`

```python
class DomainAdaptiveViewCurriculum(nn.Module):
    """Learns a per-domain view-count policy and per-view relevance score.

    Inputs
    ------
    feat : (B*T, V, J, d)
    domain_emb : (B*T, d)          # from existing domain_embedding
    cameras : K, R, t               # optional geometry features

    Outputs
    -------
    target_k : (B*T,) int-ish       # recommended number of active views
    temperature : (B*T,)            # Gumbel-softmax temperature per sample
    view_score : (B*T, V, J)        # per-view per-joint relevance
    """
```

Key components:

| Component | Role |
|-----------|------|
| `domain_embedding` | Existing `nn.Embedding(num_domains, d)` in `omniview_fusion_v5.py` (line 312). Re-used, not duplicated. |
| `view_policy_head` | 2-layer MLP on `[feat_pooled; domain_emb]` → `target_k` logits + `temperature` offset. |
| `view_score_head` | Cross-view attention (V×V) conditioned on `domain_emb` to produce per-view, per-joint scores. |
| `domain_budget_loss` | MSE between mean selected views and domain-aware target; stronger regularisation for H36M (fewer views) vs MPI (more views). |

### 2.2 Model integration in `OmniMultiViewFusionV5`

In `motionflow_mv/fusion/omniview_fusion_v5.py`:

1. Add constructor flag:
   ```python
   use_domain_adaptive_view_curriculum_v33: bool = False,
   v33_view_policy_hidden: int = 128,
   v33_min_views: int = 2,
   v33_max_views: int = 14,
   v33_budget_weight: float = 0.01,
   ```

2. Wire it next to the existing `AdaptiveViewSelector` (`motionflow_mv/fusion/adaptive_view_selector.py`). The new curriculum replaces the *fixed* `target_k` and `temperature` of `AdaptiveViewSelector` with domain-conditional ones.

3. In `forward()`:
   * If `use_domain_adaptive_view_curriculum_v33` is True, pass `domain_id` into `DomainAdaptiveViewCurriculum`.
   * Use the returned `target_k` and `view_score` to modulate triangulation weights before the DLT/triangulation step.
   * Add `v33_budget_weight * domain_budget_loss` to the existing `epi_loss`.

### 2.3 Training-script changes

In `experiments/train_omniview_fusion_v5_webbridge_multi.py`:

* Add CLI flags:
  ```python
  parser.add_argument("--use_domain_adaptive_view_curriculum_v33", action="store_true")
  parser.add_argument("--v33_view_policy_hidden", type=int, default=128)
  parser.add_argument("--v33_min_views", type=int, default=2)
  parser.add_argument("--v33_max_views", type=int, default=14)
  parser.add_argument("--v33_budget_weight", type=float, default=0.01)
  ```
* Keep `--use_variable_view_training`, `--domain_aware_view_curriculum`, and `--use_domain_embedding` enabled, because the v33 module depends on `dataset_id`.
* The existing variable-view sampling code (lines 919–963) stays; the new module only refines the *selection policy* on top of the sampled mask.

## 3. Training Command / Ablation Flags

### Recommended smoke test (RTX 4090)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
  --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
  --use_domain_embedding \
  --use_variable_view_training \
  --domain_aware_view_curriculum \
  --use_domain_adaptive_view_curriculum_v33 \
  --v33_budget_weight 0.01 \
  --variable_view_min_views 2 --variable_view_max_views 14 \
  --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 \
  --variable_view_permute \
  --d 64 --residual_hidden 128 --n_st_layers 2 --graph_num_layers 1 \
  --epochs 5 --batch_size 8 --train_samples 1000 --val_stride 10 \
  --output outputs/omniview_fusion_v33_domain_view_curriculum_smoke.pth
```

### Ablation matrix

| Run | Flags | Purpose |
|-----|-------|---------|
| `v33_baseline` | `--use_variable_view_training --domain_aware_view_curriculum` | Current v32 best practice. |
| `v33_domain_policy_only` | baseline + `--use_domain_adaptive_view_curriculum_v33 --v33_budget_weight 0.01` | Full proposal. |
| `v33_no_geometry_score` | baseline + curriculum but `view_score_head` disabled | Isolate domain-policy vs. per-view scoring. |
| `v33_fixed_target_k` | baseline + curriculum but `target_k` fixed to domain mean | Check whether adaptive k matters. |

## 4. Expected Metrics and Baseline to Beat

### Primary metrics

| Metric | How computed | Target |
|--------|--------------|--------|
| `val_MPJPE` | `eval_metric()` in `train_omniview_fusion_v5_webbridge_multi.py` (mm) | **Beat v32 baseline** (recorded v32 runs ~20–40 mm on smoke; full A800 target < 28 mm). |
| `val_MPJPE_h36m` | Per-domain validation subset (`dataset_id == 0`) | Reduce gap vs. MPI; H36M currently degrades with 14-view padding. |
| `val_MPJPE_mpi` | Per-domain validation subset (`dataset_id == 1`) | Maintain or improve. |
| `mean_active_views` | Average selected views reported per domain | H36M ≈ 2–4, MPI ≈ 6–10; avoid collapse to 2 views. |
| `view_selection_consistency` | Std of per-sample active views within a domain | Lower is better (stable policy). |

### Robustness metrics

Using the existing variable-view path, evaluate at inference with fixed view masks:

```python
# Pseudo-code; re-use existing view_mask injection in eval.
for k in [2, 4, 8, 14]:
    mask = sample_k_views(k)
    mpjpe = evaluate(view_mask=mask)
```

Target: v33 should show flatter MPJPE vs. k curve, especially for H36M where fewer views are real.

### Baseline to beat

*   **v32 domain-aware curriculum** (`scripts/launch_v32_a800_queue.py` run `"v32_domain_aware_view_curriculum"`).
*   **v30/v31 hierarchical encoder baseline** (`--use_hierarchical_multiview_v30 --v30_n_part_layers 2`).

## 5. Risks / Unknowns

1. **Selection collapse.** The domain policy head may learn to always select the minimum `target_k = 2`, trading accuracy for an easy budget loss. Mitigation: add a monotonic regulariser (existing `--monotonic_loss_weight`) and a per-domain minimum view floor.
2. **Domain embedding overfit.** With only two domains, `nn.Embedding(2, d)` can overfit. Mitigation: share the domain embedding with the existing `use_domain_embedding` path and freeze it during the first N epochs.
3. **Mixed-batch interactions.** MPI samples have 14 real views, H36M only 4. The policy must respect the padded mask; otherwise it may select non-existent H36M views (indices 4–13). The existing `domain_aware_view_curriculum` clamp handles this at the data level; v33 must enforce the same clamp inside `DomainAdaptiveViewCurriculum`.
4. **Gumbel temperature tuning.** Domain-conditional temperature adds another hyperparameter. A good default is `temperature = base * sigmoid(logit_offset)` with base `0.5`.
5. **Compute overhead.** Cross-view attention over `V=14` views is cheap (`V²` is only 196), but on the A800 full run with `batch_size=16` it still adds memory. Smoke test first, then profile with `d=64`.

## 6. Success Criteria (Go/No-Go)

*   Smoke test completes and `val_MPJPE` is within 5% of the v32 baseline on the same smoke config.
*   Full A800 run shows **per-domain active-view counts that differ meaningfully** (H36M < MPI).
*   Variable-view robustness curve is flatter than baseline by at least 10% AUC.

## 7. Files Touched (if/when implemented)

*   `motionflow_mv/fusion/domain_adaptive_view_curriculum_v33.py` (new)
*   `motionflow_mv/fusion/omniview_fusion_v5.py` (add flag + wiring)
*   `experiments/train_omniview_fusion_v5_webbridge_multi.py` (add CLI flags + pass `domain_id`)
*   `docs/swarm_iter_next/v33_domain_view_curriculum.md` (this proposal)

---

*This proposal does not modify any existing source files; it only describes the intended v33 integration path.*
