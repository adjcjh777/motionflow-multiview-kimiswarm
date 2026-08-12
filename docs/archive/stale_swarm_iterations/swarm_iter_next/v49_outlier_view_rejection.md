# v49: Iterative Self-Critique Outlier-View Rejection

**Slug:** `outlier_view_rejection`  
**Date:** 2026-08-09  
**Target stack:** `OmniMultiViewFusionV5` (v46 sparse-view + v47 temporal + v48 domain)  
**Tracking issue:** #166  
**Labels:** `experiment`, `P1-next`  

---

## 1. Problem statement

The existing v33 outlier-view detector is **single-pass and feed-forward**: it computes one reprojection residual, thresholds it with hand-tuned or learned scales, and down-weights suspicious views before a single re-triangulation. In the v46-v48 stack this is no longer sufficient:

1. **Sparse views amplify false positives.** With v46 view-dropout, rejecting one of only two available views leaves an ill-posed triangulation. The v33 detector does not know how many views are actually present or how reliable the v46 per-view reliability head thinks each view is.
2. **Temporal inconsistency.** v33 decides per-frame. A view that is clean at frame `t` but occluded at `t+1` can only be caught if each frame is processed independently, wasting the v47 temporal head.
3. **Domain shift in residual statistics.** v48 mixes studio (H36M/MPI) and in-the-wild (3DPW) data. Residual distributions differ across domains, so a single learned threshold can be too aggressive for 3DPW and too lenient for H36M.
4. **No feedback into self-evolution.** v37 already predicts per-(view, joint) reliability, but that score is not used by v33; conversely, v33’s hard/soft outlier decisions are not fed back to update v37’s reliability target or v46’s sparse-view reliability head.

**Goal for v49:** turn outlier-view rejection into an **iterative, self-critiquing, sparse-aware component** that fuses v37 reliability, v46 sparse-view weights, and v47 temporal context, while remaining domain-robust via v48.

---

## 2. Proposed approach

v49 replaces the one-shot v33 detector with a small iterative block that runs inside the v25 triangulation path and exchanges information with v46/v47/v48:

```text
v25 initial triangulation  ->  v37 reliability r_vj
         |
         v
[ v49 IterativeOutlierViewRejection ]
    ├── Iterative residual refinement (2-3 Gauss-Newton / DLT passes)
    ├── Fuse v37 reliability + v46 sparse-view weight + reprojection z-score
    ├── Temporal consistency prior from v47 (soft prior on outlier labels)
    └── Domain-robust normalization (v48 per-domain residual whitening)
         |
         v
final clean-view weights  ->  re-triangulation  ->  v47 temporal head
```

### 2.1 Core ideas

1. **Iterative consensus.** Run `v49_n_iters` times: triangulate with current weights, reproject, update outlier weights, re-triangulate. The first pass is the v33-style detector; subsequent passes refine it. Each pass is lightweight because it reuses the same camera parameters.
2. **Reliability fusion.** Combine three sources into a single per-(view, joint) score:
   - `z_score` from reprojection residuals (v33),
   - `r_vj` from v37 self-critique view reliability,
   - `s_v` from v46 sparse-view reliability head.
   The fused outlier weight is:
   ```
   w_vj = sigmoid( - (z_score - z_thresh) * (r_vj * s_v) * beta )
   ```
   At init `r_vj ≈ 1` and `s_v ≈ 1`, so the block starts as the v33 detector.
3. **Sparse-aware guard.** When `n_active_views <= v49_min_views_for_strict` (default 3), clamp the maximum fraction of views that can be rejected per joint (e.g. 25%). This prevents the detector from discarding the last usable views.
4. **Temporal feedback.** The v47 temporal head already produces per-frame pose refinements; v49 additionally asks v47 to output a per-view temporal outlier prior `π_vj(t) = P(view v is outlier at t)`. This prior is added to the v49 fused score. If v47 is disabled, the prior is uniform (no-op).
5. **Domain robustness.** v48’s per-domain FiLM/conditional-BN normalizes the residual features before the v49 MLP, so thresholds do not drift when domains are mixed.
6. **Self-evolution feedback loop.** After the final triangulation, compute the per-view reprojection residual again and treat it as a supervised target for v37 reliability: high residual -> low reliability. This closes the loop: v49 outlier decisions train v37, and v37 reliability improves v49 in the next iteration/epoch.

### 2.2 Where it sits in the pipeline

- v49 runs **after** the first v25 triangulation and **before** the v47 temporal head.
- It consumes the same `pred_3d_raw` that v33 currently uses.
- It produces updated per-view weights that are passed into the adaptive Gauss-Newton step, then into v25 multi-view geometry fusion, and finally into v47/v48.
- If v33 is also enabled, v49 supersedes it (do not run both; issue a warning if both flags are true).

---

## 3. Concrete code-level changes

### 3.1 New module

`motionflow_mv/fusion/outlier_view_rejection_v49.py`:

```python
class IterativeOutlierViewRejectionV49(nn.Module):
    def __init__(
        self,
        z_thresh: float = 3.0,
        soft_beta: float = 1.0,
        n_iters: int = 2,
        min_views_strict: int = 3,
        max_reject_frac: float = 0.25,
        use_v37_reliability: bool = True,
        use_v46_reliability: bool = True,
        use_v47_temporal_prior: bool = True,
        use_v48_domain_norm: bool = True,
        reliability_fusion_hidden: int = 64,
    ):
        ...

    def forward(
        self,
        pred_3d: torch.Tensor,          # (B, T, J, 3)
        points_2d: torch.Tensor,        # (B, T, V, J, 2)
        K: torch.Tensor,                # (B, T, V, 3, 3)
        R: torch.Tensor,                # (B, T, V, 3, 3)
        t: torch.Tensor,                # (B, T, V, 3)
        view_mask: torch.Tensor,        # (B, T, V)
        v37_reliability: Optional[torch.Tensor] = None,   # (B, T, V, J)
        v46_reliability: Optional[torch.Tensor] = None,   # (B, T, V)
        v47_outlier_prior: Optional[torch.Tensor] = None, # (B, T, V, J)
        domain_id: Optional[torch.Tensor] = None,          # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights: (B, T, V, J) final soft outlier weights in [0, 1].
            aux_loss: scalar consistency loss (can be 0).
        """
```

### 3.2 Files to touch

| File | Change |
|------|--------|
| `motionflow_mv/fusion/outlier_view_rejection_v49.py` | New iterative detector + reliability fusion module. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Add v49 flags; instantiate module; call it after v25 triangulation and before adaptive GN; pass v37/v46/v47 tensors. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` / `build_model_from_args` | Add CLI flags and plumb through YAML config. |
| `motionflow_mv/fusion/self_critique_view_reliability_v37.py` | Accept an optional `outlier_feedback` target to update reliability targets. |
| `motionflow_mv/fusion/temporal_aggregation_v47.py` | Optionally output a per-view outlier prior when `v49_use_temporal_prior=True`. |
| `configs/benchmark_v49_outlier_view_rejection_smoke.yaml` | Smoke config. |
| `scripts/run_v49_outlier_view_rejection_smoke_local_4090.sh` | Smoke script. |
| `tests/test_outlier_view_rejection_v49.py` | Unit tests for iterative refinement, sparse-aware guard, and v37/v46/v47 integration. |

### 3.3 New flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_outlier_view_rejection_v49` | bool | `False` | Master switch (mutually exclusive with v33). |
| `v49_outlier_z_thresh` | float | `3.0` | Base robust z-score threshold. |
| `v49_outlier_soft_beta` | float | `1.0` | Softness of exponential down-weighting. |
| `v49_outlier_n_iters` | int | `2` | Number of triangulate -> reject -> re-triangulate passes. |
| `v49_outlier_min_views_strict` | int | `3` | Below this active view count, cap rejection fraction. |
| `v49_outlier_max_reject_frac` | float | `0.25` | Max fraction of active views that can be rejected per joint in sparse mode. |
| `v49_outlier_use_v37_reliability` | bool | `True` | Fuse with v37 self-critique reliability. |
| `v49_outlier_use_v46_reliability` | bool | `True` | Fuse with v46 sparse-view reliability. |
| `v49_outlier_use_v47_temporal_prior` | bool | `True` | Use v47 temporal outlier prior. |
| `v49_outlier_use_v48_domain_norm` | bool | `True` | Use v48 per-domain residual normalization. |
| `v49_outlier_reliability_fusion_hidden` | int | `64` | Hidden dim of the reliability-fusion MLP. |
| `v49_outlier_feedback_weight` | float | `0.01` | Weight of the self-evolution feedback loss to v37. |

---

## 4. Risks / failure modes

| Risk | Why | Mitigation |
|------|-----|------------|
| **Iterative triangulation is too expensive.** Each v49 pass re-runs DLT/GN. | Latency / memory increase. | Cap `v49_outlier_n_iters=2` and reuse the same `P` matrix; profile on smoke. |
| **Sparse-view guard is too conservative/too aggressive.** | At 2 views, any rejection breaks triangulation; at 8 views, a 25% cap may miss true outliers. | Tune `v49_outlier_min_views_strict` and `v49_outlier_max_reject_frac` per dataset. |
| **v37 reliability collapses / saturates.** | If v37 always predicts low reliability, all views are down-weighted. | Initialize v37 bias high; use the feedback weight to keep updates gradual. |
| **v47 temporal prior over-smoothes.** | A static temporal prior may suppress legitimate fast-motion outliers. | Only use the prior as a soft bias; keep the reprojection residual dominant. |
| **v48 domain normalization hides real outliers.** | Domain FiLM may whitewash residual peaks. | Use a residual-agnostic shortcut connection so the z-score is always observable. |
| **Mutual exclusion with v33 is violated.** | Running both detectors doubles the rejection and removes too many views. | Raise `ValueError` if both `use_outlier_view_rejection_v33` and `use_outlier_view_rejection_v49` are true. |

---

## 5. Success metrics and experiments

### 5.1 Metrics

| Metric | How to measure | Target |
|--------|----------------|--------|
| `val_MPJPE` | Mixed H36M/MPI/WebBridge validation | ≤ v46 baseline − 1 mm at full views |
| `MPJPE@k` for `k=2,3,4` | `experiments/eval_variable_views.py` | ≥ 5% improvement over v46 at `k ≤ 3` |
| Outlier recall @ 0.5 | Augmented validation clips (known injected outliers) | ≥ 0.75 |
| Clean-view precision @ 0.5 | Non-augmented validation clips | ≥ 0.95 |
| v37 reliability AUC | Rank views by v37 reliability vs. ground-truth reprojection error | ≥ 0.85 |
| Iteration gain | Delta between pass 1 and pass `v49_outlier_n_iters` MPJPE | ≥ 1 mm improvement at `k ≤ 3` |

### 5.2 Recommended experiments

| Stage | Hardware | Config / script | Goal |
|-------|----------|-----------------|------|
| **Smoke** | RTX 4090 | `configs/benchmark_v49_outlier_view_rejection_smoke.yaml` | `val_MPJPE < 80 mm`, no NaN/OOM, passes `tests/test_outlier_view_rejection_v49.py`. |
| **Full** | A800-D | v46/v47 checkpoint + v49 head, `v49_outlier_n_iters=2` | ≥ 5% relative improvement at `k ≤ 3` over v46; no regression at full views. |
| **Ablation** | RTX 4090 | disable v37 / v46 / v47 / v48 interaction one at a time | Identify which feedback source matters most. |

### 5.3 Expected smoke outcome

Using the same 500-sample smoke recipe as v46:

```yaml
model:
  use_v46_sparse_view_generalization: true
  use_outlier_view_rejection_v49: true
  v49_outlier_n_iters: 2
  v49_outlier_min_views_strict: 3
  v49_outlier_max_reject_frac: 0.25
```

Expected: `val_MPJPE  70-78 mm` (v46 smoke ~80 mm baseline), with visible improvement when `outlier_view_prob=0.3` augmentation is active.

---

## 6. Self-evolution feedback loop

v49 is the **feedback hub** of the v46-v48 self-evolution pipeline:

1. **v37 -> v49:** v37 predicts per-(view, joint) reliability `r_vj` from refined tokens.
2. **v49 -> v25/v46:** v49 fuses `r_vj` with reprojection residuals and sparse-view reliability to produce clean triangulation weights.
3. **v46 -> v49:** the v46 sparse-view reliability head passes `s_v` so v49 knows which views are already suspected to be weak.
4. **v47 <-> v49:** v47 uses the temporally refined poses to produce a per-view outlier prior; v49 returns hard/soft outlier decisions that v47 can use to discount noisy frames in the next forward pass.
5. **v49 -> v37 (feedback):** the final reprojection residual after v49 rejection becomes a supervised target for v37:
   ```python
   target_reliability = sigmoid(-reproj_err * 10)
   loss_v37_feedback = v49_outlier_feedback_weight * MSE(r_vj, target_reliability)
   ```
   This closes the loop: better outlier rejection trains a better reliability estimator, which in turn improves outlier rejection.

---

## 7. Relation to other variants

- **v33 outlier-view rejection:** v49 supersedes v33. Do not enable both.
- **v37 self-critique view reliability:** v49 consumes v37 scores and feeds back residual targets.
- **v46 sparse-view generalization:** v49 uses v46 reliability and protects against over-rejection when views are sparse.
- **v47 temporal aggregation:** v49 can consume a v47 outlier prior and its output refines the input to v47.
- **v48 domain generalization:** v49 relies on v48 residual normalization for domain-robust thresholds.

---

## 8. Next steps

1. Implement `motionflow_mv/fusion/outlier_view_rejection_v49.py` with unit tests.
2. Wire v49 into `omniview_fusion_v5.py` with a mutual-exclusion check against v33.
3. Add CLI/YAML flags in the trainer and create smoke config + script.
4. Run smoke on RTX 4090 and compare `MPJPE@k` with v46 baseline.
5. Queue a full A800 run starting from the best v47/v48 checkpoint.
6. Update `AGENTS.md` status table once smoke completes.
