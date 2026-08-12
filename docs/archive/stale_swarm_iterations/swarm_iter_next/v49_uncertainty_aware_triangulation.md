# v49: Uncertainty-Aware Triangulation

**Status:** Proposal / ready for design review  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #167 (proposed)  
**Depends on:** v46-SVG (#160), v37 self-critique reliability  

---

## 1. Problem Statement

Current triangulation in `OmniMultiViewFusionV5` combines detector confidence with learned per-view reliability (v46) and self-critique view reliability (v37), but it still treats each 2-D observation as a deterministic point. When views are sparse or corrupted, the DLT system is under-constrained and the model has no explicit estimate of **per-ray, per-joint geometric uncertainty** to guide re-weighting.

The v33 uncertainty-aware triangulation proposal introduced this idea but was never integrated into the v46–v48 mainline. Re-implementing it as v49 lets us:

1. Reuse the v46 reliability head and v37 self-critique loop instead of building a standalone heavy head.
2. Keep the change minimal and wired into the sparse-view / temporal / domain stack.
3. Provide a calibrated uncertainty signal that can drive the self-evolution feedback loop.

---

## 2. Proposed Approach

Add a lightweight `UncertaintyAwareTriangulationV49` module that predicts a per-view, per-joint 2-D log-variance from the existing multi-view tokens, converts it into a precision weight for DLT, and supervises the prediction with a reprojection negative-log-likelihood (NLL) loss.

```text
v25 Multi-View Geometry Fusion output tokens (B, T, V, J, d)
    |
    ▼
[UncertaintyAwareTriangulationV49]
    |
    ├── 2-layer MLP -> per-view, per-joint log-variance Σ_vj
    ├── Precision-weighted DLT (reuses uncertainty_weighted_triangulation.py)
    ├── Residual refinement around v25 output (identity at init)
    └── Reprojection NLL auxiliary loss
    |
    ▼
v46 reliability weights r_v (modulates Σ_vj)
    |
    ▼
v47 Temporal Aggregation / v48 Domain Refinement
```

The module is placed **after** the v25 geometry-fusion block and **before** the v46 sparse-view reliability weighting, so v46 can use the uncertainty as an soft mask. It is optional and identity-at-init.

---

## 3. Concrete Code-Level Changes

### New files

- `motionflow_mv/fusion/uncertainty_aware_triangulation_v49.py`
  - `UncertaintyAwareTriangulationV49`
  - Inputs: `points_2d (B, T, V, J, 2)`, `features (B, T, V, J, d)`, `proj_matrices (B, T, V, 3, 4)`, `pred_3d_init (B, T, J, 3)`, optional `view_mask (B, T, V)`
  - Outputs: `pred_3d_ref (B, T, J, 3)`, `uat_loss (scalar)`
  - Internal helpers: `_uncertainty_mlp`, `_precision_weighted_dlt`
- `configs/benchmark_v49_uncertainty_aware_triangulation_smoke.yaml`
- `scripts/run_v49_uncertainty_aware_triangulation_smoke.sh`

### Modified files

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add flags and instantiate `UncertaintyAwareTriangulationV49` when enabled.
  - Call it after v25 geometry fusion; fold `uat_loss` into the geometry loss term.
- `motionflow_mv/fusion/sparse_view_generalization_v46.py`
  - Optionally consume the v49 uncertainty to modulate the v46 reliability score before weighted triangulation.
- `motionflow_mv/fusion/self_critique_view_reliability_v37.py`
  - Optionally use v49 predicted log-variance as an additional input to the v37 reliability MLP.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Expose CLI flags and pass-through in `build_model_from_args`.
- `experiments/eval_variable_views.py`
  - Report `reproj_nll@k` as a calibration diagnostic alongside `MPJPE@k`.

### New training flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_uncertainty_aware_triangulation` | bool | `False` | Master switch. |
| `v49_uat_loss_weight` | float | `0.01` | Weight of the reprojection NLL loss. |
| `v49_uat_log_var_min` | float | `-10.0` | Clamp lower bound for predicted log-variance. |
| `v49_uat_log_var_max` | float | `10.0` | Clamp upper bound. |
| `v49_uat_hidden` | int | `64` | Hidden dim of the uncertainty MLP. |
| `v49_uat_residual_gate_init` | float | `0.0` | Initial residual scale (identity at init). |
| `v49_uat_feed_v37` | bool | `True` | Feed predicted uncertainty into v37 self-critique reliability. |
| `v49_uat_feed_v46` | bool | `True` | Use uncertainty to modulate v46 reliability weights. |

---

## 4. Risks / Failure Modes

| Risk | Mitigation |
|------|------------|
| NLL loss dominates early training and destabilizes v25. | Ramp `v49_uat_loss_weight` from `0.0` over the first 3 epochs; clamp `log_var` aggressively. |
| Uncertainty collapses to a constant. | Supervise with reprojection NLL on every active view; add a small entropy regularizer on predicted precisions. |
| Redundancy with v37/v46 reliability. | Treat v49 as *observation uncertainty*, v37 as *reliability*, v46 as *sparse-view weight*. Ablate each. |
| Sparse-view degeneracy (<2 views). | Fall back to v25 output when active views < 2 for a joint. |
| Computational overhead. | Reuse existing `uncertainty_weighted_triangulation.triangulate_uncertainty_weighted_batched`; the DLT system is tiny. |

---

## 5. Success Metrics and Recommended Experiments

### Smoke experiment (RTX 4090)

| Field | Value |
|-------|-------|
| Config | `configs/benchmark_v49_uncertainty_aware_triangulation_smoke.yaml` |
| Hardware | Local RTX 4090 |
| Flags | `use_v49_uncertainty_aware_triangulation`, `use_v46_sparse_view_generalization`, `use_v47_temporal_aggregation` |
| Goal | `val_MPJPE < 75 mm`, no NaN/OOM, finite `reproj_nll` |
| Expected | ~70–74 mm, comparable to v47 smoke baseline |

### Full experiment (A800-D)

| Field | Value |
|-------|-------|
| Hardware | A800-D |
| Base | Best v47 checkpoint or warm-start from v46-SVG |
| Flags | `use_v49_uncertainty_aware_triangulation`, `v49_uat_loss_weight 0.01`, `v49_uat_feed_v37`, `v49_uat_feed_v46` |
| Goal | ≥1 mm improvement over v47 at full views; ≥5% relative improvement at `MPJPE@2`/`MPJPE@3` |
| Metric | `val_MPJPE`, `MPJPE@k`, `reproj_nll@k` |

### Ablations

1. `v49_uat_only` — enable only the new head on top of v25.
2. `v49_uat_no_v46` — disable v46 modulation to isolate the DLT re-weighting effect.
3. `v49_uat_no_v37` — disable v37 feedback to isolate the auxiliary NLL loss.
4. `v49_uat_low_weight` — `v49_uat_loss_weight=0.001`.
5. `v49_uat_high_weight` — `v49_uat_loss_weight=0.1`.

---

## 6. Self-Evolution Feedback Loop

v49 closes a lightweight self-evolution loop around triangulation uncertainty:

1. **Predict:** `UncertaintyAwareTriangulationV49` outputs per-view log-variance `Σ_vj` from the multi-view tokens.
2. **Weight:** v46 reliability weights and v37 self-critique scores are modulated by `Σ_vj`, so noisy views contribute less.
3. **Triangulate:** A precision-weighted DLT produces the 3-D pose.
4. **Critique:** The refined pose is projected back to each view; the reprojection residual `r` supervises `Σ_vj` through the NLL loss `0.5 * (r^T Σ^{-1} r + log det Σ)`.
5. **Update:** Gradients from the NLL loss update the uncertainty MLP, improving the next forward pass.

This loop is fully differentiable and adds no extra training stage. It directly implements the paper's self-evolution idea: the model learns to estimate its own observation uncertainty from reprojection feedback and uses that estimate to re-weight triangulation.

---

## 7. Paper Story Fit

v49 supports the ICRA/CVPR 2027 claim: *Our multi-view fusion is robust to noisy and sparse observations because it learns per-view, per-joint observation uncertainty end-to-end and uses that uncertainty to weight triangulation.* It is a small, focused addition that sits naturally after v25 and before v46/v47, improving both accuracy and calibration without changing the overall pipeline.

---

## 8. Next Steps

1. Wait for v46-SVG smoke (#160) and v37 self-critique integration to land.
2. Implement `UncertaintyAwareTriangulationV49` and unit tests.
3. Wire flags into `OmniMultiViewFusionV5` and the trainer.
4. Add `configs/benchmark_v49_uncertainty_aware_triangulation_smoke.yaml` and smoke script.
5. Run smoke on RTX 4090 and compare `MPJPE@k` and `reproj_nll@k` against v46/v47 baselines.
6. Queue full A800 run once smoke targets are met.
