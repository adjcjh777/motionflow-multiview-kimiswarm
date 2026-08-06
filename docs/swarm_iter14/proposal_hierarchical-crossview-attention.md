# Proposal: Hierarchical Cross-View Attention

**Author:** Iter14 design swarm  
**Date:** 2026-08-06  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  
**Related prior work in repo:** Factorised ST+PP model (`motionflow_mv/fusion/ray_attention_temporal_crossview_factorized_residual_principal_point_model.py`), Iter13 smoke result **57.68 mm** MPJPE on 500 samples / 5 epochs.

---

## 1. Problem

The current factorised model alternates one view-level and one temporal-level transformer layer per block, which may prevent deep cross-view reasoning because temporal mixing happens before view-level features have been refined.

## 2. Hypothesis

Stacking a dedicated view-level attention *stage* before a separate temporal-level attention *stage* will improve cross-view feature aggregation and yield better pose accuracy than the alternating factorised design, without increasing parameters or latency.

## 3. Method

### 3.1 Architecture changes

Create a new model file:

- **New:** `motionflow_mv/fusion/ray_attention_temporal_crossview_hierarchical_residual_principal_point_model.py`
  - Subclass `RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint`.
  - Replace the alternating `for view_layer, temporal_layer in zip(...)` loop with a hierarchical pipeline:
    1. **View stage:** run all `n_view_layers` view-level `TransformerEncoderLayer`s sequentially over `(B*T*J, V, d)`.
    2. **Temporal stage:** run all `n_temporal_layers` temporal-level `TransformerEncoderLayer`s sequentially over `(B*V*J, T, d)`.
  - Keep the per-frame encoder (`_extract_frame_features`), positional embeddings, `PrincipalPointCorrection`, weight head, DLT triangulation, and residual MLP unchanged.
  - Keep constructor signature identical to the factorised PP model so existing configs remain compatible.

This differs from the current factorised implementation in `motionflow_mv/fusion/ray_attention_temporal_crossview_factorized_residual_principal_point_model.py:111-118`, where view and temporal layers are interleaved.

### 3.2 Loss / data changes

No loss or data changes; use the existing MPJPE + confidence-weighted triangulation training objective and the standard MPI-INF-3DHP loader.

### 3.3 Files to create or modify

- **Create:** `motionflow_mv/fusion/ray_attention_temporal_crossview_hierarchical_residual_principal_point_model.py`
- **Create:** `experiments/train_hierarchical_crossview_pp_smoke_mpiinf3dhp.py` (copy of `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` with the new model imported)
- **Modify (optional, post-smoke):** `experiments/benchmark_runtime.py` to register the new model for latency comparison.

---

## 4. Smoke-Test Plan

Run a 5-epoch smoke on a small subset to validate that the hierarchical stage ordering trains stably and does not underperform the factorised baseline on the same sample budget.

| Setting | Value |
|---|---|
| Dataset | MPI-INF-3DHP S2/Seq1 |
| Samples | 500 (matching Iter13 factorised smoke) |
| Epochs | 5 |
| Batch size | 2 |
| Model dims | `d=32`, `residual_hidden=64`, `n_view_layers=2`, `n_temporal_layers=2` |
| Trainer | Adapted from `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` |

**Pass/fail criteria:**

- **Pass:** val MPJPE ≤ 60 mm and no NaNs / crashes.
- **Pass:** val MPJPE is ≤ the factorised smoke result (57.68 mm) at the same sample count.
- **Pass:** predicted `pp_delta` and `focal_scale` remain finite and bounded.
- **Fail:** MPJPE > 75 mm, any NaN/Inf, or training time > 2× the factorised smoke.

---

## 5. Evaluation Plan

If the smoke passes, run the standard evaluation scripts used by the project:

- **Metrics:** MPJPE, PA-MPJPE, PCK@50/100/150, AUC on MPI-INF-3DHP.
- **Scripts:**
  - `experiments/eval_full_metrics.py --model hierarchical_crossview_pp` (or equivalent eval entry)
  - `experiments/benchmark_runtime.py` to compare single-frame and single-clip latency against the factorised PP baseline and the 9.32 mm anchor.
- **Expected target for full run:** clean val MPJPE ≤ 9.8 mm (within 0.5 mm of the 9.32 mm anchor).

---

## 6. Estimated GPU/CPU Cost on RTX 4090

| Phase | Hardware | Time |
|---|---|---|
| Smoke (5 epochs, 500 samples) | RTX 4090 | ~15–25 min |
| Full training (20–50 epochs, full split) | RTX 4090 / A800-D | ~4–8 h on RTX 4090 |
| Eval + latency benchmark | RTX 4090 or CPU | ~10–20 min |

The hierarchical design has the same number of layers and parameters as the factorised model, so FLOPs and memory are comparable; the only difference is stage ordering.

---

## 7. Risks & Fallback

| Risk | Mitigation / Fallback |
|---|---|
| Hierarchical ordering is no better than alternating (wasted GPU time). | If smoke MPJPE is not better than the factorised smoke, keep the factorised model and abandon this branch. |
| Temporal attention after all view layers causes gradient instability. | Add layer norm and residual connections as in the base model; if NaNs appear, reduce learning rate or clip gradients. |
| Longer receptive context needed; 2+2 layers may be insufficient. | Increase `n_view_layers` / `n_temporal_layers` to 4 in a follow-up smoke. |
| No improvement on full data. | Integrate only the view-stage as a drop-in replacement inside the existing factorised block rather than a full hierarchical refactor. |

