# Swarm Iteration 23 Summary — v46 Sparse-View Generalization

**Status:** Synthesis complete  
**Tracking issue:** #160  
**Branch:** `v46-svg`  
**Date:** 2026-08-09

---

## 1. Executive summary

Swarm iteration 23 is the design phase for **v46 Sparse-View Generalization (SVG)** (#160), which aims to make `OmniMultiViewFusionV5` robust to missing and variable camera counts. Eight agent reports landed in `docs/swarm_iter23/reports/`. The synthesis is clear:

- **Baseline:** v45-AGF is training stably on the local RTX 4090 (Epoch 1 `val_MPJPE` = 31.95 mm) and is the preferred starting point for v46.
- **Production baseline:** The A800 v25 all-train baseline is running on GPU6, ~150 steps into Epoch 1, with first validation expected around  05:25–05:50 UTC.
- **Integration path:** v46 can reuse the existing `VariableViewSetAggregator`, `view_mask` plumbing, and `augment_clip` view-dropout path. The v45-AGF reliability head can be reused as the v46 per-view reliability head.
- **Module design:** `SparseViewGeneralizationV46` is a lightweight ISAB + MLP reliability head that outputs `(B, T, V, J)` weights and is inserted after the set aggregator / before the ST transformer. It multiplies the existing triangulation confidence.
- **Data:** 3DPW pseudo files already fit the v46 pipeline with no loader changes; 3DPW actual-mode could be used for sparse-view evaluation.
- **Training:** View-dropout augmentation with curriculum (ramp `p_drop` to 0.3 over the first half of epochs) is the recommended augmentation strategy.
- **Next iteration:** v47 proposes to add lightweight temporal aggregation on top of v46.

The implementation can proceed without major blockers; the main dependency is GPU availability for smoke testing.

---

## 2. Reports synthesized

| Report | Agent | Type | Key finding |
|---|---|---|---|
| `agent01_v45_status.md` | Agent-01 | ANALYZE | v45-AGF medium 4090 run: Epoch 1 `val_MPJPE` = **31.95 mm**; predicted final best **25–29 mm**. Stable, no NaN/OOM. |
| `agent02_a800_status.md` | Agent-02 | ANALYZE | A800 v25 all-train baseline on GPU6, ~150 steps, loss 19.91; first val estimated **~05:25–05:50 UTC**. |
| `agent03_variable_view_review.md` | Agent-03 | ANALYZE | `VariableViewSetAggregator` / `view_mask` / `augment_clip` already provide the integration points for v46. |
| `agent04_v45_reuse.md` | Agent-04 | ANALYZE | `AdaptiveGeometryFusionV45` reliability weights can be reused directly as the v46 per-view reliability head. |
| `agent05_v46_design.md` | Agent-05 | DESIGN | Concrete API and integration plan for `SparseViewGeneralizationV46`; ISAB + 2-layer MLP; identity init; no change to v25. |
| `agent17_3dpw_for_svg.md` | Agent-17 | ANALYZE | 3DPW pseudo fits v46 with no loader changes; actual-mode can provide a sparse-view eval benchmark. |
| `agent18_qwen_selfevolution.md` | Agent-18 | ANALYZE | Qwen3.8 self-evolution maps to v46: self-critique → reliability head; curriculum → view-dropout; selection → `MPJPE@k`. |
| `agent19_v47_combined_architecture.md` | Agent-19 | ANALYZE | v47 = v46 + lightweight temporal aggregation head, staged after v46 lands. |

---

## 3. Synthesis

### 3.1 Baseline readiness

- **v45-AGF (local 4090):** Epoch 1 `val_MPJPE` 31.95 mm is already within the v46 smoke target (< 80 mm). If final best is < 28 mm, v46 should build on v45-AGF rather than v25.
- **v25 all-train (A800):** Running, healthy, GPU-bound; first validation is the gate for whether v25 still outperforms v45/v42 locally (~17 mm target from historical v25 A800 runs).

### 3.2 Architecture: keep it small and reuse

The design consensus from Agent-03, Agent-04, and Agent-05 is that v46 should **not** introduce a new heavy architecture. Instead:

1. **Permutation-invariant view processing:** Reuse `VariableViewSetAggregator` (already handles `view_mask`).
2. **Reliability head:** Reuse `AdaptiveGeometryFusionV45` or implement a small MLP on top of the set-aggregated tokens.
3. **Triangulation:** Multiply v46 weights into the existing confidence before DLT / `MultiViewGeometryFusionV25`.
4. **Training augmentation:** Reuse the existing `augment_clip` view-dropout path with new CLI flags.

This satisfies the proposal’s design principle: *geometry fusion remains the foundation; sparse-view training is an augmentation.*

### 3.3 Data and curriculum

- **Loader changes:** None required for the smoke. The existing `WebBridgeCanonical17Dataset` already pads to 14 views and returns `view_mask`.
- **3DPW:** Pseudo files provide a natural 4-view sparse case; actual-mode (single moving camera) can be an evaluation-only stress test.
- **Curriculum:** Ramp view-dropout probability from 0.0 to the target (default 0.3) over the first half of epochs, mirroring the existing `variable_view_training` curriculum.
- **Domain-aware dropout (optional):** Treat 3DPW gently (lower dropout, same `min_views`) because it has fewer real views.

### 3.4 Evaluation

The v46 success metrics are:

| Metric | Target |
|---|---|
| `val_MPJPE` (full views) | No regression vs. v45/v25 baseline |
| `MPJPE@k` for k = 2, 3, 4 | Report mean/std over all view subsets |
| Smoke | `val_MPJPE` < 80 mm, no NaN/OOM |

`experiments/eval_variable_views.py` should be extended to emit a CSV/JSON of `MPJPE@k` for `k = 2, 3, 4, full`.

### 3.5 Connection to v47

Agent-19 proposes v47 as a temporal aggregation head on top of v46. This is queued behind #160 and depends on v46 first demonstrating sparse-view improvement at `MPJPE@2/3`.

---

## 4. Implementation checklist (derived from reports)

| Agent | File | Action |
|---|---|---|
| Agent-06 | `motionflow_mv/fusion/sparse_view_generalization_v46.py` | Implement `SparseViewGeneralizationV46` with identity init, masking, and gradient tests. |
| Agent-07 | `motionflow_mv/data/view_dropout_augmentation_v46.py` | Implement `drop_views` helper with `min_views` enforcement and curriculum support. |
| Agent-08 | `motionflow_mv/fusion/omniview_fusion_v5.py` | Add v46 flags, instantiate module, wire into forward after set aggregator. |
| Agent-09 | `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add CLI flags and call `drop_views`, composing mask with existing `view_mask`. |
| Agent-10 | `configs/benchmark_v46_svg_smoke.yaml` | Create smoke config enabling v46 with curriculum dropout. |
| Agent-11 | `scripts/run_v46_svg_smoke_local_4090.sh` | Create executable smoke launch script. |
| Agent-12 | `tests/test_sparse_view_generalization_v46.py` | Add unit/integration tests for shape, masking, gradients, and compatibility. |
| Agent-13 | `experiments/eval_variable_views.py` | Extend to report `MPJPE@k` CSV/JSON for `k = 2, 3, 4, full`. |
| Agent-14 | `scripts/launch_v33_a800_queue.py` | Add v46 full-run queue entry. |
| Agent-15 | `docs/proposals/v46_sparse_view_generalization.md` | Polish proposal with 3DPW actual-mode eval plan. |
| Agent-16 | `AGENTS.md` | Update conventions for v46 (optional, if needed). |

---

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| GPU blocked by v45-AGF medium | v46 smoke must wait for RTX 4090 or a free A800 GPU (per AGENTS.md). |
| v46 overlaps with existing variable-view training | Frame v46 as a learned reliability head + sparse-view augmentation, not a replacement. |
| v45-AGF reliability head conflicts with v46 | v46 weights are applied *before* v45; both can be enabled and multiplied. |
| `< 2` active views breaks triangulation | Enforce `min_views >= 2` in `drop_views` and fallback to uniform weights when too few views remain. |
| 3DPW over-dropped | Use dataset-aware dropout (Option B in Agent-17 report) for full runs. |
| Full-view regression | Identity-like init and curriculum keep full-view behavior stable at start. |

---

## 6. Open questions

1. **v25 first val:** Will the A800 v25 all-train baseline hit the historical ~17 mm target? First val is expected ~05:25–05:50 UTC.
2. **v45 final:** Will v45-AGF medium finish below 28 mm? This determines whether v46 builds on v45 or v25.
3. **Per-view vs. per-joint weights:** The current design outputs per-view reliability broadcast to `(B, T, V, J)`. If ablations show per-joint weights help, extend the MLP later without changing the API.
4. **3DPW actual-mode:** Is disk/bandwidth available to convert and evaluate 3DPW actual-mode for `MPJPE@1/2`?
5. **v47 timing:** Does v46 need to land on `main` before v47 design is finalized, or can it be drafted in parallel?

---

## 7. Next concrete steps

1. Wait for the A800 v25 first validation and the v45-AGF final val to confirm the baseline.
2. Implement `SparseViewGeneralizationV46` and the view-dropout helper (Agent-06, Agent-07).
3. Wire v46 into `OmniMultiViewFusionV5` and the trainer (Agent-08, Agent-09).
4. Create smoke config/script and run on RTX 4090 once the GPU is free.
5. Extend `eval_variable_views.py` to report `MPJPE@k` and validate with 3DPW actual-mode if feasible.
6. Add the v46 full run to the A800 queue after smoke passes.

---

*Generated by Agent-20 as a synthesis of the eight reports in `docs/swarm_iter23/reports/` for the v46 Sparse-View Generalization swarm (#160).*
