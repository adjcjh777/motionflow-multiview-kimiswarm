# Complexity vs. Accuracy Trade-off Across v31-v43

**Scope:** Analyze the model-complexity / accuracy trade-off for the MotionFlow-MultiView variants from v31 to v43, as input to the v44 decision process.

**Date:** 2026-08-09  
**Sources:** `docs/v43_decision_criteria.md`, `docs/a800_results_summary.md`, `docs/results_snapshot_2026_08_09.md`, `motionflow_mv/fusion/omniview_fusion_v5.py`, and `scripts/analyze_model_size_flops.py`.

---

## 1. Executive Summary

The strongest model in the v31-v43 family is **v25 geometry fusion** at **17.17 mm** (A800 full). Every architectural addition trialled since v25 (hierarchical encoders, ray-conditioned attention, view-joint graph networks, uncertainty-gated refinement, reliability estimators, physical losses, and domain-weighted losses) has **increased complexity without improving best epoch-1 validation MPJPE** on the held-out sets.

This creates a clear tension for v44: the v31-v43 stack adds **~1.35 M additional parameters** and yet lags a much simpler baseline by **>8 mm**. The upcoming A800 results for v42/v43 (with physical/domain losses and adaptive residuals) will decide whether any complex-stack run can close this gap.

---

## 2. Variant Map (v31-v43)

| Variant | Module added | What it does | Added trainable params (d=128) |
|---------|--------------|--------------|-------------------------------|
| v25 baseline | `MultiViewGeometryFusionV25` | Geometry-aware cross-view attention + learned depth triangulation | 191 k (module only) |
| v31 | `HierarchicalViewEncoderV31` | Geometry-biased hierarchical encoder (part-scale layers) | 497 k |
| v32 | `TrajectoryConsistencyRefinerV32` | Temporal smoothness / drift consistency losses | tiny (loss-only) |
| v33 HMSP | `HierarchicalMultiscaleCrossViewSpatialPyramidV33` | Multi-scale cross-view attention with adaptive scale fusion | 513 k |
| v33 UAT | `UncertaintyAwareTriangulationV33` | Per-view log-variance + precision-weighted DLT | 9 k |
| v33 OVR | `OutlierViewDetectorV33` | Learned per-joint/part/domain outlier down-weighting | 8 k |
| v33 RCA | `RayConditionedCrossViewAttentionV33` | Ray embeddings + ray-intersection bias in cross-view attention | 215 k |
| v34 VJGN | `ViewJointGraphNetworkV34` | (view, joint) graph attention over skeleton/symmetry/cross-view edges | 150 k |
| v34 GVJGN | `GeometryViewJointGraphNetworkV34` | v34 + epipolar/ray-intersection edge bias | 150 k + geometry overhead |
| v35 TVJGN | `TemporalViewJointGraphNetworkV35` | v34 + temporal edges across frames | 150 k |
| v36 UGIGR | `UncertaintyGatedIterativeGraphRefinementV36` | Source-gated graph attention + iterative refinement | 92 k |
| v37 SCVR | `SelfCritiqueViewReliabilityV37` | Per-(view,joint) reliability score from reprojection residuals | 21 k |
| v39-v41 | Reliability coupling + physical/domain losses | Couple v37 to v36; bone/joint/symmetry/floor priors; per-domain MSE weights | small |
| v42 | v36 + physical loss + domain weights | Tests whether v37 is needed | v36 + overheads |
| v43 | v42 + adaptive per-node residual | Scale v36 residual by per-node uncertainty gate | v36 + 0 |

---

## 3. Complexity Metrics

### 3.1 Per-module parameter counts

Measured with `tmp/profile_v31_v43_complexity.py` (CPU-only, d=128, 4 views, 17 joints):

| Module | Params |
|--------|--------|
| v25 geometry fusion (module) | 191 k |
| v31 hierarchical encoder | 497 k |
| v33 HMSP | 513 k |
| v33 ray-conditioned attention | 215 k |
| v34 VJGN | 150 k |
| v35 TVJGN | 150 k |
| v36 UGIGR | 92 k |
| v37 SCVR | 21 k |
| v33 uncertainty-aware triangulation | 9 k |
| v33 outlier-view detector | 8 k |
| **Sum of v31-v43 add-ons** | **~1.35 M** |

### 3.2 Approximate full-model capacity

The v4/v5 base model (OmniMultiViewFusionV4/V5) is reported in `docs/paper_draft_icra_cvpr_2027.md` at **243 k params** for d=64, residual_hidden=128. The add-on modules above (d=128) are therefore comparable in size to the base model itself. A representative v42/v43 full configuration with d=128 is expected to land in the **1.5-1.8 M parameter** range.

### 3.3 Forward-pass cost (v25 anchor)

`scripts/analyze_model_size_flops.py` reports for the v25 module:

- **191 k params**
- **0.202 GFLOPs** forward pass (B=2, T=4, V=4, J=17, d=128)
- Per-module overhead of v31-v43 modules is dominated by the quadratic cross-view / graph attention terms, with HMSP and ray attention adding the largest multiplicative factors.

---

## 4. Accuracy Metrics

### 4.1 Best completed A800 runs (source: `docs/a800_results_summary.md`)

| Rank | Run | val_MPJPE (mm) | Notes |
|------|-----|----------------|-------|
| 1 | v25 geometry fusion full | **17.17** | Best overall; strong baseline |
| 2 | v25 geometry fusion small | 18.31 | Small subset |
| 3 | v11 IR | 20.06 | Legacy baseline |
| 4 | v18 deformable attention | 20.24 | Strong v18 baseline |
| 5 | v31 domain balanced | 25.90 | First v31 A800 result |
| 6 | v32 combined | 26.49 | v33/v34 stack |
| 7 | v32 trajectory consistency | 26.51 | Temporal loss only |
| 8 | v32 ray attention | 26.58 | v33 RCA variant |
| 9 | v33 ray conditioned attention | 26.85 | v33 RCA on A800 |
| 10 | v31 hierarchical more dropout | 26.97 | Hardened v31 |
| 11 | v33 uncertainty-aware triangulation | 27.57 | v33 UAT on A800 |
| 12 | v32 physical alignment | 27.75 | v28 physical loss |
| 13 | v31 physical floor only | 28.41 | Physical warmup ablation |
| 14 | v33 outlier view rejection | 30.57 | v33 OVR on A800 |
| 15 | v31 geometry attention | 33.69 | Geometry-biased attention |
| 16 | v31 outlier view adaptive | 37.87 | Adaptive thresholds |

### 4.2 Best local RTX 4090 epoch-1 (source: `docs/results_snapshot_2026_08_09.md`)

| Rank | Run | val_MPJPE (mm) | Notes |
|------|-----|----------------|-------|
| 1 | v2 d128 no graph | 24.71 | Local best; simpler architecture |
| 2 | v2 d128 dense graph v2 | 25.19 | |
| 3 | v34 HMSP + GVJGN stack | 25.50 | Complex v33/v34 stack |
| 4 | v33 combined | 25.78 | HMSP + UAT + OVR + RCA |
| 5 | v42 v36 + physical + domain | 26.16 | d=64, old manifest |
| 6 | v36 UGIGR | 26.42 | |
| 7 | v37 SCVR | 26.94 | |
| 8 | v35 TVJGN | 27.08 | |
| 9 | v34 VJGN | 27.17 | |
| 10 | v33 HMSP | 27.32 | |

---

## 5. Trade-off Analysis

### 5.1 Complexity is not translating to accuracy

- The v25 baseline alone reaches **17.17 mm** on A800.
- The v31-v43 module stack adds **>1 M parameters** but the best result in that family is **25.90 mm** (v31 domain balanced) on A800.
- Locally, the best v31-v43 result is **25.50 mm**, still far behind v25.
- Even stacking many modules (v33 combined, v34 HMSP+GVJGN) does not break the **25 mm** local barrier, while the much smaller v25/v18 baselines sit well below it.

### 5.2 Pareto frontier

Given the current data, the Pareto-efficient options are:

1. **v25 geometry fusion** — lowest complexity in the strong family, best accuracy (17.17 mm).
2. **v18 deformable attention** — slightly higher complexity, 20.24 mm, still much better than v31-v43.
3. **v31-v34 complex stack** — highest complexity, but dominated by v25 and v18 on both axes.

### 5.3 Hypotheses for the failure mode

1. **Overfitting.** `docs/a800_results_summary.md` notes that v25 full reached its best at epoch 1 and then overfit (17.17 -> 59.14 mm). Adding capacity without stronger regularisation accelerates this.
2. **Loss interaction.** Physical/domain losses (v40-v42) and uncertainty gating (v36) may fight with the triangulation head if their weights are not carefully balanced.
3. **Identity-at-init stacking.** Most v31-v43 modules are gated to be identity at init. Training small residual gates with large additive stacks can be brittle; the model may never leave the basin where the gates stay closed.
4. **Smoke vs. full mismatch.** Local smoke results (20-80 samples) are not always representative, but the A800 full-scale results confirm that the gap persists at scale.

### 5.4 Pending experiments that could change the picture

Per `docs/v43_decision_criteria.md`, the following A800 runs are queued/running:

| Run | Purpose |
|-----|---------|
| v25 all-train baseline | Is v25 strong enough even with full WebBridge mixed data? |
| v25 + physical + domain | Can selective v40/v41 additions help a simple baseline? |
| v42 (v36 + physical + domain) | Does the complex stack need v37? |
| v43 base (adaptive per-node residual) | Does per-node residual help at d=64? |
| v43 scaled (d=128, 10k samples/seq) | Does capacity help? |
| v43 all-train (full WebBridge mixed) | Does more data help? |

---

## 6. Recommendations for v44

Based on the current trade-off, the v44 design should branch on the pending A800 results:

1. **If v25 all-train is best (< 17 mm or beats v42/v43):**
   - Build v44 on the **v25 geometry-fusion baseline**.
   - Port only the smallest, highest-ROI additions: domain weights (v41) and physical loss (v40) if they do not hurt the v25 baseline.
   - Drop the hierarchical, graph-network, and reliability-stack modules.

2. **If v43 all-train beats v25 all-train:**
   - Keep the v43 architecture, but add heavy regularisation (SWA, increased weight decay/dropout, shorter epochs) to combat epoch-1 overfitting.

3. **If v43 scaled is best but v43 all-train is not:**
   - Data quantity is not the bottleneck; model capacity is. Consider scaling d or layers further, or simplify the data pipeline while increasing capacity.

4. **If v42 beats v43 base:**
   - The per-node adaptive residual is not helping; remove it and focus on v42 + stronger regularisation.

---

## 7. Bottom Line

- **Best accuracy:** v25 geometry fusion, **17.17 mm**, ~191 k module params.
- **Best v31-v43 accuracy:** v31 domain balanced, **25.90 mm**, with >1 M additional params.
- **Implication:** The v31-v43 architectural investments have not yet paid off. v44 should either return to a v25-based stack or wait for the pending v42/v43 A800 runs to justify continued complexity.
