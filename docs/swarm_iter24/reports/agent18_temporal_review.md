# Agent-18: Temporal Module Review for v47

**Scope:** Review the existing temporal modules that could overlap with the proposed `TemporalAggregationV47` head, identify what is already implemented, and recommend how to avoid duplication. The review covers v26 (`temporal_geometry_fusion_v26`), v35 (`temporal_view_joint_graph_network_v35`) and v45 (`adaptive_geometry_fusion_v45`).

**Files inspected**
- `docs/swarm_iter24_action_plan.md`
- `docs/proposals/v47_combined_architecture.md`
- `AGENTS.md`
- `motionflow_mv/fusion/temporal_geometry_fusion_v26.py` + `tests/test_temporal_geometry_fusion_v26.py`
- `motionflow_mv/fusion/temporal_view_joint_graph_network_v35.py`
- `motionflow_mv/fusion/adaptive_geometry_fusion_v45.py` + `tests/test_adaptive_geometry_fusion_v45.py`
- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`

**Tests run**
```bash
python -m pytest tests/test_temporal_geometry_fusion_v26.py tests/test_adaptive_geometry_fusion_v45.py -q
```
Result: **33 passed in 4.96s**.

---

## 1. v26 — TemporalGeometryFusionV26

`motionflow_mv/fusion/temporal_geometry_fusion_v26.py`

### What it does
- Extends `MultiViewGeometryFusionV25` by inserting a **spatio-temporal geometry attention** block **inside** the v25 ray-token pipeline.
- Input/operates on ray tokens `(B, T, V, J, d)`.
- For each `(t, v_q, j)` query, it attends to keys in a local temporal window across **all views**, using epipolar distance and ray-intersection logits as attention biases plus a learnable per-offset temporal bias.
- Has a learnable scalar `residual_gate` (default `0.0`) so the temporal path is a no-op at init.
- Optionally adds a temporal smoothness loss on the refined 3D trajectory.

### Integration in v5
Instantiated as `TemporalGeometryFusionV26` when `use_temporal_geometry_fusion_v26=True` in `omniview_fusion_v5.py:533`. It replaces the normal `MultiViewGeometryFusionV25` block and is invoked around line 1471.

### Key take-aways for v47
- **Not duplicated by v47.** v26 fuses temporal information at the **ray-token / mid-feature** level before/inside the triangulation head. v47, per the proposal, fuses temporal information at the **output 3D pose** level after triangulation.
- v26’s local-window geometry attention is heavier and operates on `(B,T,V,J,d)` tokens. v47 is intended to be a much smaller pose-level smoother.

---

## 2. v35 — TemporalViewJointGraphNetworkV35

`motionflow_mv/fusion/temporal_view_joint_graph_network_v35.py`

### What it does
- Extends v34’s `(view, joint)` graph with **temporal edges** connecting the same `(view, joint)` node across adjacent frames.
- Operates on 5D feature tokens `(B, T, V, J, d)` after the graph-joint attention stage in `omniview_fusion_v5.py`.
- Uses a cached spatio-temporal edge index (bone, symmetry, cross-view, self, temporal) and a gated residual with sigmoid-initialized gate.

### Integration in v5
Enabled by `use_temporal_view_joint_graph_network_v35=True`, applied around line 1173 inside the feature-processing stream **before** the final triangulation/residual head.

### Key take-aways for v47
- **Not duplicated by v47.** v35 is a graph-based temporal module over **feature tokens** and is wired inside the v5 feature stream. v47 is a transformer-based smoother over **3D poses** at the very end of the pipeline.
- v35 requires building a graph per `(T, J)` signature and is tied to the skeleton topology. v47 works on plain pose trajectories and is skeleton-agnostic, which is simpler for sparse-view generalization where the number of views varies.

---

## 3. v45 — AdaptiveGeometryFusionV45 (a.k.a. v45-AGF / TGA)

`motionflow_mv/fusion/adaptive_geometry_fusion_v45.py`

### What it does
- Predicts **per-view / per-joint reliability weights** from reprojection residuals using a tiny MLP.
- It is **not a temporal module**. It is a per-frame adaptive weighting scheme used inside `MultiViewGeometryFusionV25` before DLT triangulation.
- Zero-initialized final layer gives weights ≈ 1.0 at init (identity-like).

### Integration in v5 / v25
- Passed into `MultiViewGeometryFusionV25` via `use_v45_adaptive_geometry_fusion` (see `omniview_fusion_v5.py:562` and `multiview_geometry_fusion_v25.py:412`).
- The v46 sparse-view generalization head (`SparseViewGeneralizationV46`) is a separate, feature-based reliability head that is applied later in the v5 pipeline around line 1326.

### Key take-aways for v47
- v47 should **not re-implement per-view reliability prediction**. The proposal already states that v47 reuses v45/v46 reliability weights as part of the v46 base.
- v47 can rely on the existing `view_mask` and, if needed, the v46 reliability map to know which frames/views are trustworthy.

---

## 4. Proposed v47 TemporalAggregationV47

From `docs/proposals/v47_combined_architecture.md`:

- Operates on the **triangulated 3D pose** `(B, T, J, 3)` produced by v46.
- Uses a small transformer encoder (`d_model=64`, `n_heads=4`, `num_layers=2`) over `(time, joint)` tokens.
- Conditions each token on the number of contributing views (`log(n_views_t)`).
- Applies a gated residual refinement `P_t + g · ΔP_t` with `g` initialised to `0.0` (identity at init).
- Adds a small temporal smoothness loss.

---

## 5. Duplication risk assessment

| Existing module | Operates on | Overlap with v47? | How to avoid |
|-----------------|-------------|-------------------|--------------|
| v26 `TemporalGeometryFusionV26` | Ray tokens `(B,T,V,J,d)` before triangulation | **None** — different abstraction level | Do not put v47 inside v25; keep it post-triangulation |
| v35 `TemporalViewJointGraphNetworkV35` | Feature tokens `(B,T,V,J,d)` inside v5 | **None** — graph vs. transformer, different inputs | Do not reuse v35 graph code; v47 is pose-level and skeleton-free |
| v45 `AdaptiveGeometryFusionV45` | Per-view reliability from residuals | **None** — v45 is not temporal; v47 is not a reliability head | Reuse `view_mask` / v46 reliability, do not add another residual-predicting MLP |
| v32 `TrajectoryConsistencyRefinerV32` (existing in v5) | Output 3D poses `(B,T,J,3)` | **Partial overlap** in the *pose-level temporal smoothing* space | Keep v47 gated and optional; v32 is a simple consistency refiner, v47 is a learned transformer head. They can coexist if both are gated. |

### Main non-duplication argument
v26, v35 and v45 solve temporal or reliability problems at **earlier stages** of the network (features, tokens, per-view weights). v47 is the first module that applies a learned temporal model **after** the v46 sparse-view triangulation, directly to the final 3D poses. Therefore the functionality is complementary rather than duplicated.

---

## 6. Recommendations for v47 implementation

1. **Place v47 after v46 triangulation.** Add the new call in `omniview_fusion_v5.py` after the v46/v25 triangulation path returns `pred_3d_gn` / `pred_3d` (around the residual refinement section, lines ~1556–1580), not inside `MultiViewGeometryFusionV25`.

2. **Keep the v47 head independent.** Implement a new file `motionflow_mv/fusion/temporal_aggregation_v47.py` with its own `TemporalAggregationV47` class. Do not subclass v26 or v35 — the input shape and semantics are different.

3. **Preserve identity at init.** Use the same gated-residual pattern as v26/v35: initialize the output projection and final mixing gate to zero. This protects v46 baselines and keeps backward compatibility with checkpoints.

4. **Reuse existing view-sparsity signals.** Pass the existing `view_mask` to v47 and, if useful, the v46 reliability map. Do not re-implement per-view weight prediction; that is the job of v45/v46.

5. **Avoid conflict with v32.** v32 (`TrajectoryConsistencyRefinerV32`) also refines the output pose trajectory. If a config enables both v32 and v47, apply them as sequential gated residuals and make sure v47’s loss weight is configurable (`v47_temporal_loss_weight`) so the user can disable the extra smoothness term.

6. **Unit-test the differences.** Tests should cover:
   - Identity at init with `residual_gate_init=0.0`
   - Variable-length clips (`T=1` and `T < temporal_window`)
   - View masks and view-count conditioning
   - Shape invariance: `(B,T,J,3) -> (B,T,J,3)`

---

## 7. Open questions / follow-up

- Should v47 consume the v46 per-view reliability map `(B,T,V,J)` as an input, or only the coarser `view_mask` and `log(n_views_t)`? The proposal mentions view-count conditioning; using the full reliability map would be a natural extension but is not required for the first smoke test.
- Should v47 be allowed to run alongside `use_temporal_geometry_fusion_v26` or `use_temporal_view_joint_graph_network_v35`? The design is additive, but stacking three temporal mechanisms may over-smooth fast motion. A smoke ablation with/without v26/v35 would clarify.
- The trainer (`experiments/train_omniview_fusion_v5_webbridge_multi.py`) should support freezing v25/v45/v46 weights during the first epoch, as noted in the proposal. This staging is independent of the module code but important for avoiding duplication of training logic.

---

**Conclusion:** v47 does not duplicate v26, v35 or v45 because it targets a distinct stage of the pipeline (post-triangulation 3D pose refinement vs. mid-feature temporal fusion / per-frame reliability). The recommended implementation keeps v47 as a lightweight, gated, pose-level transformer head that reuses existing `view_mask` / v46 signals without re-implementing reliability prediction or geometry attention.
