# v31-v43 Multi-View Fusion Bottleneck Analysis

The v31-v43 model implementations live in `motionflow_mv/fusion/` and are wired together in `OmniMultiViewFusionV5` (`motionflow_mv/fusion/omniview_fusion_v5.py`). Despite a large number of added geometry, graph, uncertainty, and physical-prior modules, the current A800/RTX 4090 snapshot shows that the strongest baseline is still the simpler v25 geometry-fusion path (17.17 mm on A800), while the v31-v43 complex stack lags at 26-37 mm. The three bottlenecks below explain why the added machinery is not translating into higher accuracy.

## 1. Geometry Is Used Only as an Attention Bias, Not as a Hard Fusion Constraint

The model extracts per-view, per-joint feature tokens first, fuses them with attention, and only triangulates *after* the spatio-temporal transformer. Geometry is injected as soft, additive attention biases rather than as a binding constraint:

- `hierarchical_multiview_v31.py` computes `geometry_bias = -epi_dist + ray_intersection_logit` and adds it to cross-view attention scores, gated by a learned sigmoid (`geometry_gate`) initialized to near-zero contribution (`motionflow_mv/fusion/hierarchical_multiview_v31.py:319`).
- `ray_conditioned_attention_v33.py` adds ray embeddings to queries/keys and a ray-intersection logit bias to attention scores, but it is still a residual attention block (`motionflow_mv/fusion/ray_conditioned_attention_v33.py:124-130`).
- Final triangulation happens in the main forward path via `triangulate_dlt_batched_lstsq` *after* all feature processing, using learned weights/covariance/visibility that are not directly supervised by 3D geometry (`motionflow_mv/fusion/omniview_fusion_v5.py:1353`).

**Impact:** The transformer can ignore or down-weight geometric consistency when it conflicts with learned feature patterns. Because the 3D triangulation step is downstream and not tightly coupled to the feature-learning objective, the network can learn to be locally consistent in feature space while producing geometrically inconsistent 3D estimates.

---

## 2. The Uncertainty/Reliability/Gating Stack Is Under-Constrained and Overfits

v36-v44 add iterative uncertainty gating, self-critique view reliability, adaptive per-node residuals, and edge-type-aware gates. These modules are identity-at-init, but once their gates open they introduce many weakly supervised degrees of freedom:

- `uncertainty_gated_iterative_graph_refinement_v36.py` predicts per-node uncertainty gates from features (`uncertainty_mlp`), then multiplies attention weights by `sigmoid(uncertainty_logits)` (`motionflow_mv/fusion/uncertainty_gated_iterative_graph_refinement_v36.py:230-306`). The gate is learned from features alone, with no direct 3D error signal.
- `self_critique_view_reliability_v37.py` predicts reliability scores from the already-refined tokens and supervises them against the reprojection error *after* triangulation (`motionflow_mv/fusion/self_critique_view_reliability_v37.py:71-105`; `motionflow_mv/fusion/omniview_fusion_v5.py:1547-1567`). This creates a feedback loop: reliability is estimated from features that themselves depend on the same unreliable views.
- The v39 coupling runs v37 before v36 so that reliability can gate the uncertainty gates (`motionflow_mv/fusion/omniview_fusion_v5.py:1135-1183`), and v43 further scales the residual by the final node gate (`motionflow_mv/fusion/uncertainty_gated_iterative_graph_refinement_v36.py:326-328`).

**Impact:** Empirically, the complex stack overfits quickly. The local/A800 snapshot shows v36 reaches 26.42 mm epoch-1 but then overfits to 76.88 mm, while v37 is worse than v36 at 26.94 mm (`docs/results_snapshot_2026_08_09.md`). The extra capacity is being spent modeling its own estimates rather than the true multi-view geometry.

---

## 3. View-Order and Variable-View Handling Remain Brittle

The architecture is still built around a fixed number of ordered views. The learned `view_pos_embed` is always added, even when optional camera-conditioned embeddings or set aggregators are enabled:

```python
view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
feat = feat + view_emb
```
(`motionflow_mv/fusion/omniview_fusion_v5.py:1034-1035`).

Optional permutation-invariant components (`use_camera_view_embedding`, `use_set_view_aggregator`, `use_perceiver_aggregator`) exist but are off by default and do not remove the fixed-view positional embedding. View masks are applied additively/zeroing (`_build_view_attention_mask`, `out * view_mask`), but the model has no true set-invariant pooling over views.

**Impact:** Accuracy is capped because the model is not robust to variable view counts, missing views, or different camera orderings. Cross-dataset mixing (H36M, MPI, WebBridge) and real-world capture rigs with dropped cameras therefore remain harder than necessary, even with the v38 expanded manifest and v41 domain-weighted loss.

---

## Summary

The v31-v43 additions address real problems—cross-view consistency, outlier views, temporal coherence, and physical plausibility—but their impact is limited by:

1. **Soft geometry:** geometry is advisory attention bias, not a hard fusion constraint.
2. **Self-referential gating:** uncertainty/reliability modules learn from features and post-hoc reprojection, leading to overfitting.
3. **Fixed-view assumptions:** learned view embeddings and ordered attention prevent true variable-view, permutation-invariant fusion.

Until these are addressed, the simpler v25 geometry-fusion baseline is likely to remain the most accurate path.
