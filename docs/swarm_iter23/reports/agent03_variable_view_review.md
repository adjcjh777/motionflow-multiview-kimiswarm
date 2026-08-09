# Agent-03 Review: Variable-View Components for v46 Sparse-View Generalization

**Scope:** Review existing `VariableViewSetAggregator` and variable-view training code and identify concrete integration points for the v46 Sparse-View Generalization (SVG) module.

**Author:** Agent-03 (v46 swarm)  
**Branch:** `v46-svg`  
**Tracking issue:** #160  
**Date:** 2026-08-09

---

## 1. Components reviewed

| File | Role for v46 | Key observations |
|------|--------------|------------------|
| `motionflow_mv/fusion/variable_view_set_aggregator.py` | Set-aggregator backbone | `VariableViewSetAggregator` already consumes `(B,T,V,J,d)` tokens and a `(B,T,V)` mask, returns permutation-equivariant tokens with masked views zeroed. Uses Induced Set Attention Blocks (ISAB). |
| `motionflow_mv/fusion/variable_view_inference.py` | Inference-time hardening | `VariableViewInferenceWrapper` / `HardenedVariableViewInferenceWrapper` mask/pad inactive views and fall back to DLT for very few views. No training-time behaviour. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Main integration target | `OmniMultiViewFusionV5` already pipelines `view_mask` through: feature extraction → set/perceiver aggregator → hierarchical blocks → ST transformer → weight head → v25 refinement. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Training-loop hooks | `augment_clip()` supports `variable_view_subset` and `view_dropout_rate`; the trainer already has `use_variable_view_training`, a curriculum, and domain-aware clamping. |
| `experiments/eval_variable_views.py` | Variable-view eval baseline | Evaluates all `C(V,k)` (or sampled) subsets and reports mean MPJPE per `k`. Does **not** yet emit `MPJPE@k` or robustness curves. |

---

## 2. Detailed integration points

### 2.1 Reuse `VariableViewSetAggregator` as a mask-aware token stage

`VariableViewSetAggregator.forward(x, view_mask)` accepts a `(B,T,V,J,d)` tensor and a `(B,T,V)` or `(B,V)` mask. It already:

- expands the mask to `(B*J*T, V)` for `nn.MultiheadAttention` `key_padding_mask`,
- zeros out masked views in the output.

This makes it a natural pre-processing block for the proposed `SparseViewGeneralizationV46`:

- **Integration point:** v46 can be instantiated *after* the existing set aggregator (or after the perceiver aggregator) and before the ST transformer, ensuring the model processes an arbitrary subset of views as a true unordered set.
- **Caveat:** the current `OmniMultiViewFusionV5` still adds a *learned* `view_pos_embed` (`omniview_fusion_v5.py:1043`) before the aggregator. For v46 we should keep the camera-conditioned embedding (`use_camera_view_embedding`) or camera-conditioned v31 embedding so that dropped views do not leave empty positional slots.

### 2.2 Hook v46 reliability head into the v25 triangulation path

`MultiViewGeometryFusionV25` already accepts a `view_mask` and multiplies confidences by it before DLT (`multiview_geometry_fusion_v25.py:512`). The v45 `AdaptiveGeometryFusionV45` predicts per-view/per-joint reliability weights from reprojection residuals and multiplies them into the DLT weights.

- **Integration point:** v46’s reliability head can be modeled as a **sparse-view variant of v45**: instead of using reprojection residuals alone, it predicts reliability from the *set-aggregated tokens* and the masked confidence, then feeds those reliabilities into `MultiViewGeometryFusionV25`.
- **Minimal change:** add a new kwarg `use_v46_sparse_view_generalization` to `MultiViewGeometryFusionV25` that optionally constructs and calls a lightweight `SparseViewGeneralizationV46` head whose output is multiplied with `confidence` before DLT, exactly as v45 does.

### 2.3 Training loop: extend the existing variable-view augmentation

`experiments/train_omniview_fusion_v5_webbridge_multi.py` already implements:

- `augment_clip(..., view_dropout_rate, variable_view_subset, min_views)` (line ~614),
- a full variable-view curriculum via `--use_variable_view_training` (line ~1037),
- mixed-loader `base_view_mask` based on `dataset_id` (line ~1017).

**Recommended v46 integration (minimal, no rewrite):**

1. Add CLI flags:
   - `--use_v46_sparse_view_generalization`
   - `--v46_svg_view_dropout_prob` (default 0.3)
   - `--v46_svg_min_views` (default 2)
   - `--v46_svg_use_curriculum` (default True)
2. Reuse `augment_clip()` by plumbing `v46_svg_view_dropout_prob` into the existing `view_dropout_rate` argument.
3. Add a second call path: when v46 is enabled but the legacy `use_variable_view_training` is off, still generate a `view_mask` from the dropout and pass it to `model.forward(..., view_mask=view_mask)`.
4. Ensure the mixed-loader `base_view_mask` is composed with the v46 mask (currently `view_mask = view_mask * base_view_mask`; this composition can stay).

### 2.4 Model forward: add v46 flag and insert the module

In `OmniMultiViewFusionV5.__init__`:

- Add constructor arguments:
  - `use_v46_sparse_view_generalization`
  - `v46_svg_hidden`
  - `v46_svg_min_views`
  - `v46_svg_dropout`
- Instantiate `self.sparse_view_generalization_v46 = SparseViewGeneralizationV46(...)` only when the flag is on.

In `OmniMultiViewFusionV5.forward`:

- After the set/perceiver aggregator (`omniview_fusion_v5.py:1050-1054`), optionally call the v46 module:

```python
if self.use_v46_sparse_view_generalization and self.sparse_view_generalization_v46 is not None:
    feat = self.sparse_view_generalization_v46(feat, view_mask=view_mask_flat.view(B, T, V))
```

- The v46 module should return updated tokens of the same shape; any per-view reliability weights it produces can be passed down to the v25 geometry fusion block.

### 2.5 Evaluation: extend `eval_variable_views.py` to report `MPJPE@k`

The current script reports per-subset mean MPJPE. To satisfy the v46 success criteria, extend it to:

- Accept an `--mpjpe_at_k` argument, e.g. `2,3,4,full`.
- For each requested `k`, report `MPJPE@k` as the mean over all `C(V,k)` (or sampled) subsets.
- Save a CSV/JSON with columns: `k, mean_mm, std_mm, n_subsets`.
- Optional: add a "dropout robustness curve" by sweeping `k` and plotting mean vs. `k`.

No source-code changes are needed for this analysis task; the recommendations above are the integration points for the IMPLEMENT agents (Agent-06 through Agent-13).

---

## 3. Potential conflicts and mitigations

| Risk | Mitigation |
|------|------------|
| v46 reliability head overlaps with v45 `AdaptiveGeometryFusionV45` | Make v46 operate on *set-aggregated tokens* rather than reprojection residuals; it is complementary to v45. If both flags are on, multiply v46 and v45 weights. |
| Variable-view training already exists; v46 could duplicate it | v46 should be framed as a **learned reliability head + sparse-view module**, not a replacement for the augmentation. Reuse `augment_clip` and the existing `view_mask` plumbing. |
| Set aggregator is gated by `use_set_view_aggregator`; v46 needs it | Document that v46 is most effective with `use_set_view_aggregator=True` (or `use_perceiver_aggregator=True`). The module itself should still work without it but with weaker sparse-view generalization. |
| Mixed-loader `base_view_mask` from `dataset_id` may clash with random dropout | Compose masks by multiplication (`view_mask = v46_mask * base_view_mask`) as the trainer already does. |

---

## 4. Summary for implementers

1. **Agent-06** (`sparse_view_generalization_v46.py`): implement a module that consumes `(B,T,V,J,d)` tokens and `(B,T,V)` mask and outputs updated tokens and optionally reliability weights. Keep it small (lightweight MLP / ISAB).
2. **Agent-08** (`omniview_fusion_v5.py`): add `use_v46_sparse_view_generalization` flag, instantiate the v46 module, and insert it after the set aggregator / before the ST transformer.
3. **Agent-09** (`train_omniview_fusion_v5_webbridge_multi.py`): add CLI flags and wire `v46_svg_view_dropout_prob` into the existing `augment_clip` view-dropout path.
4. **Agent-13** (`eval_variable_views.py`): extend to emit `MPJPE@k` CSV/JSON.
5. **Agent-12** (tests): add unit tests that exercise v46 with random view masks and confirm masked views have zero weight after the v46 block.

---

## 5. Files referenced

- `motionflow_mv/fusion/variable_view_set_aggregator.py`
- `motionflow_mv/fusion/variable_view_inference.py`
- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`
- `motionflow_mv/fusion/adaptive_geometry_fusion_v45.py`
- `motionflow_mv/fusion/perceiver_view_aggregator.py`
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- `experiments/eval_variable_views.py`
- `tests/test_variable_view_inference_hardened.py`

---

## 6. No blockers

The existing variable-view and set-aggregator infrastructure is already mature enough to host v46. The main implementation work is wiring a new lightweight module into the v5 forward path and reusing the existing `view_mask` and augmentation plumbing rather than inventing a parallel mechanism.
