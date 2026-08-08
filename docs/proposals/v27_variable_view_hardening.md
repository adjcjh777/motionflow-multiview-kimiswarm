# v27: Variable-View Hardening

**Task identifier:** `design_v27_variable_view_hardening`  
**Depends on:** v25 (`docs/proposals/v25_multiview_geometry_fusion.md`), v26 (`docs/proposals/v26_temporal_geometry_fusion.md`)

## 1. Problem

v25 and v26 are *variable-view compatible* — they accept `view_mask` and an arbitrary number of views — but they are not yet *variable-view robust*. Three concrete gaps exist:

1. **Outlier detector is present but idle.** `OutlierViewDetector` lives in `motionflow_mv/fusion/outlier_view_detector.py` and is instantiated by `MultiViewGeometryFusionV25` (flags `use_outlier_view_detector`, `outlier_z_thresh`, `outlier_soft_beta` in `motionflow_mv/fusion/multiview_geometry_fusion_v25.py:397-426`), yet the current v25/v26 forward path does not actually apply its soft weights to triangulation or loss weighting. It is a no-op unless explicitly wired in.
2. **Training augmentation is blind.** `experiments/train_omniview_fusion_v5_webbridge_multi.py:831-837` already injects synthetic outlier views, but the model receives no explicit supervision to learn robust rejection; the geometry losses still treat all active views equally.
3. **No view-count conditioning.** The model uses learned `view_pos_embed` plus camera-conditioned embeddings, but has no explicit embedding of the *number of active views*. This makes few-view (2–4) and many-view (8–14) behavior share the same feature statistics, which hurts the low-view tail.

## 2. Proposed method

A small, warm-startable hardening layer that reuses existing v25/v26 components and adds minimal new code.

### 2.1 Wire the outlier detector into v25/v26 triangulation

In `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`, after the initial DLT triangulation (`triangulate_initial`) and inside `DepthProposalTriangulation`, multiply the per-view triangulation weights by the outlier detector output:

```python
# Inside MultiViewGeometryFusionV25.forward, after pred_3d_init is available
if self.use_outlier_view_detector:
    outlier_weights, _ = self.outlier_view_detector(
        pred_3d_init, points_2d, K, R, t, view_mask=view_mask
    )  # (B, T, V, J)
    # Reshape to (B*T, V, J) and multiply into triangulation/attention weights.
    weights = weights * outlier_weights
```

The detector is identity-at-init (`residual_scale=0` → weights ≈ 1), so enabling it on a v25 checkpoint does not perturb training. It only learns to down-weight views when the geometry loss or MPJPE benefits.

### 2.2 Add a view-count embedding

Add a tiny learned embedding keyed by the number of active views:

```python
self.view_count_embed = nn.Embedding(15, d)  # supports 2–14 views + padding
```

In the forward pass, compute `active_count = view_mask.sum(dim=-1)` (integer in `[2, 14]`) and add `view_count_embed[active_count]` as a broadcast residual to the ray tokens before geometry attention. This lets the geometry attention and depth head adapt their priors to few-view vs. many-view regimes. Default to all-zero embedding at init to keep the block an identity residual.

### 2.3 Variable-view curriculum during training

Add a data-layer helper `VariableViewCurriculum` that, per epoch, samples a target active-view count from a schedule:

| Epoch range | Active view distribution | Notes |
|-------------|--------------------------|-------|
| 0–2         | fixed 2 views            | Warm-start from v25/v26 checkpoint; force few-view reasoning. |
| 3–6         | uniform {2, 4}             | Expand to the low-view tail. |
| 7–12        | uniform {2, 4, 8}          | Mid-range. |
| 13+         | uniform {2, 4, 8, 14}      | Full variable-view training. |

The schedule is a constructor argument so it can be disabled by setting `curriculum_epochs=0`.

### 2.4 Cross-subset consistency loss

When `T ≥ 2` and at least one frame has ≥ 4 active views, randomly drop one active view to produce a second triangulation. Add a consistency loss:

```python
L_cons = mean_j || pred_3d(V_active) - pred_3d(V_active \ {v}) ||_2
```

This is computed only for in-batch examples where `active_count ≥ 4`; otherwise it is zero. Weight: `v27_consistency_loss_weight=0.05`.

### 2.5 Integration points

| File | Change |
|------|--------|
| `motionflow_mv/fusion/multiview_geometry_fusion_v25.py:397-426` | Accept and wire `use_outlier_view_detector` into triangulation weights; add `view_count_embed`. |
| `motionflow_mv/fusion/temporal_geometry_fusion_v26.py` | Propagate the same changes because it wraps v25. |
| `motionflow_mv/fusion/omniview_fusion_v5.py:337-364` | Surface new toggles `v27_use_view_count_embed`, `v27_use_variable_view_curriculum`, `v27_consistency_loss_weight`; pass them into the v25/v26 modules. |
| `motionflow_mv/data/variable_view_curriculum.py` *(new)* | Small helper that maps epoch → active-view distribution and applies view masking. |

No changes to the core training loop are required; the curriculum is applied inside the data loader or via a lightweight collator.

## 3. Expected impact

Targets are relative to the best v25/v26 baseline on the same validation split.

| Metric | Target | Rationale |
|--------|--------|-----------|
| `val_MPJPE` full 14-view | −3 to −5% | Outlier rejection and view-count conditioning clean up triangulation weights. |
| 2-view `val_MPJPE` | −8 to −12% | Explicit few-view training + view-count embedding. |
| 4-view `val_MPJPE` | −5 to −8% | Consistency loss + curriculum. |
| 8-view `val_MPJPE` | −3 to −5% | Similar to full-view; smaller absolute gap. |
| 14-view `val_MPJPE` | −2 to −4% | Marginal improvement; baseline already strong. |

Absolute estimate (H36M/WebBridge mixed val): `val_MPJPE` drops from ~19.5 mm to ~18.5–18.9 mm if v25 small already clears the v18 baseline; otherwise the improvement is measured relative to that baseline.

## 4. Implementation cost

| Item | Estimate |
|------|----------|
| Lines of code | ~200–250 (detector wiring, embedding, curriculum helper, consistency loss, tests). |
| New files | `motionflow_mv/data/variable_view_curriculum.py`; update `tests/test_multiview_geometry_fusion_v25.py` and `tests/test_temporal_geometry_fusion_v26.py`. |
| Model parameters | `< 0.5%` increase (`view_count_embed` 15 × d + detector gate is already present). |
| Training time | +10–15% (extra forward pass for consistency loss; curriculum has negligible cost). |
| Data needs | None; reuses existing H36M/MPI/WebBridge loaders. |
| GPU memory | +<3% (one extra embedding lookup and one cached residual). |

## 5. Risks / mitigation

| Risk | How it appears | Mitigation |
|------|----------------|------------|
| Outlier detector collapses to all-zero weights. | `outlier_weights` → 0 → triangulation degenerates. | Initialize `residual_scale=0` and start with `v25_outlier_soft_beta=1.0`; monitor mean outlier weight per epoch. |
| View-count embedding overfits to training camera counts. | Large gap between 2-view and 14-view val MPJPE. | Clamp embedding norm; validate on held-out camera subsets. |
| Consistency loss over-smooths fast motion. | Temporal MPJPE rises relative to per-frame. | Only apply when `active_count ≥ 4`; weight is small (`0.05`). |
| Curriculum is too aggressive early on. | Training loss spikes in first epochs. | Default starts from a v25/v26 warm checkpoint and ramps over 12 epochs. |
| Variable-view masking conflicts with fixed-view positional embeddings. | `view_pos_embed` still expects `V` fixed slots. | Keep `view_count_embed` as a residual; do not remove `view_pos_embed`. |

## 6. Minimal experiment plan

### 6.1 Flags / config additions

```yaml
model:
  use_multiview_geometry_fusion_v25: true
  v27_use_view_count_embed: true
  v27_use_variable_view_curriculum: true
  v27_consistency_loss_weight: 0.05
  v25_use_outlier_view_detector: true
  v25_outlier_z_thresh: 3.0
  v25_outlier_soft_beta: 1.0

training:
  warm_start: outputs/v25_small/latest.pth
  curriculum_epochs: 12
  curriculum_schedule: [[0, 2], [3, 6], [7, 12], [13, 999]]
  curriculum_view_counts: [[2], [2, 4], [2, 4, 8], [2, 4, 8, 14]]
```

### 6.2 Smoke test

```bash
# 1. Unit test the detector wiring.
pytest tests/test_multiview_geometry_fusion_v25.py tests/test_temporal_geometry_fusion_v26.py -q

# 2. Smoke a 2-epoch training run on RTX 4090 / small WebBridge subset.
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --config configs/benchmark_webbridge_h36m_test_smoke.yaml \
  --use_multiview_geometry_fusion_v25 \
  --v25_use_outlier_view_detector \
  --v27_use_view_count_embed \
  --v27_use_variable_view_curriculum \
  --curriculum_epochs 2 \
  --epochs 2 \
  --batch_size 4
```

### 6.3 Evaluation protocol

After the smoke run succeeds, run the variable-view benchmark:

```bash
python scripts/eval_variable_view_v25.py \
  --checkpoint outputs/v27_variable_view_hardening_smoke/latest.pth \
  --view_counts 2 4 8 14 \
  --datasets h36m mpi webbridge
```

Success is defined as no regression at 14 views and a relative improvement of at least 5% at 2 and 4 views compared to the v25/v26 baseline run with the same checkpoint lineage.

## 7. Why this direction is smaller than the alternatives

Compared to the other v27 candidates in `docs/proposals/v27_next_iteration_decision_matrix.md`:

* **Uncertainty-aware depth proposals** changes the triangulation core and requires a new distribution head.
* **Camera refinement inside the fusion loop** risks the same v21-style camera regression.
* **Diffusion-based pose refiner** is high cost and high inference latency.

Variable-view hardening, by contrast, uses the outlier detector and variable-view infrastructure already present in the repo, adds no new model stage, and is warm-startable from any v25/v26 checkpoint. It directly attacks the most visible failure mode of the current pipeline: robustness when the number or quality of views varies.

## 8. References

* `motionflow_mv/fusion/omniview_fusion_v5.py:337-364` — v25/v26 instantiation and flags.
* `motionflow_mv/fusion/multiview_geometry_fusion_v25.py:397-426` — outlier detector parameters.
* `motionflow_mv/fusion/outlier_view_detector.py:22-56, 59-144` — reprojection residual and soft down-weighting.
* `motionflow_mv/fusion/variable_view_set_aggregator.py:80-211` — existing permutation-invariant view aggregation.
* `motionflow_mv/fusion/variable_view_inference.py:335-515` — inference wrappers for variable views.
* `experiments/train_omniview_fusion_v5_webbridge_multi.py:831-837, 1384-1387` — existing outlier-view augmentation.
