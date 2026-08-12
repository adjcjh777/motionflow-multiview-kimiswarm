# 2/3-View Failure Analysis and Visualisation — T18

**Tracking issue:** #76  
**Branch:** `feat/swarm-iter20-v4`  
**Date:** 2026-08-07  
**Author:** Agent T18

## 1. Motivation

The single biggest blocker for OmniMultiViewFusion v4 is the catastrophic
collapse when fewer than four views are available at inference time.  On the
H36M v2 dense-graph A800 checkpoint the numbers are stark:

| Active views | Mean MPJPE (mm) | Std (mm) |
|-------------:|----------------:|---------:|
| 2 | **1990.56** | 750.11 |
| 3 | **1619.90** | 729.20 |
| 4 | 14.99 | 0.00 |

This is a real-world deployment issue: capture rigs frequently lose cameras to
occlusion, motion blur, or calibration drift.  Real-time multi-camera setups may
only have 2–3 reliable views for some joints or frames.  The goal of T18 is to
diagnose *why* the model fails so severely and to provide reusable tooling that
lets every v4 ablation expose the same signals.

## 2. Deliverables

* `scripts/visualize_variable_view_failure.py` — CPU/GPU diagnostic script that
  loads a v2/v3/v4 checkpoint, runs inference with k=2,3,4 active views, and
  writes per-joint errors, view weights, visibility predictions, and
  triangulation reprojection residuals.
* This report — `docs/swarm_iter20/failure_analysis_2_3_views.md` — documenting
  the failure hypotheses and recommended experiments.

## 3. Methodology

The script works as follows:

1. **Load model and data.**  Either a real `.npz` dataset
   (`points_2d`, `confidences`, `joints_3d`, `camera_K/R/t`) or a synthetic
   smoke dataset is used.  The script auto-detects architecture hyper-parameters
   from the checkpoint state dict.
2. **Variable-view masking.**  It re-uses
   `VariableViewInferenceWrapper` to zero the confidence of dropped views while
   keeping the fixed-view model unchanged.
3. **Diagnostics collected per k.**
   * `pred_3d` vs. ground-truth → per-joint MPJPE and overall MPJPE/PA-MPJPE.
   * `weights` from the model → mean triangulation weight per (view, joint).
   * `visibility` from the model → mean predicted visibility per (view, joint).
   * Reprojection residual per (view, joint) computed from the predicted 3D
     skeleton and the camera rig.
4. **Visual outputs.**  Matplotlib figures are written to
   `outputs/failure_analysis_variable_views/`.

Run the smoke test on CPU:

```bash
python scripts/visualize_variable_view_failure.py --smoke
```

Run on a real checkpoint:

```bash
python scripts/visualize_variable_view_failure.py \
    --checkpoint outputs/omniview_fusion_v2_h36m_d128_dense_graph_a800.pth \
    --dataset data/webbridge/h36m/S9/acts_02_multiview_m.npz
```

## 4. Smoke-test observations

Running the script with a freshly-initialised v2 model on synthetic data gives
the following baseline numbers (these are not meant to be accurate, only to show
that the tool runs end-to-end):

| k | MPJPE (mm) | PA-MPJPE (mm) | max per-joint error (mm) |
|---|-----------:|--------------:|-------------------------:|
| 2 | 140.29 | 17.22 | 147.83 |
| 3 |  95.23 | 15.57 | 107.16 |
| 4 | 137.94 | 12.21 | 146.83 |

The script produces:

* `per_joint_error_k*.png` — bar charts of per-joint MPJPE for each k.
* `view_weights_k*.png` — heat-maps of mean triangulation weights.
* `visibility_k*.png` — heat-map of predicted visibility.
* `triangulation_residual_k*.png` — reprojection residual per (view, joint).
* `residual_distribution_k*.png` — histogram of residuals.

## 5. Failure hypotheses

Based on the code path and the A800 result (2-view ~1990 mm, 3-view ~1620 mm,
4-view ~15 mm) we propose three concrete failure modes.  Each is testable with
the script above.

### Hypothesis 1 — Triangulation degeneracy from uncalibrated scale (most likely)

**Observation.** With k=4 the model triangulates correctly and the residual
refinement head has learned to clean up the DLT output.  With k<4 the DLT itself
is still mathematically valid, but the **adaptive Gauss-Newton refinement and the
residual MLP appear to have been trained with the implicit assumption that four
views are always present.**

The residual refinement head concatenates pooled features and the GN output:

```python
residual_input = torch.cat([feat_pooled, pred_3d_gn], dim=-1)  # (B*T, J, d+3)
delta = self.residual_mlp(residual_input)
pred_3d = pred_3d_gn + delta
```

If the distribution of `pred_3d_gn` (and the implicit scale of `delta`) was
learned almost exclusively from 4-view inputs, then dropping to 2–3 views
produces a `pred_3d_gn` whose scale lies outside the head's training domain.
The residual then adds an uncalibrated offset that blows up the skeleton.

**Test.** Compare the magnitude of `pred_3d_gn` vs. `pred_3d` for k=2,3,4.
If `||delta||` is orders of magnitude larger for k<4, this hypothesis is
confirmed.

**Mitigation.** Add explicit view-dropout augmentation during training (already
planned in v4) and/or make the residual head scale-aware (e.g. condition on the
number of active views, or use a skeleton-graph residual refiner whose dynamics
are tied to bone lengths).

### Hypothesis 2 — Attention collapse / view positional embedding leakage

**Observation.** The spatio-temporal transformer processes all `V` view slots
simultaneously and adds learned view positional embeddings:

```python
view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
```

When a view is masked by `apply_view_mask`, the pixel and confidence values are
zeroed, but the **positional embedding of the dropped view is still present**.
The model may therefore attend to the zero-padded view slot as if it were a
real view and propagate noise through the transformer layers.  For k=4 the noise
cancels; for k<4 the signal-to-noise ratio collapses.

**Test.** Look at the `view_weights_k*.png` heat-maps.  If inactive views
receive non-negligible weights or if active-view weights become highly uniform
(no clear dominant views), attention collapse is likely.  The visibility maps
(`visibility_k*.png`) should also show whether the model correctly predicts
zero visibility for dropped views.

**Mitigation.** Use a learned or fixed sentinel embedding for masked views, or
apply an hard attention mask so dropped views cannot be attended to.  The v4
`AdaptiveViewSelector` is a step in this direction.

### Hypothesis 3 — Visibility head fallback guard is too aggressive / not aggressive enough

**Observation.** The visibility head in v2/v3 is designed to force all views on
when fewer than `min_visible_views` are predicted visible.  The fallback guard
is:

```python
visible = (visibility > self.visibility_threshold).float()
visible_count = visible.sum(dim=1)
fallback = (visible_count < self.min_visible_views).float().unsqueeze(1)
effective_visibility = visibility + (1.0 - visibility) * fallback
```

With k=2 the model has no choice but to keep both active views.  With k=3 the
head may still decide (incorrectly) that one of the three views is occluded and
fall back to all three — which is fine — but the *weight* assigned to the
questionable view may be near zero, effectively performing 2-view triangulation
without a fallback guard.  Conversely, if the visibility head is over-confident,
it may assign high visibility to a view whose reprojection residual is large,
therefore corrupting the DLT.

**Test.** Overlay `visibility_k*.png` with `triangulation_residual_k*.png`.  If
high-visibility views also have large reprojection residuals for k<4, the
visibility head is miscalibrated for low-view regimes.

**Mitigation.** Train with the v2 standalone `VisibilityGatedFusionV2` head,
which uses per-joint context across views and an uncertainty channel, and add
an explicit loss that penalises high visibility when reprojection residuals are
large.

## 6. Additional candidate: coordinate-unit / scale issue

A fourth, related possibility is that the model internally mixes units.  When
k=4 the redundant views provide enough geometric constraints to keep the 3D
solution in metres.  With fewer views, any unit mismatch between the
principal-point correction output, the DLT scale, and the residual refinement
can magnify.  However, the fact that 4-view performance is good makes a pure
unit bug less likely than a distribution-shift / scale-domain problem
(Hypothesis 1).

## 7. Recommended next steps

1. **Run the script on the real H36M v2 dense-graph checkpoint.**  Compare the
   per-joint error bars, weight heat-maps, and residual heat-maps for k=2,3,4.
2. **Instrument the model to log `||delta||` and `||pred_3d_gn||`.**  This is
   the fastest way to validate or refute Hypothesis 1.
3. **Add hard view dropout to the v4 trainer.**  If Hypothesis 1/2 is correct,
   view-dropout augmentation is the cheapest fix.
4. **Replace the dense residual MLP with `SkeletonGraphResidualRefiner` in v4**
   so the residual correction respects bone-length constraints and cannot drift
   by metres when views are dropped.
5. **Add a scale / active-view-count embedding** to the residual refinement head
   so it can adapt to k=2,3,4.

## 8. Files produced by T18

* `scripts/visualize_variable_view_failure.py`
* `docs/swarm_iter20/failure_analysis_2_3_views.md`
* `outputs/failure_analysis_variable_views_smoke/` (smoke-test artefacts)

## 9. References

* `docs/results_h36m_v2_dense_graph_a800.md` — source of the 2/3/4-view MPJPE numbers.
* `docs/v4_architecture_design_proposal.md` — v4 design proposal including view-dropout and adaptive view selection.
* `motionflow_mv/fusion/variable_view_inference.py` — variable-view masking strategy.
* `motionflow_mv/fusion/visibility_gated_fusion_v2.py` — context-aware visibility head.
