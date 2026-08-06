# Visualization Tools Roadmap – Iter 11+ (ICRA/CVPR 2027)

## Current state

The project already has three useful but narrow visualization scripts:

- `experiments/visualize_fusion.py` – single-frame 2D reprojections, 3D skeleton overlay, and per-view/joint attention heatmap for `RayAttentionFusionModelV3`.
- `experiments/visualize_temporal_v1.py` – temporal-clip 3D pose GIF, joint trajectories, and per-frame MPJPE time series for `RayAttentionFusionModelTemporal`.
- `experiments/visualize_residual_corrections_v1.py` – per-joint raw-DLT vs. residual-corrected trajectories and a per-joint MPJPE bar chart.

The new all-in-one model, `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` (`motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`), returns only `pred_3d, weights, log_var, nll_loss`. None of the existing scripts target it, and none of the intermediate quantities (DLT estimate, Gauss-Newton refinement, residual delta, or cross-view attention) are exposed. Visualization therefore lags behind the modeling advances, which slows debugging and limits the figures we can put in the paper.

## Proposed improvements

### 1. Advanced-model visualizer with exposed intermediates

Add a dedicated script `experiments/visualize_advanced_model_v1.py` that runs the full model and renders:

- Per-view per-joint **uncertainty heatmap** (`log_var`).
- **Triangulation pipeline** comparison: raw DLT → Gauss-Newton refined → residual-corrected.
- Per-view **reprojection error maps** after the final prediction.
- Per-joint MPJPE and PA-MPJPE for each stage.
- Temporal trajectory plots for selected joints across all three stages.

To make this possible, the model’s `forward` must optionally return the DLT and GN estimates. This is a minimal, non-breaking change.

```python
# In ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py

def forward(self, x, cameras=None, K=None, R=None, t=None,
            n_iter=1, return_intermediates=False):
    # ... existing preprocessing ...
    pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)
    pred_3d_gn = _triangulate_weighted_gauss_newton(
        points_2d, weights, K, R, t, pred_3d_raw,
        num_iters=self.gn_iters, damping=self.gn_damping
    )
    pred_3d = pred_3d_gn
    for _ in range(max(1, int(n_iter))):
        residual_input = torch.cat([feat_pooled, pred_3d], dim=-1)
        delta = self.residual_mlp(residual_input)
        pred_3d = pred_3d + delta

    # ... reshape ...
    if return_intermediates:
        return pred_3d, weights, log_var, nll_loss, pred_3d_raw, pred_3d_gn, delta
    return pred_3d, weights, log_var, nll_loss
```

The visualizer would then produce a single-page figure with one row per pipeline stage, making it trivial to see where the remaining error lives.

### 2. Multi-model comparison dashboard

Create `experiments/compare_models_visual.py` to load any two checkpoints (e.g. the current best `ray_attention_temporal_crossview_residual` at ~11.17 mm and the new advanced model) on the **same validation clip** and output:

- Side-by-side 3D skeletons and trajectory overlays.
- Per-joint MPJPE bar chart with both models.
- Per-frame MPJPE time series overlay.
- PCK curves and AUC for the clip.
- Per-view error contribution bar chart.

Metrics computed with `motionflow_mv/eval/metrics.py` (`mpjpe_batch`, `pa_mpjpe`, `pck_batch`, `pck_auc`). This removes the common mistake of comparing numbers across different validation subsets.

### 3. Uncertainty and attention diagnostics

The uncertainty head is a key novelty, but there is currently no way to verify it is calibrated. Add:

- **Uncertainty calibration plot**: predicted `exp(log_var)` vs. observed squared reprojection error, binned by predicted uncertainty. Compute Pearson correlation and mean calibration error.
- **View-level attention heatmap**: capture attention weights from `self.view_attn` by calling `forward` with `need_weights=True` (PyTorch `MultiheadAttention` supports this). Plot head-averaged `(V, V)` attention for a representative joint to check whether views attend to geometrically consistent neighbors.
- **Temporal attention rollout** (optional): register hooks on `st_transformer` layers and visualize `(T*V, T*V)` attention for selected joints.

Use `need_weights=True` in `self.view_attn` to obtain `(N*J, V, V)` attention matrices.

### 4. Failure-mode atlas

Implement `experiments/visualize_failure_atlas.py` to rank validation frames by MPJPE and generate a compact grid of the worst cases. Each panel shows:

- Input 2D detections in one view.
- Predicted 3D skeleton vs. GT.
- Per-view contribution weights and uncertainty.
- Reprojection error per view.

This turns the numeric error into actionable patterns (e.g. self-occlusion, extreme depth, camera with bad calibration).

### 5. Publication-quality video renders

Extend the temporal visualizer to output:

- Rotating 3D skeleton video (MP4) using `matplotlib.animation` or Open3D.
- Optional multi-view 2D overlay video if raw camera frames are provided via `--frames_dir`.
- Consistent color scheme and skeleton topology for H36M and MPI-INF-3DHP, loaded from a small JSON file instead of hard-coded arrays.

## Experiments to run

1. **Advanced model smoke visualization** on the first MPI-INF-3DHP validation clip (`data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`). Goal: verify that GN + residual visibly reduces the per-joint error relative to raw DLT.
2. **Head-to-head comparison** of the new advanced model vs. the current 11.17 mm cross-view residual baseline on the same 100-frame clip. Track MPJPE, PA-MPJPE, PCK, AUC.
3. **Uncertainty calibration audit** on 1,000 validation frames. Target Pearson `r > 0.5` and calibration error < 10 px².
4. **Failure atlas** for the top-20 worst frames across MPI-INF-3DHP validation.
5. **Supplemental video production** for the best 5 clips for the paper supplement.

## Metrics to track

| Metric | Why it matters |
|---|---|
| MPJPE / PA-MPJPE | Primary 3D accuracy; must improve or at least match the 11.17 mm baseline. |
| PCK@50/100/150, AUC | Standard benchmarks for the paper tables. |
| Per-joint MPJPE | Reveals which joints the advanced model still misses. |
| Per-view reprojection error | Catches calibration or view-specific issues. |
| Uncertainty calibration (MSE vs. predicted var) | Validates the uncertainty head as a genuine contribution. |
| Attention entropy / view concentration | Sanity check that cross-view attention is geometrically meaningful. |

## Risks and mitigations

- **Model/visualization drift.** Put the `return_intermediates` flag in the model and version the visualizer with the model file.
- **Fragile attention extraction.** Wrap `need_weights=True`/hook calls in try/except and skip attention plots if unsupported.
- **Slow rendering.** Default to PNG frames/GIF; render MP4 only for final paper clips.
- **Dependency issues.** Keep the `Agg` backend and support headless execution.
- **Opportunity cost.** Build visualizers in parallel; the advanced-model visualizer is the highest priority.

## Recommended next step

Implement the non-breaking `return_intermediates` flag in `ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py` and the `experiments/visualize_advanced_model_v1.py` script. Run it on the MPI-INF-3DHP validation clip immediately after the next training run; the resulting per-stage error bars and uncertainty heatmaps will tell us whether the Gauss-Newton and uncertainty components are actually helping or are merely adding parameters.
