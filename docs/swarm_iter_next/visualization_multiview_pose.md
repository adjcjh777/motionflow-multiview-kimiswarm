# visualization_multiview_pose – iteration notes

## Direction
Build lightweight, smoke-testable visualization tools that compare multi-view pose estimators on a common validation clip. The immediate goal is to help diagnose why local RTX 4090 runs of v26/v27/v28 overfit while the A800 v25 small baseline stays at 18.31 mm val_MPJPE.

## What already exists
- `scripts/visualize_multiview_pose.py` – renders multi-view skeleton grids for a single pose.
- `scripts/visualize_variable_view_failure.py` – diagnostic plots for variable-view inference (v2/v3 models).
- `scripts/visualize_v25_geometry_attention.py` – visualises v25 geometry-aware attention maps and geometry biases.
- `experiments/visualize_fusion.py`, `visualize_temporal_v1.py`, `visualize_residual_corrections_v1.py` – older single-model visualizers for earlier ray-attention variants.
- `docs/swarm_iter11_visualization_tools_report.md` – roadmap proposing a multi-model comparison dashboard, uncertainty calibration plots, and failure atlases.

## What was added in this iteration
- `scripts/visualize_model_comparison.py`
  - Loads a canonical WebBridge/H36M `.npz` clip (or synthetic data with `--smoke`).
  - Triangulates a DLT baseline.
  - Optionally loads a trained model checkpoint via `--checkpoint` and `--model_class`.
  - Produces:
    - `per_frame_mpjpe.png` – MPJPE time series.
    - `per_joint_mpjpe.png` – per-joint MPJPE bar chart.
    - `error_heatmap.png` – per-joint per-frame error heatmap.
    - `summary.json` – scalar metrics (MPJPE, PA-MPJPE).
- `tests/test_visualize_model_comparison.py` – CPU smoke tests.

## Smoke test result
```bash
python scripts/visualize_model_comparison.py --smoke --out_dir outputs/model_comparison_smoke
```

```
DLT MPJPE: 1.97 mm
DLT PA-MPJPE: 2.07 mm
```

Real H36M `s_01_acts_02` clip (2995 frames, 4 views):
```
DLT MPJPE: 1.95 mm
DLT PA-MPJPE: 3.14 mm
```

## Next steps
1. Add `--second_checkpoint` / `--labels` support so two local checkpoints (e.g. v25 small vs a 4090 checkpoint) can be compared directly on the same clip.
2. Extend the script to load the current best model class (`MultiViewGeometryFusionV25`) with sensible defaults.
3. Add per-view reprojection error plots and uncertainty heatmaps once the advanced uncertainty model is being evaluated.
4. Use this tool on the local 4090 v25 baseline once its checkpoint is ready, to visualise where it diverges from the A800 v25 small baseline.
