# MotionFlow Multi-View: Paper Direction v2

## Problem
MotionFlow is currently monocular. In practice, the same action is often captured by multiple synchronized cameras. We want to extend MotionFlow to fuse per-view 2D/3D estimates into a single, physically consistent 3D skeleton, and feed it back into the MotionFlow pipeline.

## Core idea
Build a **minimal, modular multi-view fusion pipeline** that can be plugged into MotionFlow:

```
Multi-view videos
      |
      v
Per-view 2D keypoint + confidence (existing MotionFlow single-view module)
      |
      v
Calibration / camera model
      |
      v
Fusion model (DLT baseline + learned variants)
      |
      v
3D skeleton in world coordinates
      |
      v
Optionally: temporal smoothing / learned refinement
```

## What we have validated (Shelf 300–600, 5 views)

| Iteration | Model | mean (px) | median (px) | max (px) |
|-----------|-------|-----------|-------------|----------|
| Baseline | DLT | 9.88 | 5.52 | 1044.68 |
| 5 | RobustTriangulationModel | 11.64 | 5.98 | 5739.47 |
| 6 | ResidualRefinerModel | 9.90 | 5.52 | 1038.20 |
| 7 | TemporalRefinerModel (window=5) | 9.89 | 5.49 | 1044.45 |
| 7 | TemporalRefinerModel (window=9) | 9.89 | 5.49 | 1044.35 |
| 8 | Temporal (synthetic pre-train + fine-tune) | 9.94 | 5.53 | 1044.67 |
| 9 | Temporal (A800-D, hidden=256, d=128, window=15) | 9.97 | 5.49 | 1044.66 |

## Key findings
1. **Geometric DLT is a very strong baseline**. Any learned fusion model must provide a value beyond pure triangulation.
2. **Reprojection-only supervision is insufficient**. Learned models can match DLT but not beat it, because DLT already minimizes a related geometric objective.
3. **Temporal consistency gives marginal gains**. Median improves slightly, but outliers remain.
4. **Synthetic pre-training does not transfer**. Domain gap and lack of real 3D labels limit transfer.
5. **Scaling up model/capacity on A800-D does not help**. A much larger temporal model (hidden=256, d=128, window=15) still only matches DLT, confirming the bottleneck is supervision, not compute.

## Paper-worthy contribution
Instead of claiming to beat DLT on reprojection error, frame the paper around:

1. **A practical multi-view extension of MotionFlow**:
   - Defines the interface between single-view 2D extraction and multi-view fusion.
   - Supports arbitrary number of calibrated cameras.
   - Produces world-coordinate 3D skeletons.

2. **A systematic empirical study of fusion choices**:
   - DLT baseline, learned per-view weighting, residual refinement, temporal refinement, synthetic pre-training.
   - Quantitative comparison on real data.

3. **Open engineering artifacts**:
   - Modular `motionflow_mv` package.
   - Reproducible training/evaluation scripts.
   - All results tracked via GitHub Issues/PRs.

## Future work to establish a clear advantage
- Use a dataset with **real 3D ground truth** (Human3.6M, Panoptic) and train with 3D loss.
- Add **explicit skeleton / bone-length priors** to reduce physically implausible poses.
- Integrate with the actual MotionFlow single-view estimator and demonstrate end-to-end multi-view inference.

## Proposed next step (Iteration 9)
Run the pipeline end-to-end on a small multi-view video set (or Shelf frames) using the existing 2D predictions, produce 3D skeletons, and demonstrate integration with MotionFlow's output format.

A first demo is provided in `experiments/run_multiview_pipeline_shelf.py`, which triangulates all matched Shelf frames and saves the 3D skeletons to `outputs/shelf_pipeline_3d.pkl`.
