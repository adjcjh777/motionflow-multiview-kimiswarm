# Direction 19: Interpretability & Failure Analysis

## Problem Statement

The current best model, `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`,
beats the baseline on MPI-INF-3DHP, but we do not yet know *why* it fails or whether its
internal signals—per-view fusion weights, learned principal-point (PP) correction, and
residual refinement—actually align with failure modes. A rigorous interpretability
profile will guide the P0 calibration/visibility directions and provide paper-ready
failure figures.

## Simplest Concrete Next Step

Run the existing failure-analysis script on a smoke subset of MPI-INF-3DHP S2/Seq1, then
post-process the saved artifact `failure_arrays.npz` with a small CPU-only correlation
script to quantify how fusion weights, PP correction, and residual corrections relate to
per-view and per-frame errors.

## Files to Touch

* `experiments/analyze_failures_crossview_pp.py` — existing failure-analysis script
  (read/run, no edits needed).
* `experiments/analyze_failure_correlations.py` — **new** post-processing script that
  loads the saved `.npz` and reports correlations.
* `docs/swarm_iter7/interpretability_failure_analysis.md` — this report.

### Rough sketch of the new script

```python
import numpy as np

data = np.load("outputs/failure_analysis_crossview_pp_smoke/failure_arrays.npz")
per_view_reproj = data["per_view_reproj_px"]   # (T, V)
mean_weights     = data["mean_weights"]        # (V,)
pp_delta_norm    = data["pp_delta_norm_px"]    # (T, V)
residual_norm    = data["residual_norm_mm"]    # (T, J)

view_err = per_view_reproj.mean(axis=0)
print("weight vs view-error r  =", np.corrcoef(mean_weights, view_err)[0, 1])
print("pp delta vs view-error r=", np.corrcoef(pp_delta_norm.mean(0), view_err)[0, 1])
```

## Expected Success Metric

A CPU-only interpretability run that finishes in minutes and yields:

* Per-joint, per-frame, and per-view error profiles (existing).
* Pearson correlations between interpretable signals and errors (new):
  * fusion weight vs. view reprojection error,
  * PP correction magnitude vs. view error,
  * residual correction magnitude vs. frame/joint MPJPE.

The near-term goal is not a number to beat, but a reproducible failure profile that feeds
back into the calibration and visibility P0 directions.

## Resource Requirements

CPU-only; no GPU training. Inference on the 500-frame smoke subset runs on the CPU in a
few minutes.

## Commands and Results

### 1. Run failure analysis (CPU)

```bash
python experiments/analyze_failures_crossview_pp.py \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth \
    --clip_len 13 --stride 13 --batch_size 32 --d 64 --n_st_layers 2 --residual_hidden 128 \
    --out_dir outputs/failure_analysis_crossview_pp_smoke \
    --report_dir docs/swarm_iter7 --device cpu --seed 42
```

Result on the smoke subset:

* MPJPE: **10.10 mm**
* PA-MPJPE: **7.53 mm**
* Mean residual correction: **60.50 mm**
* Full report: `docs/swarm_iter7/failure_analysis_crossview_pp.md`
* Arrays & plots: `outputs/failure_analysis_crossview_pp_smoke/`

### 2. Compute interpretability correlations (CPU)

```bash
python experiments/analyze_failure_correlations.py
```

Result (Pearson r):

| Signal pair | r |
|---|---|
| mean_weight vs mean_reproj (per view) | 0.460 |
| mean_weight vs median_reproj (per view) | 0.588 |
| pp_delta vs mean_reproj (per view) | nan |
| pp_delta vs median_reproj (per view) | nan |
| residual_magnitude vs frame_mpjpe (per frame) | 0.153 |
| residual_magnitude vs mpjpe (per joint) | 0.021 |

Interpretation:

* **Fusion weights** are *positively* correlated with reprojection error on this smoke
  subset, suggesting the current fusion mechanism is not reliably down-weighting bad
  views. This supports the visibility-aware adaptive fusion P0 direction.
* **PP correction** is clamped to a uniform magnitude across all views in the smoke
  subset (`nan` correlation), so it does not yet adapt per-view. The full dataset or a
  checkpoint with larger `principal_point_max_offset` should be checked.
* **Residual refinement** magnitude is only weakly correlated with frame/joint error,
  indicating the residual head applies diffuse corrections rather than targeting the
  worst frames/joints.

## Next Iteration Suggestions

1. Run the same pipeline on the *full* S2/Seq1 (6502 frames) once the GPU is free for
   the longer inference pass, or on a few representative sequences, to confirm the
   correlation findings.
2. Extend `analyze_failure_correlations.py` with Spearman rank correlation and
   per-sequence aggregation.
3. Add a scatter-plot helper to visualize weight-vs-error per view (paper figure).
4. Feed the findings back to the visibility-gated fusion direction: if weights still
   correlate positively with error, add an explicit reliability/uncertainty head.

## Commit

```bash
git add docs/swarm_iter7 outputs/failure_analysis_crossview_pp_smoke/failure_arrays.npz
git add experiments/analyze_failure_correlations.py
git commit -m "Add CPU interpretability failure-correlation script and smoke results"
```

If the push network fails, the local commit remains and this note records it.
