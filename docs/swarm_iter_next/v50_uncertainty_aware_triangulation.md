# v50 Uncertainty-Aware Triangulation with Iterative Residual-Guided Reweighting

## Architecture

`UncertaintyAwareTriangulationV50` replaces the single-pass v33/v45 triangulation in `motionflow_mv/fusion/omniview_fusion_v5.py` with a lightweight, gradient-safe iterative reweighting block.  For each joint it predicts two per-view quantities: a **heteroscedastic log-variance** `log σ²_vj` (view-dependent uncertainty) and a **residual-based outlier weight** `w_vj ∈ (0,1)`.  The block first triangulates using precision-weighted DLT with `1/σ²_vj`, then computes per-view reprojection residuals, and finally refines the weights for a second triangulation step.  Both weight families are initialized to identity (all-ones) so the module is a strict superset of v45 adaptive triangulation weights and can be trained end-to-end from a v46 checkpoint.  A sparse-view guard guarantees at least `min_views=2` remain after soft down-weighting, and a learned temperature `τ` controls the softness of the residual gate.

## New Config Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v50_uncertainty_aware_triangulation` | bool | `False` | Enable the module. |
| `v50_uat_num_iter` | int | `2` | Triangulation-reweighting iterations (kept ≤ 2 for gradient stability). |
| `v50_uat_hidden` | int | `64` | MLP hidden dim for log-variance and residual-weight predictors. |
| `v50_uat_min_views` | int | `2` | Minimum views retained after outlier soft-rejection. |
| `v50_uat_residual_temperature` | float | `1.0` | Temperature `τ` for residual-to-weight mapping. |
| `v50_uat_identity_init` | bool | `True` | Initialize all learned gates at identity (recommended for warm-start). |

## Loss Term

Add a per-joint reprojection negative log-likelihood term:

```text
L_uat = (1 / VJ) Σ_v,j  w_vj · [ log σ²_vj + (r_vj)² / σ²_vj ]
```

where `r_vj` is the 2-D reprojection residual for view `v` and joint `j`.  Weighted by `loss.v50_uat_weight`.

| Loss setting | Default |
|--------------|---------|
| `loss.v50_uat_weight` | `0.01` |
| `loss.v50_uat_residual_clip` | `50.0` mm |

The clipping bound prevents the first-epoch outliers from dominating training while the residual weights are still noisy.

## Evaluation Metric

Primary: `val_MPJPE@full` and `MPJPE@2/3/4` from `experiments/eval_variable_views.py`.  Secondary: per-view residual-reliability Spearman correlation ≥ 0.25 and the fraction of views whose outlier weight `w_v < 0.1` on H36M validation (outlier recall).

## Expected MPJPE Impact

Based on v46-SVG local smoke `val_MPJPE@full ≈ 32.97 mm`, we expect v50 to improve sparse-view triangulation by reducing the influence of unreliable dropped-in views.  Target on the same smoke: `MPJPE@2` −3 to −5 mm, `MPJPE@3` −2 to −3 mm, with `MPJPE@full` within 1 mm of the v46 baseline.  On A800 full runs the improvement should concentrate in the `MPJPE@2` column because full-view triangulation is already strong.

## Main Risk / Mitigations

| Risk | Mitigation |
|------|-----------|
| Re-introducing v27-style TTE instability (iterations collapse or diverge) | Cap iterations at 2; identity init; clamp residual gate to `[0.05, 0.95]`; freeze for the first 500 steps. |
| Outlier weights become too soft and wash out correct views | Add entropy regularizer `Σ w log w` and enforce `min_views=2` hard mask. |
| Regression on full-view accuracy if uncertainty is over-expressed | Identity init + small loss weight (`0.01`) guarantees the module starts as no-op and must earn its gain. |
| v46/v48 stack still queued; cannot validate A800 numbers yet | Smoke first on RTX 4090 with `configs/benchmark_v46_svg_smoke.yaml` warm-started from the existing v46 checkpoint. |
