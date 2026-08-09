# v50: Sparse-View Triangulation Agreement Loss (v50-STAL)

## Architecture

`v50_SparseViewTriangulationAgreementLoss` sits immediately after the v45 Adaptive Geometry Fusion / v46 Sparse-View Generalization fusion head and before v47 Temporal Aggregation. The module is parameter-free and identity-at-init: it samples a small, fixed set of view subsets (e.g. pairs, triplets, and quadruplets), runs the existing DLT triangulator on each subset independently, and penalises disagreement between these subset triangulations and the final fused 3-D pose. Each subset is weighted by its inverse triangulation uncertainty and by the v45/v46 per-(view,joint) reliability, so noisy/occluded subsets contribute less. Gradients are stopped into the raw 2-D keypoints and flow only through the fused 3-D pose, keeping the geometry block stable.

## New config flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v50_sparse_view_triangulation_agreement` | bool | `False` | Master switch. |
| `v50_stal_weight` | float | `0.01` | Overall loss weight λ. |
| `v50_stal_view_subsets` | list | `[2, 3, 4]` | View-counts of subsets to sample. |
| `v50_stal_num_subsets_per_forward` | int | `4` | Subsets sampled per forward pass. |
| `v50_stal_use_uncertainty_weighting` | bool | `True` | Weight by inverse triangulation uncertainty. |
| `v50_stal_use_reliability_gate` | bool | `True` | Gate by v45/v46 reliability. |
| `v50_stal_warmup_epochs` | int | `1` | Linear ramp-up of λ. |
| `v50_stal_huber_delta` | float | `50.0` | Huber delta in millimetres. |

## Loss term

For each sampled subset `s`:

```
L_s = Huber( || T_s − pred_3d || , δ )
L_STAL = λ · ramp · Σ_s ( w_s / Σw ) · L_s
w_s = uncertainty(s)^(−1) · mean(reliability_s)
```

`T_s` is the DLT triangulation from subset `s`, `pred_3d` is the fused pose, `ramp` is the linear warmup factor, and `uncertainty(s)` is the mean per-joint reprojection residual of the subset triangulation. Default `λ = 0.01`; start ablations at `λ = 0.001`.

## Evaluation metric

Primary metrics are the standard `val_MPJPE` and sparse-view `MPJPE@k` (k = 2, 3, 4, full). Additional diagnostics: `triang_agreement_mm` (mean disagreement between subset triangulations and fused pose), `subset_uncertainty_mean`, and `MPJPE@subset_vs_full` to catch regressions on full views.

## Expected MPJPE impact

Local smoke (d=64, 500 samples) on top of the v46-SVG baseline (reported epoch-1 val_MPJPE = 32.97 mm) should show `MPJPE@2` improve by approximately 2–3 mm, `MPJPE@3` by 1–2 mm, and `MPJPE@full` remain within 0.5 mm. On the A800 full run the same pattern is expected, with the largest relative gain in the sparse-view regime where subset triangulations are most informative.

## Main risk / mitigations

| Risk | Mitigation |
|------|------------|
| **Two-view DLT triangulations are degenerate or noisy.** | Skip subsets with uncertainty above a fixed threshold; use Huber loss; clamp gradients. |
| **Sampling subsets is expensive when V is large.** | Cap samples at `v50_stal_num_subsets_per_forward=4`; pre-compute ray directions; reuse the same random seed within an epoch for caching. |
| **Loss competes with v40 physical loss and v48 domain loss.** | Start with `λ=0.001`, linear warmup, and ablate against the v48 baseline before stacking. |
| **Gradients through triangulation may destabilise training.** | Stop gradients into 2-D inputs; only back-propagate through `pred_3d`. |

## Dependencies and wiring

Depends on v45-AGF, v46-SVG, and v37 reliability. Wire into `experiments/train_omniview_fusion_v5_webbridge_multi.py` after the supervised loss, add flags to `motionflow_mv/fusion/omniview_fusion_v5.py`, and smoke-test with `configs/benchmark_v50_stal_smoke.yaml`.
