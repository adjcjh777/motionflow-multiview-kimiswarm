# v50 Outlier View Rejection v2

## Module Name

`OutlierViewRejectionV2` — geometric + learned reliability outlier rejection for sparse multi-view pose estimation.

## Architecture

`OutlierViewRejectionV2` sits between the v25/v45 geometry-fusion backbone and the v46 Sparse-View Generalization (SVG) head. It takes per-view 2-D keypoint predictions, camera parameters, and an initial triangulated 3-D pose, then iteratively refines a soft per-(view, joint) keep weight.

For each joint and each visible view, the module computes the normalized reprojection residual:

```
r_{v,j} = || P_v · X_j - x_{v,j} ||_2
```

where `P_v` is the projection matrix for view `v`, `X_j` the 3-D joint, and `x_{v,j}` the detected 2-D keypoint. A learned per-(view, joint) reliability score `ρ_{v,j}` (reusing the existing v37/v39 self-critique reliability head) is combined with the residual into a single outlier score:

```
s_{v,j} = σ( (r_{v,j} / τ - μ) / ρ_{v,j} )
```

`τ` is the soft residual temperature, `μ` a learned bias, and `σ` the sigmoid. The score is converted to a keep weight `w_{v,j} = 1 - s_{v,j}`.

The module is iterative: in each of `K` steps it re-triangulates the 3-D pose using the current weights in a weighted DLT step, recomputes residuals, and updates `w`. A sparse-aware guard ensures that at least `min_keep_fraction` of views for a given joint are retained, even if every residual is large. The final pose is fed to v46 SVG, and the updated reliability scores are optionally returned to the self-evolution feedback head.

All operations are differentiable except for the sparse-aware guard, which uses straight-through gradients; the module is identity-at-init because `ρ` is initialized to a high value and `μ` is initialized so that `w ≈ 1`.

## New Config Flags

| Flag | Default | Description |
|------|---------|-------------|
| `use_v50_outlier_view_rejection_v2` | `False` | Enable the module. |
| `v50_ovr_v2_num_iterations` | `2` | Number of residual/reliability refinement steps. |
| `v50_ovr_v2_soft_temperature` | `0.1` | Temperature for converting residuals to soft keep weights. |
| `v50_ovr_v2_residual_threshold` | `2.5` | Residual threshold in pixels (before normalization). |
| `v50_ovr_v2_min_keep_fraction` | `0.5` | Minimum fraction of views to retain per joint. |
| `v50_ovr_v2_reliability_gate_threshold` | `0.1` | Floor for reliability score to avoid division by zero. |
| `v50_ovr_v2_loss_weight` | `0.01` | Weight of the outlier-consistency auxiliary loss. |

## Loss Term

`L_outlier = - Σ_{v,j} [ w_{v,j} log ρ_{v,j} + (1 - w_{v,j}) log(1 - ρ_{v,j}) ]`

This binary-cross-entropy-style term encourages the learned reliability score `ρ` to agree with the geometrically-derived keep weight `w`. The total loss is `L_total = L_pose + v50_ovr_v2_loss_weight * L_outlier`. The default weight of `0.01` keeps the auxiliary term small relative to the main pose MSE.

## Evaluation Metric

Primary metrics are `MPJPE@k` for `k = 2, 3, 4` from `experiments/eval_variable_views.py`, plus the full-view `val_MPJPE`. We additionally monitor the Spearman correlation between per-view reliability and reprojection residual magnitude; the goal is `Spearman > 0.3`.

## Expected MPJPE Impact

Based on v33 outlier-view rejection smoke (`82.02 mm`) and the improved v46-SVG smoke (`32.97 mm`), we expect v2 to give the largest gains in the sparsest settings:

- `MPJPE@2`: **-3 to -5 mm**
- `MPJPE@3`: **-1.5 to -2.5 mm**
- `MPJPE@4`: **-0.5 to -1 mm**
- `val_MPJPE` (full views): **within ±0.5 mm** of the v46 baseline, with a small positive gain possible as cleaner triangulation propagates upward.

## Main Risk / Mitigations

**Risk: Over-aggressive rejection collapses useful views.** If the soft gate learns to drop too many views, sparse settings (`k=2`) can degrade because the module discards good observations that were only moderately noisy.

**Mitigations:**
1. `min_keep_fraction` hard floor guarantees at least half the available views are kept.
2. Identity-at-init: default parameters make `w ≈ 1`, so the module starts as a no-op and only learns to reject when training evidence justifies it.
3. Clamp `ρ` to `[0.1, 0.95]` so reliability never becomes a hard 0/1 mask.
4. Smoke-test with `v50_ovr_v2_loss_weight = 0.0` first to verify the base architecture before enabling the auxiliary loss.
