# v51 Model-Level Triangulation Refinement (MLTR)

**Focus area:** `model_level_triangulation_refinement`  
**Narrative fit:** closes the self-evolution loop between 2-D evidence, camera geometry, and the final 3-D pose, with explicit gains under sparse-view and cross-domain conditions.

## 1. What and why

v45 taught the model adaptive per-view triangulation weights; v46 added sparse-view dropout and a reliability head. v50 SEFH will feed residuals back to update reliability. v51 pushes this one step further: it makes **triangulation itself** a learned, differentiable refinement step. Instead of a single DLT pass followed by correction heads, the model iteratively re-weights and re-projects the 3-D estimate using geometric residuals as feedback. This is the natural place to anchor the next round of self-evolution: the pose estimator becomes its own bundle-adjustment critic at the model level.

## 2. Architecture

**Module:** `ModelLevelTriangulationRefinerV51` → `motionflow_mv/fusion/triangulation_refiner_v51.py`

Inputs at each forward step:
- Initial 3-D joint positions `J_0 ∈ R^{J×3}` from the existing v45/v46 triangulation.
- 2-D keypoints `x_vj ∈ R^{V×J×2}` and calibrated cameras `(K_v, R_v, t_v)`.
- Per-view reliability scores `ρ_v` from v46/v50 SEFH (optional, concatenated as features).
- Ray directions `d_vj = R_v^T K_v^{-1} [x_vj; 1]` and baseline epipolar residual `e_vj`.

**Core block:**
1. Project the current 3-D estimate to each view, compute normalized reprojection residual `r_vj`.
2. Pack per-view per-joint tokens: `z_vj = MLP([J_j, d_vj, r_vj, e_vj, ρ_v])`.
3. Two-layer cross-view self-attention with masked attention for missing views (sparse-view safe).
4. Predict two outputs:
   - Per-view per-joint **precision weights** `α_vj ∈ (0, 1)` (sigmoid, clamped).
   - Per-joint **3-D residual correction** `Δ_j ∈ R^3` (zero-initialized).
5. Update: `J_refined = J_0 + Δ`, with an optional weighted mid-point reprojection using `α_vj` as DLT weights.

**Identity-at-init:** `Δ` is initialized to zero and `α_vj` to uniform, so the module is transparent at startup and warm-starts safely from v46/v50 checkpoints.

## 3. New config flags and defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_model_level_triangulation_refinement` | bool | `False` |
| `v51_mltr_hidden` | int | `64` |
| `v51_mltr_num_layers` | int | `2` |
| `v51_mltr_num_iterations` | int | `1` |
| `v51_mltr_residual_init_zero` | bool | `True` |
| `v51_mltr_precision_clamp_min` | float | `0.05` |
| `v51_mltr_precision_clamp_max` | float | `5.0` |
| `v51_mltr_use_epipolar_feature` | bool | `True` |
| `loss.v51_mltr_loss_weight` | float | `0.01` |

## 4. Loss term

```
L_mltr = loss.v51_mltr_loss_weight * (
    Huber(J_refined, J_gt)
  + mean_vj( α_vj · ||proj_v(J_refined)_j - x_vj||_2 )
  - λ_ent · mean_j H(α_·j)
)
```

- The first term supervises the refined 3-D pose directly.
- The second term forces precision weights to agree with reprojection evidence.
- The entropy term prevents collapse to a single dominant view, preserving sparse-view robustness.

## 5. Evaluation metric

- `MPJPE@k` for `k ∈ {2, 3, 4, full}` via `experiments/eval_variable_views.py`.
- `ΔMPJPE@2` vs. the v50 SEFH baseline.
- `Spearman(α_vj, reprojection_error_vj)` to verify that precision weights are geometrically meaningful.

## 6. Expected MPJPE impact

- `MPJPE@2`: −3 to −5 mm (largest gain, because sparse subsets benefit most from learned re-weighting).
- `MPJPE@3`: −2 to −4 mm.
- `MPJPE@full`: −0.5 to −1.0 mm (small but consistent; full-view DLT is already strong).
- Cross-domain (3DPW actual-mode): an additional −2 to −4 mm at `MPJPE@2` by letting precision weights compensate for calibration and detection drift.

## 7. Main risk

**Risk:** The refinement head may overfit to DLT bias or collapse to near-uniform weights, erasing the sparse-view gain.  
**Mitigation:** zero-initialize the residual; clamp precision to `[0.05, 5.0]`; freeze base weights for the first epoch; start with `loss.v51_mltr_loss_weight=0.001` before raising to `0.01`; cap iterations at one for smoke and A800 full runs.

## 8. Smoke and full-run plan

- **Smoke config:** `configs/benchmark_v51_mltr_smoke.yaml`.
- **Local smoke:** warm-start from best v50 SEFH checkpoint; accept if `val_MPJPE@full` is within 1 mm of baseline and `MPJPE@2` improves by ≥2 mm.
- **A800 full:** queue after v50 results; use `d=128`, 10k samples, 5 epochs, early stopping.

## 9. One-line summary

v51 turns triangulation from a fixed geometric routine into a learned, residual-aware, self-correcting model component, directly improving sparse-view and cross-domain accuracy while preserving the strong full-view baseline.
