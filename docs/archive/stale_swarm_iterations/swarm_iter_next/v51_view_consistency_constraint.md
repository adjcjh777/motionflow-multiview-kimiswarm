# v51 View Consistency Constraint

**Focus-area agent:** view_consistency_constraint  
**Status:** design proposal, no code written, no GPU training launched.  
**Dependencies:** v46 sparse-view generalization, v50 self-evolution feedback head (warm-startable add-on).

## 1. Module proposal

**`ViewConsistencyConstraintV51`** (`motionflow_mv/fusion/view_consistency_constraint_v51.py`)

A lightweight, differentiable geometric-consistency head that forces the lifted 3-D pose to agree with the multi-view evidence. The module consumes the current 3-D pose estimate, 2-D keypoints, and calibrated camera parameters, then computes per-joint, per-view-pair residuals for:

- **Reprojection**: `||K_v [R_v | t_v] P_j − x_{v,j}||_2`
- **Epipolar**: `|x_{v,j}^T F_{vw} x_{w,j}|`
- **Triangulation agreement**: discrepancy between DLT triangulations from different minimal view subsets

A 2-layer MLP maps these residuals to per-pair consistency weights `w_{v,w}` and a per-joint gate `g_j`. At initialization the weights are uniform (`w = 1/V`, `g = 1`), so the module is identity-at-init and preserves the v50/v48/v46 full-view baseline. During training the weighted residuals enter a Huber-smoothed auxiliary loss that down-weights view pairs with large geometric disagreement, effectively telling the model: *trust the geometry, not the average*.

The module reuses the per-view reliability produced by the v50 Self-Evolution Feedback Head: a view with low reliability automatically receives a smaller consistency weight, preventing noisy or outlier views from distorting the constraint.

## 2. New config flags with defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_view_consistency_constraint` | bool | `False` |
| `v51_vcc_hidden` | int | `64` |
| `v51_vcc_num_layers` | int | `2` |
| `v51_vcc_reproj_weight` | float | `1.0` |
| `v51_vcc_epipolar_weight` | float | `1.0` |
| `v51_vcc_triang_agreement_weight` | float | `0.5` |
| `v51_vcc_loss_weight` | float | `0.01` |
| `v51_vcc_min_views_for_constraint` | int | `3` |
| `v51_vcc_identity_init` | bool | `True` |
| `v51_vcc_consistency_temperature` | float | `1.0` |
| `v51_vcc_huber_delta` | float | `0.1` |

## 3. Loss term

```text
L_vcc = v51_vcc_loss_weight * Σ_{j,v,w} w_{v,w} · g_j · Huber_δ(
    α_reproj   · r_reproj^{(v,w,j)}
  + α_epipolar · r_epipolar^{(v,w,j)}
  + α_triag    · r_triag^{(v,w,j)}
)
```

An optional regularizer `γ · Var(w)` prevents collapse to uniform weights and encourages the head to discriminate consistent from inconsistent view pairs. The total loss is added to the existing supervised pose loss.

## 4. Evaluation metric

- `MPJPE@k` for `k = 2, 3, 4, full`
- Mean per-view reprojection error after the pose is refined
- Spearman(consistency_weight, true_residual) target `> 0.35`
- Per-pair epipolar residual distribution

## 5. Expected MPJPE impact

- `MPJPE@2`: −2 to −4 mm (sparse subsets benefit most from explicit geometric agreement)
- `MPJPE@3`: −1 to −2 mm
- `MPJPE@full`: ±0.5 mm (baseline preserved by identity init)
- Cross-domain 3DPW actual-mode `MPJPE@2`: up to −5 mm, because noisy in-the-wild keypoints are naturally down-weighted by the consistency gate

## 6. Main risk

**Degenerate over-constraint.** If the loss weight or learned weights become too aggressive, the constraint may pull the 3-D pose toward a geometrically self-consistent but incorrect configuration, especially when camera calibration is slightly inaccurate.

**Mitigation:** keep `v51_vcc_loss_weight` low (default `0.01`), clamp weights to `[0.05, 1.0]`, initialize near identity, freeze base weights for the first epoch, and require at least `v51_vcc_min_views_for_constraint` before activating the triangulation-agreement term.

## 7. Paper-story fit

v51 extends the self-evolution narrative from "the model critiques its own views" (v50) to "the model enforces geometric consensus across views." It directly targets the sparse-view frontier: when only two or three cameras are available, the usual fusion average is brittle, but a learned geometric consistency constraint can still recover a coherent 3-D pose. In the cross-domain setting it treats out-of-distribution 2-D detections as views that violate epipolar agreement and therefore deserve lower weight, closing another self-evolution loop.
