# v10 Design Proposal: Robust Covariance-Aware Triangulation Improvements

**Topic:** Robust covariance-aware triangulation improvements  
**Target:** ICRA/CVPR 2027 — MotionFlow-MultiView  
**Date:** 2026-08-07  

---

## 1. Problem in the Current v7/v8/v9 Pipeline

### 1.1 Current pipeline status

| Version | Status | Key configuration |
|---------|--------|-------------------|
| v7 | Stable, step-50 loss ~55 and decreasing | `use_full_precision_dlt=True`, `use_robust_dlt_reweight=False` |
| v8 | Stable after clamping fixes | v7 + `use_robust_dlt_reweight=True` + precision-matrix/weight clamping |
| v9 | Unstable, loss explodes to ~3000 at step 50-100 | v8 + 2D reprojection loss + PA loss + monotonic loss |

### 1.2 Root cause

The core robust-DLT block in `motionflow_mv/fusion/omniview_fusion_v5.py` (lines 491-533) currently:

1. Builds a per-view 2×2 precision matrix from the predicted lower-triangular `L`.
2. Solves once with those precision-weighted DLT equations.
3. Computes per-view reprojection residuals, maps them to a **single Gaussian weight**
   ```python
   rho = torch.exp(-mahal / 2.0).clamp(min=1e-3, max=1.0)
   ```
4. Re-solves once with `weights_robust = weights * rho` (detached).

This single Gaussian-kernel reweighting has three weaknesses that v9 amplified:

- **Gaussian down-weighting is not robust to gross outliers.** A single bad 2D keypoint creates a large Mahalanobis distance; `exp(-mahal/2)` goes to the clamp floor (`1e-3`) almost instantly, but the *gradient* and the residual energy are still dominated by the outlier because the reprojection loss added in v9 is unbounded MSE on the *same* residuals.
- **No iterative refinement.** A single reweighting step cannot recover from a poor initial point estimate, especially early in training when the covariance/precision head is uncalibrated. Iterated reweighted least-squares (IRLS) is the standard remedy.
- **No calibrated robust scale.** The predicted precision matrices are clamped to `[-1e3, 1e3]`, but the model still learns an arbitrary scale. Residuals of a few pixels can map to Mahalanobis values of 0.01 or 100 depending on the predicted covariance, making the robust threshold inconsistent across training steps.

The v9 loss explosion is therefore a symptom: the unbounded 2D reprojection loss feeds back through unstable, single-pass robust weights, and the precision head over-shrinks variances to make the Gaussian penalty vanish.

---

## 2. Concrete v10 Change: Iterative Robust M-Estimator Triangulation (IRLT)

### 2.1 Goal

Replace the one-shot Gaussian reweight in v8 with a **fixed 2–3 step IRLS loop** using a **bounded robust kernel** and an **auto-scaled residual threshold**, while keeping the rest of the v8/v9 architecture untouched.

### 2.2 Where to change

Primary file: `motionflow_mv/fusion/omniview_fusion_v5.py`  
Secondary: `motionflow_mv/fusion/triangulation.py` (optional helper)  
Loss script: `experiments/train_omniview_fusion_v5_webbridge_multi.py`

### 2.3 Proposed implementation

Inside the existing `if self.use_full_precision_dlt:` block, replace the single robust reweight pass with:

```python
# --- v10: iterative robust M-estimator DLT -----------------------------
# 1. Initial precision-weighted solve (already present).
weights = weights * confidences * visibility
weights = weights.clamp(min=1e-4, max=1e4)
eye2 = torch.eye(2, device=L.device, dtype=L.dtype).view(1, 1, 1, 2, 2)
cov = L @ L.transpose(-2, -1) + 1e-3 * eye2
try:
    precision_matrix = torch.linalg.inv(cov)
except RuntimeError:
    precision_matrix = torch.linalg.inv(cov + 1e-2 * eye2)
precision_matrix = torch.where(
    torch.isnan(precision_matrix) | torch.isinf(precision_matrix),
    eye2.expand_as(precision_matrix),
    precision_matrix,
)
precision_matrix = precision_matrix.clamp(min=-1e3, max=1e3)
Rt = torch.cat([R, t[..., None]], dim=-1)
P = K_corrected @ Rt
pred_3d_raw = triangulate_dlt_batched_lstsq(
    points_2d, P, weights, precision_matrix=precision_matrix
)

if self.use_robust_dlt_reweight:
    # 2. IRLS with Cauchy/Huber-like kernel.
    pred_3d_h = torch.cat(
        [pred_3d_raw, torch.ones(pred_3d_raw.shape[0], pred_3d_raw.shape[1], 1,
                                 device=pred_3d_raw.device, dtype=pred_3d_raw.dtype)],
        dim=-1,
    )  # (N, J, 4)
    x_h = (P.unsqueeze(2) @ pred_3d_h.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
    x_pred = x_h[..., :2] / (x_h[..., 2:3] + 1e-8)  # (N, V, J, 2)
    residual = x_pred - points_2d  # (N, V, J, 2)

    # Auto-scale: median absolute deviation over the valid views for each joint.
    r_sq = residual.pow(2).sum(dim=-1)  # (N, V, J)
    median_r = r_sq.median(dim=1, keepdim=True)[0]  # (N, 1, J)
    scale = (median_r + 1e-6).clamp(min=1e-4)  # (N, 1, J)

    for _ in range(self.robust_dlt_iterations):  # default 2
        residual_col = residual.unsqueeze(-1)  # (N, V, J, 2, 1)
        mahal = residual_col.transpose(-2, -1) @ precision_matrix @ residual_col
        mahal = mahal.squeeze(-1).squeeze(-1)  # (N, V, J)

        # Cauchy-like robust weight with bounded influence.
        # Clamp the normalized distance so early bad covariances don't dominate.
        q = (mahal / scale).clamp(max=25.0)
        rho = 1.0 / (1.0 + q)  # (N, V, J)

        # Weight floor: never zero-out a view completely.
        rho = rho.clamp(min=0.1, max=1.0)

        # Stop-gradient between IRLS steps to keep gradients stable.
        weights_robust = (weights * rho * view_mask_flat.unsqueeze(-1)).detach()
        weights_robust = weights_robust.clamp(min=1e-4, max=1e4)

        pred_3d_raw = triangulate_dlt_batched_lstsq(
            points_2d, P, weights_robust,
            precision_matrix=precision_matrix.detach()
        )

        # Update residual using the new point estimate.
        pred_3d_h = torch.cat(
            [pred_3d_raw, torch.ones_like(pred_3d_raw[..., :1])], dim=-1
        )
        x_h = (P.unsqueeze(2) @ pred_3d_h.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
        x_pred = x_h[..., :2] / (x_h[..., 2:3] + 1e-8)
        residual = x_pred - points_2d
# ----------------------------------------------------------------------
```

### 2.4 Key design choices

| Choice | Rationale |
|--------|-----------|
| **Cauchy kernel `1/(1+q)`** | Bounded influence: outliers cannot dominate the loss, unlike the Gaussian `exp(-q/2)` whose tail never fully dies. |
| **MAD auto-scale** | Removes the need to hand-tune a pixel threshold; the robust threshold adapts to the residual distribution of each joint/batch. |
| **Weight floor `min=0.1`** | Prevents the model from learning to collapse a view to near-zero weight, which is one cause of view-collapse in variable-view training. |
| **2 iterations, stop-gradient** | Standard IRLS practice. Detaching between iterations avoids backpropagating through the inverse precision and the re-solve. The final solve still receives gradients through `weights_robust` on the last pass. |
| **Clamp `q ≤ 25`** | Keeps the robust kernel numerically stable even when the covariance head predicts degenerate precision. |

### 2.5 New flag

Add `robust_dlt_iterations: int = 2` to `OmniMultiViewFusionV5.__init__`.  
Add CLI arg `--robust_dlt_iterations` in `experiments/train_omniview_fusion_v5_webbridge_multi.py`.

---

## 3. Validation Plan

### 3.1 Smoke test (minutes)

```bash
python motionflow_mv/fusion/omniview_fusion_v5.py
```

Already exercises the forward pass. After the change, ensure:
- No `NaN`/`Inf` in outputs.
- Gradients still flow through the new IRLS block.
- Variable-view mask still zeroes out masked views.

### 3.2 Small fast run (hours)

Run a 1–2 epoch experiment on the WebBridge H36M smoke split with the v8 config plus the new IRLS:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --use_full_precision_dlt \
  --use_robust_dlt_reweight \
  --robust_dlt_iterations 2 \
  --reproj_loss_weight 0.0 \
  --pa_loss_weight 0.0 \
  --monotonic_loss_weight 0.0 \
  --output outputs/v10_robust_irlt_smoke.pth
```

**Pass criteria:**
- Loss decreases and stays below 100 by step 200.
- No gradient `NaN`/explosion.
- Val MPJPE comparable to or better than the v8 smoke run.

### 3.3 Full A800 run (days)

Once smoke/small runs pass, run the full H36M+MPI mixed-dataset variable-view config:

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed.yaml \
  --use_mixed_loader \
  --use_full_precision_dlt \
  --use_robust_dlt_reweight \
  --robust_dlt_iterations 2 \
  --use_variable_view_training \
  --variable_view_min_views 2 \
  --variable_view_max_views 14 \
  --use_domain_embedding \
  --reproj_loss_weight 0.0 \
  --pa_loss_weight 0.0 \
  --monotonic_loss_weight 0.0 \
  --output outputs/v10_robust_irlt.pth
```

Compare against the v8 checkpoint as the anchor. If stable, re-enable the v9 losses one at a time starting with the smallest-weight reprojection loss.

---

## 4. Expected Impact

| Metric | Expected change | Mechanism |
|--------|-----------------|-----------|
| **Training stability** | Large improvement | Cauchy kernel + MAD scale stop the unbounded MSE feedback that blew up v9. |
| **MPJPE** | -5% to -10% vs v8 | Better outlier suppression and iterative refinement give cleaner 3D estimates. |
| **Variable-view robustness** | Moderate gain | Weight floor prevents view collapse on small view subsets. |
| **Calibration noise** | Slight gain | MAD scaling adapts to larger residuals from perturbed cameras. |
| **Throughput** | -5% to -10% | Two extra `lstsq` solves per forward pass; still cheap relative to the ST transformer. |

---

## 5. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| **MAD scale is noisy for small view counts (V=2)** | Medium | Fallback: if `V < 4`, replace median with a fixed percentile (e.g., 75th) or a learned scale parameter. |
| **Weight floor `0.1` is too high and under-downweights bad views** | Medium | Sweep floor in `{0.05, 0.1, 0.2}` on the smoke run; keep the value that gives the lowest val MPJPE. |
| **Cauchy kernel is too aggressive early in training** | Low | The `q ≤ 25` clamp and weight floor bound the worst-case behavior; also start with `robust_dlt_iterations=1` for the first epoch if needed. |
| **Extra `lstsq` calls hurt throughput** | Low | Profile the forward pass; if the triangulation step becomes >15% of total time, reduce to `robust_dlt_iterations=1` or fuse the residual projection. |
| **Re-enabling v9 losses still explodes** | Medium | Re-introduce them one at a time with small weights (`reproj_loss_weight=0.001`) and gradient clipping; do not use the v9 full-weight config as the first test. |

---

## 6. Fallback Plan

If the IRLS change does not improve stability or MPJPE:

1. Revert to the v8 single-pass robust reweight.
2. Add a **learned scalar scale** to the existing Gaussian kernel: `rho = exp(-mahal / (2 * scale))`, where `scale` is predicted from the pooled features. This keeps the change minimal and directly addresses the uncalibrated covariance problem.
3. Keep `reproj_loss_weight=0.0` and revisit the reprojection loss only after the precision head has been trained longer with the NLL/uncertainty loss.

---

## 7. Summary

**v10 direction:** Stabilize the v8 robust covariance-aware triangulation by switching from a single Gaussian reweight to an **iterative Cauchy-kernel IRLS** with **MAD auto-scaling** and a **weight floor**. This is a minimal, focused change that directly addresses the instability that prevented v9's auxiliary losses from working. It does not require new model components, new loss terms, or new data loaders.
