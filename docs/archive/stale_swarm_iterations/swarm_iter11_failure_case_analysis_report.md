# Iter11+ Failure-Case Analysis Roadmap for MotionFlow-MultiView

**Date:** 2026-08-04  
**Scope:** Turn failure analysis from a post-hoc summary into a model-design driver for the ICRA/CVPR 2027 submission.  
**Context:** The latest combined model (`RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1`) stacks cross-view temporal attention, uncertainty-weighted DLT, differentiable Gauss-Newton (GN) refinement, and a residual MLP. The current best MPI-INF-3DHP MPJPE is ~11.17 mm (temporal-cross-view-residual), while a fast combined run reached 47.54 mm due to limited training. This report proposes concrete failure-analysis tooling to close that gap.

## 1. Current State and Gaps

### What exists
- `experiments/analyze_failures.py` clusters H36M high-error frames into `occlusion`, `outlier`, `scale`, and `orientation` using `RayAttentionFusionModelV3` or DLT.
- `experiments/analyze_failures_temporal_mpiinf3dhp.py` produces per-joint/per-frame/per-view tables and plots for the temporal-only model.
- `motionflow_mv/eval/metrics.py` provides MPJPE, PA-MPJPE, PCK, and AUC.

### Critical gaps
- **No analyzer for the combined model.** Existing scripts target older architectures; the new model’s DLT, GN, and residual stages are opaque.
- **Uncertainty is unvalidated.** The model predicts per-view log-variance, but no script checks whether high uncertainty correlates with high 3D error or reprojection residual.
- **GN step is unmeasured.** We do not know which failure modes GN fixes or hurts.
- **No controlled failure dataset.** Synthetic corruption is not systematically injected.
- **Temporal boundary effects are uncharacterized.** Clip edges and stride choices are not analyzed for the new model.

## 2. Proposed Improvements

### Improvement 1: Per-Stage Failure Dissection
Create `experiments/analyze_failures_advanced.py` that loads the combined model and returns four predictions per sample:
1. `pred_dlt` — weighted DLT output.
2. `pred_gn` — after Gauss-Newton refinement.
3. `pred_res` — after the residual MLP.
4. `pred_final` — final output with `n_iter` residual iterations.

Compute MPJPE/PA-MPJPE for each and break down by the four failure-mode clusters. This answers: *Which component fixes which failure mode?*

### Improvement 2: Uncertainty Calibration Analysis
For every `(frame, view, joint)`, bin samples by predicted `log_var` and compute actual 3D error and reprojection residual. Report:
- **Expected Calibration Error (ECE)** for uncertainty vs. reprojection error.
- **Spearman ρ** between predicted precision `exp(-log_var)` and actual error.
- **Per-joint calibration curves.**

A calibrated uncertainty head should down-weight true outliers.

### Improvement 3: Controlled Synthetic Failure Suite
Extend the synthetic generator in `tests/test_pipeline_synthetic.py` to produce labeled failure modes:
- **Occlusion:** zero out a random subset of views per joint (`confidence = 0`).
- **2D outliers:** add large Gaussian jitter to one view-joint observation.
- **View dropout:** randomly drop an entire camera for a frame.
- **Depth-ambiguous rig:** synthesize cameras with shallow ray intersection angles.
- **Motion blur:** add correlated temporal noise to 2D keypoints.

Run the analyzer on this suite to produce per-mode gain tables (DLT → GN → residual).

### Improvement 4: Temporal Error Propagation
Evaluate with `clip_len ∈ {9, 13, 27}` and `stride ∈ {1, clip_len/2, clip_len}`. Compare clip-center vs. clip-edge MPJPE to quantify temporal smoothing and compute trade-offs.

### Improvement 5: Lightweight Regression Tests
Add `tests/test_failure_modes.py` asserting that on synthetic outlier injection the model beats DLT, predicted `log_var` rises with injected noise, and GN reduces reprojection error vs. DLT.

## 3. Experiments to Run

| Experiment | Data | Expected Output |
|------------|------|-----------------|
| Per-stage failure analysis | H36M WebBridge (`data/webbridge/h36m`) and MPI-INF-3DHP S2/Seq1 | `docs/swarm_iter11/failure_dissection.md` with DLT/GN/residual breakdown per cluster |
| Uncertainty calibration | Same as above | ECE, rank correlation, calibration plots |
| Synthetic failure suite | Synthetic generator in `tests/` | `docs/swarm_iter11/synthetic_robustness_table.md` |
| Temporal propagation | MPI-INF-3DHP S2/Seq1, varying clip_len/stride | `docs/swarm_iter11/temporal_boundary_error.md` and plot |
| Ablation under failures | Latest combined model vs. DLT vs. temporal-only | Per-mode MPJPE delta table for the paper |

## 4. Metrics to Track

- **MPJPE / PA-MPJPE** overall and per failure cluster.
- **Per-stage MPJPE:** DLT, GN, residual, final.
- **Stage deltas:** `ΔGN` and `ΔRes` per failure mode.
- **Uncertainty ECE** and **Spearman ρ** between predicted precision and actual reprojection error.
- **Per-view reprojection error (px)** and **per-joint error (mm)** for worst frames.
- **Temporal edge error ratio:** clip-edge / clip-center MPJPE.
- **Inference latency** (ms/frame).

## 5. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Combined model is undertrained; analysis may be premature. | Build the analyzer to be reusable as training converges; run on the best checkpoint available. |
| GN and residual stages interact; per-stage comparison may be confounded. | Use the same checkpoint and report cumulative deltas; freeze earlier stages when possible. |
| Synthetic failure modes may not transfer to real data. | Validate top findings on real H36M/MPI-INF-3DHP subsets selected by the same indicators. |
| Heuristic failure labels may mislabel. | Report continuous indicator-vs-error curves alongside hard clusters. |
| Longer temporal windows increase compute and memory. | Bound sweep to `clip_len ≤ 27` and benchmark latency alongside accuracy. |

## 6. Proposed Code Skeleton

```python
def forward_stages(model, x, points_2d, confidences, K, R, t):
    """Return DLT, GN, residual, and final predictions."""
    feat = model._extract_frame_features(x, K, R, t)
    log_var = model.uncertainty_head(feat.permute(0, 2, 1, 3)).squeeze(-1)
    log_var = torch.clamp(log_var, model.log_var_min, model.log_var_max)
    log_var = log_var.permute(0, 2, 1)

    weights = (torch.exp(-log_var) * confidences).clamp(min=1e-4)
    Rt = torch.cat([R, t[..., None]], dim=-1)
    P = K @ Rt

    pred_dlt = _triangulate_weighted_dlt(points_2d, weights, P)
    pred_gn = _triangulate_weighted_gauss_newton(
        points_2d, weights, K, R, t, pred_dlt,
        num_iters=model.gn_iters, damping=model.gn_damping,
    )
    feat_pooled = feat.mean(dim=1)
    pred_res = pred_gn + model.residual_mlp(
        torch.cat([feat_pooled, pred_gn], dim=-1)
    )
    return pred_dlt, pred_gn, pred_res, pred_res  # final with n_iter=1


def analyze_advanced(model, loader):
    stages = {"DLT": [], "GN": [], "Residual": [], "Final": []}
    all_gt, all_indicators = [], []
    for x, gt, K, R, t in loader:
        dlt, gn, res, final = forward_stages(model, x, x[..., :2], x[..., 2], K, R, t)
        for k, v in zip(stages.keys(), [dlt, gn, res, final]):
            stages[k].append(v.detach().cpu().numpy())
        all_indicators.append(compute_failure_indicators(final, gt, ...))
    gt = np.concatenate(all_gt)
    labels, _ = classify_failure_mode(merge_indicators(all_indicators))
    for name, preds in stages.items():
        preds = np.concatenate(preds)
        print(f"{name}: {mpjpe(preds, gt):.2f} mm")
        for mode in ["occlusion", "outlier", "scale", "orientation"]:
            if (mask := labels == mode).any():
                print(f"  {mode}: {mpjpe(preds[mask], gt[mask]):.2f} mm")
```

## 7. Conclusion and Next Steps

The analyzer will expose whether the GN/residual stages fix the failures limiting MPJPE, whether the uncertainty head is calibrated, and where temporal modeling leaks errors. Next, implement `experiments/analyze_failures_advanced.py` and run it on the latest checkpoint alongside the synthetic failure suite and temporal-boundary sweep.
