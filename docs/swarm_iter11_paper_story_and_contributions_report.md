# Iter11+ Paper Story and Contributions — MotionFlow-MultiView

## 1. Current state

The codebase now contains a single “all-in-one” fusion model,
`RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1`
(`motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`).
It stacks per-frame ray-aware view/joint attention, spatio-temporal
(time × view) cross-view attention, per-view log-variance prediction,
differentiable weighted DLT, Gauss-Newton (GN) triangulation refinement, and a
residual MLP head.

The current best published number is **11.17 mm MPJPE** on MPI-INF-3DHP S2/Seq1
from the smaller `RayAttentionFusionModelTemporalResidual` (243 k params). The
new combined model has not yet been trained to convergence on the full
MPI-INF-3DHP corpus; a fast temporal-residual + reprojection run reportedly
reached only ~47 mm because of limited data/epochs. The *architectural story* is
therefore ahead of the *empirical story*.

## 2. Recommended paper story

Frame the submission around a single, defensible decomposition:

> **Geometry first, learning second.** A lightweight network predicts which
> views/joints are trustworthy, triangulates a metric 3D pose with explicit
> camera geometry, and then a learned residual corrects the structured leftover
> error using spatio-temporal context and uncertainty.

This yields four clear contributions:

1. **A unified calibrated multi-view fusion plugin.** DLT, weighted DLT,
   Gauss-Newton refinement, and residual correction form one differentiable
   module inside the MotionFlow `HumanMotionIR` pipeline.
2. **Uncertainty-weighted differentiable triangulation.** Predicted log-variance
   down-weights noisy/occluded views before DLT and informs the residual head.
3. **Spatio-temporal cross-view attention with residual refinement.** The
   `(time, view)` grid lets the model borrow evidence across views *and* frames,
   while the residual head learns the small correction left after triangulation.
4. **Systematic cross-dataset evaluation.** MPI-INF-3DHP, Human3.6M, and
   Shelf/Campus under clean, noisy, dropout, and outlier conditions.

## 3. Concrete, implementable improvements

### 3.1 Make the residual head uncertainty- and geometry-aware

The combined model’s residual MLP currently receives only the pooled feature and
the GN-refined 3D pose. It should also receive a per-joint uncertainty summary
and the reprojection residual magnitude from the GN step, so the residual can
adapt to low-confidence joints.

```python
# In RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1
log_var_summary = log_var.mean(dim=1)                      # (N*J, J)
reproj_resid = _reprojection_residual(points_2d, pred_3d_gn, P)

residual_input = torch.cat([
    feat_pooled,            # (N*J, J, d)
    pred_3d_gn,             # (N*J, J, 3)
    log_var_summary.unsqueeze(-1),
    reproj_resid.unsqueeze(-1),
], dim=-1)

delta = self.residual_mlp(residual_input)
pred_3d = pred_3d_gn + delta
```

This is a one-file change and should improve MPJPE because it prevents the
residual head from blindly trusting joints flagged as unreliable by the
uncertainty head.

### 3.2 Add bone-length and temporal-smoothness losses to the combined trainer

`experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py`
uses only 3D MSE + optional reprojection + NLL. Add the existing
`train_utils.py` losses:

```python
from experiments.train_utils import bone_length_loss, skeleton_consistency_loss

loss = criterion(pred, yb) + nll_loss
if args.reproj_weight > 0.0:
    loss += args.reproj_weight * reprojection_loss(...)
if args.bone_weight > 0.0:
    loss += args.bone_weight * bone_length_loss(pred, yb, parents=H36M_17_PARENTS)
if args.smooth_weight > 0.0:
    loss += args.smooth_weight * skeleton_consistency_loss(pred, parents=H36M_17_PARENTS)
```

Bone-length and symmetry priors help most under heavy occlusion, where the model
must fall back on skeleton-level reasoning.

### 3.3 Run a full ablation ladder on the same data split

Train the following variants on the same MPI-INF-3DHP split (S1 Seq1+Seq2 train,
S2 Seq1 val) and report identical metrics: base weighted DLT, + temporal,
+ cross-view, + uncertainty, + GN refinement, + residual, + bone/smooth losses.
Use the same seed, clip length (13), and augmentation. The combined model only
earns its complexity if it beats the 11.17 mm baseline or matches it with fewer
parameters.

### 3.4 Standardize the cross-dataset evaluation protocol

* **MPI-INF-3DHP:** use the canonical train split and official validation/test
  set. Report MPJPE, PA-MPJPE, PCK@50/100/150 mm, and AUC.
* **Human3.6M:** finish the WebBridge batch conversion (`data/webbridge/h36m`)
  and evaluate on the canonical test subjects.
* **Robustness:** on both datasets report clean, 5 px / 20 px Gaussian noise,
  50 % joint dropout, and 5 % / 20 % 2D outlier conditions. Re-use
  `experiments/eval_residual_robustness_mpiinf3dhp_v1.py`.
* **Efficiency:** report parameters, FLOPs, and RTX 4090 latency/throughput via
  `experiments/benchmark_inference_v3.py`.

## 4. Metrics to track

* **Accuracy:** MPJPE, PA-MPJPE, PCK@50/100/150 mm, AUC.
* **Per-joint:** locate which body parts benefit from each component.
* **Robustness:** relative error increase under noise/dropout/outliers.
* **Uncertainty calibration:** NLL and correlation between predicted log-variance
  and actual reprojection error.
* **Efficiency:** parameter count, latency, throughput, and memory.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Combined model underperforms smaller 11.17 mm model | Run the full ablation ladder; keep the simpler model as the main result if necessary |
| Gauss-Newton refinement is unstable or slow | Start with `gn_iters=1`; add damping schedule; profile with `benchmark_inference_v3.py` |
| Uncertainty head collapses | Clamp log-variance; initialise head to small weights; monitor NLL |
| Cross-view attention overfits (previously 15.29 mm) | Use strong augmentation, bone-length regularisation, and early stopping |
| H36M WebBridge conversion incomplete | Prioritise finishing it; otherwise report MPI-INF-3DHP as primary benchmark |

## 6. Immediate next steps

1. Feed log-variance summary and reprojection residual into the residual MLP in
   `ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`.
2. Add `--bone_weight` and `--smooth_weight` to the combined trainer.
3. Launch the ablation ladder on the full MPI-INF-3DHP training corpus.
4. Finish H36M WebBridge conversion and run the same evaluation there.
5. Update `docs/paper_draft_icra_cvpr_2027.md` with the new ablation table and
   cross-dataset benchmark table.

The Iter11+ goal is to show that the combined model produces a **measurable,
reproducible improvement** over the 11.17 mm baseline while telling a coherent
story: *triangulate with learned uncertainty, then refine with spatio-temporal
context.*
