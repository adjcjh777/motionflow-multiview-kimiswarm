# Outlier Rejection for Ray-Aware Multi-View Fusion

## Topic

Outlier rejection methods (RANSAC, M-estimators, learned inlier scores) for
calibrated multi-view human pose triangulation.

## Survey

The current `RayAttentionFusionModel` already learns per-view weights through a
self-attention head and feeds them into a differentiable weighted DLT layer.
This implicitly rejects outliers—corrupted 2D detections, occluded views, or
noisy keypoints—by driving low weights toward zero. However, the model is
trained with plain L2 loss on 3D coordinates, which is not robust to gross
outliers and can overfit to the training noise floor.

In calibrated multi-view geometry, three families of outlier handling are well
understood:

1. **Robust statistics / M-estimators.** Replace the quadratic loss with a
   robust kernel (Huber, Tukey, Geman–McClure) or use iteratively reweighted
   least-squares (IRLS) inside the DLT solver. M-estimators bound the influence
   of large residuals and are trivial to add to the 3D loss or to a
   reprojection loss.
2. **RANSAC and consensus-based methods.** Sample minimal view subsets, fit a
   3D point, and keep the solution supported by the largest inlier set. Classic
   RANSAC is non-differentiable, but variants such as differentiable RANSAC
   (DSAC/DSC) or soft-inlier scoring can be wrapped as a plugin or used during
   inference to produce pseudo-ground-truth for training.
3. **Learned inlier scores.** A separate network head predicts an inlier
   probability for each `(joint, view)` pair. The score can be learned from
   explicit labels (outlier injected), from reprojection consistency, or as an
   auxiliary task jointly with the 3D error. This is the natural next step from
   the existing per-view attention weight.

Across these, the strongest practical signal is often **reprojection
consistency**: an inlier view re-projects close to its 2D detection when the
3D point is triangulated from the remaining views. This is cheap to compute,
differentiable, and camera-calibration aware.

## Actionable Recommendations

1. **Add an M-estimator 3D loss and reprojection loss in the trainer.**
   In `experiments/train_ray_attention_real.py`, replace or augment the MSE
   loss with a Huber loss on 3D coordinates. Add an optional reprojection
   loss: project the predicted 3D point back onto each view, weight by the
   predicted per-view weight, and use Huber loss on pixel error. This makes
   the model tolerant to noisy 3D GT and bad 2D observations.

2. **Introduce a learned inlier-score head.**
   Extend `RayAttentionFusionModel` with a small MLP head that predicts
   `inlier_logits` of shape `(B, V, J)` in addition to the existing weight head.
   Supervise it with a binary cross-entropy (BCE) loss using synthetic outlier
   labels from `generate_synthetic_multiview_dataset.py`. Multiply the final
   triangulation weight by `sigmoid(inlier_logits) * confidence` so the model
   explicitly separates *inlier probability* from *view importance*.

3. **Implement a differentiable RANSAC baseline plugin.**
   Add `motionflow_mv/fusion/robust_triangulation_ransac.py` that samples
   2-view subsets, triangulates with the existing DLT, scores by reprojection
   inlier count, and returns the consensus 3D point. Expose it as a registered
   plugin and use it to generate robust pseudo-GT when 3D annotations are
   missing. Keep it optional and deterministic for reproducibility.

4. **Use reprojection-consistency hard rejection at inference.**
   In the forward pass, after weighted DLT triangulation, compute per-view
   reprojection residuals. If a view’s residual exceeds a threshold
   (e.g., 10 px or learned from training), set its weight to zero and re-run
   triangulation once. This closes the gap between soft learned weights and
   hard geometric consistency checks.

5. **Systematic robustness evaluation.**
   Extend `experiments/eval_ray_attention_robustness.py` with a calibrated
   outlier sweep: zero-mean Gaussian outliers at 5/20/50 px, 1–3 occluded
   views, and missing detections. Report per-joint breakdown and compare
   `ray_attention` variants against the DLT baseline and the RANSAC plugin.

## Potential Risks

- **Hard thresholding can discard useful ambiguous views.** Occlusion and low
  confidence are not always outliers. Thresholding should be adaptive or learned,
  not hand-tuned for one dataset.
- **M-estimator losses can collapse to low-confidence predictions.** A robust
  loss may encourage the model to down-weight all views when the problem is
  hard; keep a small regularization term that encourages the mean weight to be
  high enough.
- **RANSAC is slow and non-differentiable.** A pure RANSAC plugin is useful for
  pseudo-GT but cannot replace the learned fusion head. Differentiability
  matters for end-to-end fine-tuning on real data.
- **Cross-dataset calibration mismatch.** A rejection rule trained on Shelf may
  not transfer to Campus or Human3.6M unless it is expressed in normalized
  reprojection units (pixels) rather than metric quantities.

## Fit into the Paper Plan

The paper’s core claim is that *geometry-aware attention fusion beats both naive
attention and plain DLT in calibrated multi-view settings*. Outlier rejection
is the natural mechanism that explains why the attention head succeeds: it is
not merely learning a weighted average, but discovering which views are
reliable. Adding explicit robust components gives a clear ablation study
("soft attention only", "hard reprojection rejection", "M-estimator loss",
"RANSAC pseudo-GT") and strengthens the ICRA/CVPR narrative that the proposed
method is robust to real-world detection noise and occlusion.
