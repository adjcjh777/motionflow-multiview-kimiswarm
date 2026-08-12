# Differentiable Triangulation Improvements

**Scope:** Iterative triangulation, bundle-adjustment-style refinement, and
weighted least-squares extensions for the ray-aware attention fusion plugin.  
**Target venue:** ICRA/CVPR 2027, as part of the MotionFlow multi-view fusion
contribution.

## 1. Current state and open gap

`RayAttentionFusionModel` triangulates each joint with a single-step weighted
DLT (`motionflow_mv/fusion/ray_attention_model.py`).  A transformer predicts
per-view, per-joint weights; the weights are multiplied by confidences and fed
into a differentiable linear least-squares solve.

This already outperforms the earlier end-to-end attention head (3.7 m vs.
0.0021 m MPJPE on the synthetic GVHMR demo), because the DLT layer encodes the
geometry.  However, four limitations remain:

1. **Single linear solve.** The estimate is never revised using reprojection
   discrepancies.
2. **Ad-hoc weighting.** Weights enter through a square-root hack, not a
   statistically grounded precision model.
3. **No robust M-estimator.** A single outlier view can still bias the linear
   solution before the attention head fully suppresses it.
4. **Per-joint isolation.** Joint-to-joint skeleton constraints are ignored.

## 2. Brief survey

**Iterative reweighted least squares (IRLS).**  After DLT, recompute residuals
and update weights via a robust kernel (Huber, Geman-McClure).  IRLS is the
classical way to make triangulation robust to outliers (Hartley & Zisserman,
2004; Martinez et al., 2017).

**Bundle-adjustment-style refinement.**  Minimize reprojection error with
respect to the 3D point using a few Gauss-Newton or Levenberg-Marquardt steps,
keeping camera parameters fixed.  This is differentiable, cheap, and known to
sharply reduce reprojection error (Kneip & Furgale, 2014; Vakhitov et al.,
2021).

**Weighted least squares with learned precision.**  Replace scalar weights with
learned log-precisions (inverse variances).  The optimal linear system is then
weighted by precision, and the training objective becomes a negative
log-likelihood rather than plain MSE (Iskakov et al., 2019; Remelli et al.,
2020).

**Skeleton-aware triangulation.**  Bone-length and temporal-smoothness priors
can be added as soft constraints during triangulation or as additional losses
during training, helping when joints are occluded.

## 3. Actionable recommendations

1. **Add a differentiable IRLS head.**  After the initial weighted-DLT estimate,
compute per-view reprojection residuals and update weights through a robust
kernel.  Repeat for 2–3 iterations.  This remains fully differentiable and adds
only a small per-joint loop.

2. **Add a differentiable Gauss-Newton refinement layer.**  Run 2–5 GN steps
that minimize weighted reprojection error `Σ sqrt(w_i) ||P_i X − x_i||²` with
respect to `X` only.  This gives a bundle-adjustment flavor without touching
camera parameters.

3. **Use learned precision instead of ad-hoc weights.**  Let the attention head
output per-view log-precisions `Λ_vj = exp(log Λ_vj)`.  Use `Λ_vj` directly in
the normal equations, and add a small negative log-likelihood reprojection
term to the training loss.

4. **Add a skeleton consistency loss.**  During training, penalize bone-length
violations and encourage temporal smoothness.  This is loss-only and does not
affect inference latency, directly addressing occluded-joint failures.

5. **Run a controlled ablation.**  Compare (a) baseline DLT, (b) current weighted
DLT, (c) IRLS, (d) GN refinement, and (e) the combination.  Report MPJPE (mm),
mean reprojection error (px), and a per-joint breakdown.

## 4. Potential risks

- **Gradient stability.** Each iterative step calls `torch.linalg.lstsq` or an
  SVD pseudoinverse; gradients can explode for degenerate views.  Mitigate by
  using an SVD pseudoinverse in the refinement loop, or by detaching the DLT
  initialization and back-propagating only through refinement.

- **Training cost.**  GN steps add per-joint Jacobians.  With 17 joints and 4
  views the cost is modest, but unrolling many iterations can slow training by
  20–40%.  Keep ≤3 IRLS and ≤3 GN steps.

- **Synthetic-to-real mismatch.**  Current validation is mostly synthetic; real
  Shelf/Campus data has different outlier patterns.  Move to real data as soon as
  it is available.

- **Degenerate weight collapse.**  If attention weights become near-zero for all
  views, the DLT solve is singular.  Add a small ridge regularizer (`+ λI`) or a
  minimum-weight clamp.

## 5. Fit into the paper plan

These improvements turn the ray-aware plugin from a “weighted DLT wrapper” into
a **differentiable triangulation engine** and provide the following paper
contributions:

- A quantitative ablation section comparing *classical DLT*, *naive learned
  weights*, *IRLS*, and *GN refinement*.
- Clear metrics: MPJPE on Shelf/Campus and the GVHMR demo, plus
  outlier-robustness curves on the synthetic dataset.
- Alignment with the project’s next step of adding epipolar losses, because the
  GN layer naturally produces reprojection and epipolar residuals for
  additional losses.

**Immediate next step:** implement IRLS + GN layers in a new
`DifferentiableTriangulation` module, keep the existing
`RayAttentionFusionModel` interface, and benchmark on the synthetic generator
before real Shelf data is available.
