# Swarm Iteration 20 — Novelty Gate and CPU-Only P1

**Date:** 2026-08-07  
**Status:** broad claim STOP; frozen/oracle P0a diagnostic PASS; incremental
utility INCONCLUSIVE. No GPU experiment.

## Novelty gate

**STOP** the broad claim that novelty comes from combining differentiable
triangulation, uncertainty, camera correction, variable views, visibility, and
epipolar attention. Those ingredients already have close prior art:

- [Learnable Triangulation (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Iskakov_Learnable_Triangulation_of_Human_Pose_ICCV_2019_paper.html)
  learns confidence-weighted differentiable triangulation.
- [MetaPose (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Usman_MetaPose_Fast_3D_Pose_From_Multiple_Views_Without_3D_Supervision_CVPR_2022_paper.html)
  jointly reasons about unknown cameras and joint uncertainty with a learned
  bundle-adjustment-like optimizer.
- [Probabilistic Triangulation (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Probabilistic_Triangulation_for_Uncalibrated_Multi-View_3D_Human_Pose_Estimation_ICCV_2023_paper.html)
  models and updates a camera-pose distribution.
- [LOSTU (2023)](https://arxiv.org/abs/2311.11171) propagates observation and
  camera-parameter uncertainty through triangulation.
- [UPose3D (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2418_ECCV_2024_paper.php)
  combines cross-view/temporal cues, uncertainty, and a camera-count-scalable
  fusion strategy.
- [HumanBA (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_HumanBA_Human-Aware_Bundle_Adjustment_via_Global_Human-Camera_Decoupling_CVPR_2026_paper.html)
  is adjacent evidence that human and camera motion should be explicitly
  decoupled rather than folded into one residual.

These papers do not prove that the hypothesis below is new. They do make a
generic "all-in-one geometry fusion" claim indefensible without a narrower
mechanism and a direct falsification.

## P1 hypothesis: Calibration-Tangent Marginalized Fusion

Scope: calibrated multi-view human pose with small rig drift.

One camera's calibration drift produces a low-dimensional residual pattern
shared by all visible joints and frames from that camera. Local occlusion or a
bad keypoint does not share that structure. The hypothesis is to separate the
shared tangent before computing a local visibility or uncertainty score.

For camera `v`, stack reprojection residuals over time and joints into `r_v`.
Let `B_v` be their Jacobian with respect to a small camera perturbation
`delta_theta_v`, `W_v` the 2D observation precision, and `Lambda_v` the camera
prior precision. The following is a MAP/profile estimate, not a marginal
likelihood:

`delta_theta_hat = (B_v^T W_v B_v + Lambda_v)^-1 B_v^T W_v r_v`

`e_v = r_v - B_v delta_theta_hat`

With a Gaussian camera prior, true marginalization instead uses

`C = W^-1 + B Lambda^-1 B^T`

`Q = C^-1 = W - W B (B^T W B + Lambda)^-1 B^T W`

and includes both `r^T Q r` and `log|C|`. The residual `e_v` is strictly
orthogonal only for a flat prior and a full-rank tangent. For local anomaly
scoring, the point being scored must not participate in fitting its camera
offset; P0a therefore uses cross-fitted predictive residuals.

## Exact reductions

1. `Lambda -> infinity`: fixed cameras inside pixel reprojection WLS/GN. This
   is not ordinary DLT, which minimizes an algebraic error.
2. Principal-point-only tangent: `B_i = I_2`; under the Gaussian model the
   MAP/profile estimate becomes a precision-weighted ridge mean. P0a adds
   Huber IRLS as a separate robustification; that step is not exact Gaussian
   marginalization.
3. `T*J=1`: no independent shared evidence; local anomaly scoring must
   abstain rather than fit and score the same point.
4. Give every `(t, j)` its own `delta_theta`: removes cross-joint/time sharing
   and becomes pointwise camera-uncertainty fitting.
5. Keep only `delta_theta_hat` and discard predictive residual/covariance: a
   conventional calibration-correction head.
6. Delete inactive-view rows rather than zeroing fixed slots: true active-set
   inference. Fewer than two usable views cannot triangulate.

The repository's scalar determinant-weighted DLT remains a separate baseline;
it is not an exact reduction of the reprojection-residual marginal model.

## CPU-only P0a: frozen/oracle RED gate

P0a deliberately fixes the 3D pose and tests only `delta_pp=(cx, cy)`. It is a
necessary implementation gate, not evidence for the full CTMF hypothesis.

- `N=T*J in {1, 17, 153}`, four camera groups, 256 deterministic clips.
- 1 px detector noise and one camera with shared `delta_pp=(5,-5) px`.
- `20%` local outliers whose realized signal RMS exactly matches the shared
  drift signal.
- Two-fold predictive scoring; each held-out point is scored by an offset fit
  on the other fold, with three fixed Huber IRLS steps.
- Negative controls: independently shuffle camera grouping for every point,
  and replace the shared offset with per-point independent offsets. A single
  global camera permutation is only a rename and is not a valid control.

P0a uses the fixed numerical reference `5.9915`, equal to the 95th percentile
of chi-square(2). Huber cross-fit scores are not asserted to be strictly
chi-square calibrated; the gate uses their measured FPR/TPR. Both raw and
predictive scores are variance-normalised. Its gates at `N=153` are:

- shared residual ratio `<=0.30`;
- matched-local residual ratio `>=0.80`;
- mixed predictive AUC and true-positive rate `>=0.80`;
- shared-drift and mixed non-outlier predictive flag rates `<=0.10`, while the
  raw shared-drift flag rate is `>=0.80` and mixed false-positive rate drops by
  at least `0.50`;
- shuffled grouping and independent-drift ratios both `>=0.80`;
- `N=1` abstains.

## Decision

**P0a diagnostic PASS.** Shared offset removal, fixed-threshold local gating,
abstention, and both negative controls behave as intended. The earlier
`AUC gain >=0.10` gate was invalid because raw AUC `0.9869` left a maximum
possible gain of only `0.0131`; it is retained only as a descriptive ceiling
effect. See `docs/swarm_iter20/03_ctmf_pp_oracle_p0a.md`.

Incremental utility remains **INCONCLUSIVE**. This result is a favorable
PP-only robust-centering diagnostic; it does not test MPJPE, unknown 3D,
rotation/focal drift, camera update, or a marginal solver. It therefore does
not overturn the broad novelty STOP or justify GPU work. Do not expand it with
graph, top-k, entropy, or a larger transformer.
