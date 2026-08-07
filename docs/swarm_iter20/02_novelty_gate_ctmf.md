# Swarm Iteration 20 — Novelty Gate and CPU-Only P1

**Date:** 2026-08-07  
**Status:** research hypothesis only; not implemented, trained, or validated.

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
bad keypoint does not share that structure. The proposed mechanism first
projects out the shared calibration tangent, then uses only the remaining
residual for visibility and uncertainty gating.

For camera `v`, stack reprojection residuals over time and joints into `r_v`.
Let `B_v` be their Jacobian with respect to a small camera perturbation
`delta_theta_v`, `W_v` the 2D observation precision, and `Lambda_v` the camera
prior precision:

`delta_theta_hat = (B_v^T W_v B_v + Lambda_v)^-1 B_v^T W_v r_v`

`r_v_perp = r_v - B_v delta_theta_hat`

The proposed gate consumes `r_v_perp`, not the raw residual. The claim is not
that Schur complement or bundle adjustment is new. The claim to test is that
explicitly separating a shared camera-drift mode from local joint residuals
prevents recoverable calibration error from being mistaken for occlusion.

## Exact reductions

1. `Lambda -> infinity`: fixed cameras; full-matrix 2D-precision weighted
   triangulation.
2. Also set `W = sigma^-2 I`: ordinary DLT/Gauss-Newton.
3. Set `W_tjv = det(Sigma_tjv)^-1/2 I`: the repository's current scalar
   determinant-weighted DLT. The current DLT path does not consume the full
   anisotropic `2 x 2` precision.
4. Set `T=1`, `J=1` with the same residual/Jacobian/covariance model: a
   pointwise LOSTU-style problem, with no shared evidence to distinguish drift
   from a local outlier.
5. Give every `(t, j)` its own `delta_theta`: removes the shared-camera
   mechanism and reduces to pointwise camera-uncertainty marginalization.
6. Keep only `delta_theta_hat` and discard `r_v_perp` and its posterior: a
   conventional calibration-correction head.
7. Delete inactive-view rows rather than zeroing fixed slots: true active-set
   inference. Fewer than two usable views is non-identifiable and should
   abstain.

## CPU-only P0

Use the existing synthetic camera generator; no learning and no GPU.

- `J=17`, `T in {1, 9}`, active views `k in {2, 3, 4, 6}`.
- Corruptions:
  - 1 px independent detector noise;
  - per-camera shared drift across all `(T, J)`: rotation `0.5 deg`, focal
    `1%`, principal point `5 px`;
  - `20%` local joint occlusion/outliers;
  - shared drift plus local outliers.
- Match raw reprojection RMS between the shared-drift and local-outlier cases.
- Compare ordinary DLT, current scalar weighting, full-precision weighted DLT,
  a pointwise LOSTU-style solver, and the proposed marginalized solver.

Diagnostics:

- `||r_perp||_W / ||r||_W` for shared drift versus local outliers;
- outlier AUC from raw residual versus marginalized residual;
- MPJPE and area under the MPJPE-versus-view-count curve;
- scaling from `J*T=1` to `17*9`;
- camera-identity shuffle and per-joint independent drift as negative controls.

## GO / STOP

**GO** only if all hold on the mixed corruption setting:

- at least `10%` MPJPE reduction versus the best pointwise baseline for
  `k=2..4`;
- clean/no-drift regression no worse than `1%`;
- median residual ratio at most `0.3` for shared drift and at least `0.8` for
  local outliers, or outlier AUC at least `0.8`;
- benefit grows with `J*T` and disappears under camera-identity shuffle.

Otherwise **STOP** this mechanism. Do not rescue it by adding graph, top-k,
entropy, or larger transformer modules.
