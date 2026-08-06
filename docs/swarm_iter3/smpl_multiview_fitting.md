Done. Below is a concise technical report (also ready to be saved as `docs/swarm_iter3/smpl_multiview_fitting.md`). I did not write it to disk because this subagent is read-only.

---

# SMPL Multi-View Fitting for MotionFlow Multi-View

## 1. Problem statement

The current MotionFlow multi-view pipeline takes `V` calibrated, synchronized views, runs a per-view human estimator (GVHMR, ScoreHMR, etc.) to obtain per-view `HumanMotionIR`s, and then fuses per-view 2D keypoints into a world-coordinate 3D skeleton via `DLTFusion`, `RobustTriangulation`, attention, or temporal refinement. The fused skeleton is finally repackaged into a new `HumanMotionIR` by shifting the reference view’s root translation and averaging per-view shape parameters.

This leaves a clear gap: **the pipeline never produces a single, physically consistent SMPL/SMPL-X body that is jointly explained by all camera views.** Instead it produces a set of triangulated 3D joints and only approximately maps them back to SMPL parameters. Consequences include:

- **No cross-view shape consistency.** Each view’s per-view estimator predicts its own `betas`; the fused IR simply averages them (`motionflow_mv/ir/multiview_adapter.py:143-156`).
- **No multi-view reprojection constraint on the body model.** The parametric body is not used as a geometric constraint during fusion, so the resulting pose can violate bone lengths, joint limits, or inter-view consistency.
- **Loss of temporal and physical plausibility.** The fusion plugins regress 3D joint positions frame-by-frame; the final IR keeps the reference view’s `body_pose` and `global_orient` and only adjusts `transl`.
- **Ambiguous metric scale and root location.** World coordinates are inherited from triangulated joints, but the SMPL body itself is not aligned to a shared metric ground plane or gravity direction.

**SMPL multi-view fitting** is the task of estimating a single set of SMPL/SMPL-X parameters—`global_orient`, `body_pose`, `transl`, and `betas`—that, when rendered through each calibrated camera, best reproduces the observed multi-view 2D evidence and remains temporally smooth and physically plausible.

## 2. Key related work and methods

A small but representative set of methods spanning optimization, hybrid model-fitting, and recent learning-based multi-view approaches:

- **SMPLify** (Bogo et al., ECCV 2016). The seminal optimization-based approach that fits the SMPL model to 2D joints by minimizing reprojection error subject to pose and shape priors. It remains the conceptual template for any multi-view fitting stage.
- **SPIN** (Kolotouros et al., ICCV 2019). A regressor trained jointly with an SMPL model-fitting loop, showing that combining network predictions with iterative optimization improves both accuracy and robustness.
- **EFT / Exemplar Fine-Tuning** (Joo et al., CVPR 2021). Fits SMPL to 2D annotations in-the-wild and distills the result into a fine-tuned network, demonstrating that model-fitting can generate pseudo-supervision for large-scale training.
- **ScoreHMR** (Stathopoulos et al., CVPR 2024). A score-guided diffusion approach to 3D human recovery that supports single-image, video, and multiple uncalibrated views. It is directly relevant because the project already has a `ScoreHMR` adapter and because it reasons explicitly in SMPL parameter space.
- **Learnable Triangulation of Human Pose** (Iskakov et al., ICCV 2019). Confidence-weighted multi-view triangulation using volumetric or algebraic aggregation; the current `RobustTriangulationModel` (`motionflow_mv/fusion/robust_triangulation.py`) is a lightweight differentiable variant.
- **MVGFormer** (Liao et al., CVPR 2024) and **MV-SSM** (Chharia et al., CVPR 2025). Recent geometry-aware transformer / state-space approaches to multi-view 3D pose; they support the design direction of keeping a geometry module (DLT) and adding a learned parametric refinement on top.

## 3. Relation to the current codebase

Relevant components and their current limitations:

- **`HumanMotionIR`** (`motionflow_mv/ir/human_motion_ir.py:15-57`) already reserves the exact SMPL parameter keys: `body_pose`, `global_orient`, `transl`, `betas`. This makes the IR a natural target for a fitting module.
- **GVHMR / ScoreHMR adapters** (`motionflow_mv/ir/gvhmr_adapter.py`, `motionflow_mv/ir/scorehmr_adapter.py`) convert single-view estimators into per-view `HumanMotionIR`s. They provide plausible per-view initialization but no cross-view consistency.
- **`fuse_multiple_irs`** (`motionflow_mv/ir/multiview_adapter.py:18-119`) extracts per-view 2D keypoints, runs a `FusionModule`, and rebuilds the IR. The body is not re-estimated: `_align_root` shifts only `transl`, and `_average_betas` averages shape vectors. This is the exact location where a true SMPL multi-view fitter should be inserted.
- **Fusion plugins** (`motionflow_mv/fusion/`) operate on `(T, V, J, 3)` 2D evidence. `DLTFusion` is geometrically optimal for reprojection, and `RobustTriangulation` / `ResidualRefiner` / `TemporalRefiner` add learned corrections, but none produce SMPL parameters. The `attention_v2` plugin (`motionflow_mv/fusion/attention_fusion_v2_module.py`) feeds flattened projection matrices to a transformer, hinting at a geometry-aware learned fitter, but it is currently marked unstable and unregistered.
- **Calibration** (`motionflow_mv/calibration/camera.py`) provides `K, R, t` and a projection matrix. Any SMPL fitter will consume these directly.
- **`demo_gvhmr_multiview_projection.py`** (`experiments/demo_gvhmr_multiview_projection.py`) already projects SMPL joints through virtual cameras and runs the fusion plugins, but it uses the ground-truth GVHMR SMPL parameters only to *generate* 2D observations; it does not recover SMPL parameters from multi-view observations.

## 4. Concrete recommendations

### 4.1 Add a `MultiViewSMPLFitter` post-fusion stage

Insert a new module, e.g. `motionflow_mv/fusion/smpl_fitter.py`, between the current fusion plugins and the final IR writer. It should:

1. **Input:** per-view 2D keypoints + confidences, calibrated `Camera`s, and an initial SMPL parameter guess (from the per-view GVHMR/ScoreHMR average or from the fused 3D skeleton).
2. **Objective:** minimize a weighted multi-view reprojection loss  
   \[
   \mathcal{L}_{\text{reproj}} = \sum_{v,j} w_{v,j} \|\pi_v(M_j(\theta, \beta, t)) - x_{v,j}\|_2^2
   \]
   where \(M_j(\cdot)\) is the SMPL joint position for joint \(j\), and \(\pi_v\) is the projection of view \(v\).
3. **Regularization:** add SMPL pose prior, shape regularization, bone-length consistency, temporal smoothness (velocity / acceleration), and optional ground-plane / gravity terms.
4. **Output:** a single `HumanMotionIR` with valid `body_pose`, `global_orient`, `transl`, and `betas`, plus uncertainty fields (reprojection residual, per-joint visibility, parameter covariance).

A minimal first implementation can use `smplx` + PyTorch autograd with `torch.optim.LBFGS` or `Adam`, initialized from the current pipeline. A later version can replace the optimizer with a neural optimizer (e.g., HeatFormer-style) once 3D supervision is available.

### 4.2 Initialize from the existing pipeline

- Use `DLTFusion` / `RobustTriangulation` to obtain an initial 3D skeleton and root location.
- Use the per-view `body_pose` / `global_orient` from the most confident view as the initial pose.
- Use `_average_betas` as the initial shape.
- Then run the multi-view SMPL fitter for a fixed number of iterations or until reprojection error converges.

### 4.3 Train and evaluate on the right data

- **Synthetic pre-training:** render AMASS motion clips through virtual multi-view rigs with known SMPL parameters, 2D detection noise, and occlusion. This provides direct supervision on the SMPL parameters themselves.
- **Real 3D-GT fine-tuning:** Human3.6M and CMU Panoptic provide 3D joint positions and/or SMPL annotations. Use these to train the fitting/refinement network with a 3D parameter loss, not just reprojection.
- **Reprojection sanity:** Shelf/Campus remain excellent fast benchmarks for reprojection error, but they do not provide SMPL ground truth. Do not use them as the primary training signal for the fitter.
- **Downstream validation:** extend `motionflow_mv/eval/metrics.py` to report MPJPE, PA-MPJPE, MRPE (metric root error), per-joint reprojection error, bone-length consistency, and temporal jitter.

### 4.4 Extend the plugin contract (future)

Consider extending the `FusionModule` interface so that plugins may optionally output SMPL parameters, or introduce a second plugin type `SMPLFitterModule` that consumes the output of a `FusionModule`. Keep the current 2D-fusion plugins untouched so existing experiments remain reproducible.

## 5. Open questions and risks

- **3D / SMPL ground-truth availability.** The current project has no SMPL annotations for Shelf/Campus. Any learned fitter needs Human3.6M, Panoptic, AMASS, or high-quality pseudo-labels from ScoreHMR.
- **Shape vs. pose ambiguity.** `betas` should be shared across the whole sequence, but per-frame pose may vary. Enforcing this without over-constraining fast motions is non-trivial.
- **Calibration sensitivity.** SMPL fitting couples camera calibration and body parameters. Errors in intrinsics/extrinsics will be absorbed into pose/shape; consider a joint camera+body refinement step for in-the-wild capture.
- **Computational cost.** Optimization per frame or per sequence is slower than a single forward pass. Decide whether the target is offline (CVPR) or real-time (ICRA/robotics).
- **Multi-person extension.** The current `select_best_person_group` is single-person. Multi-view SMPL fitting for multiple people requires cross-view person association and is out of scope for this note.
- **Licensing.** SMPL-X, Human3.6M, and CMU Panoptic have research-only or registration requirements. Ensure the project’s licensing goals are compatible before making any dataset a core dependency.

---

**Brief summary:** SMPL multi-view fitting is the natural next step after the current 2D-keypoint fusion pipeline. The IR already supports the required SMPL parameter keys, but the multi-view adapter only approximates them. A new `MultiViewSMPLFitter`—optimization-based at first, then possibly learned—should minimize multi-view reprojection with body, temporal, and shape priors, trained on Human3.6M/Panoptic/AMASS rather than reprojection-only Shelf data. This would produce physically consistent, metric-scale SMPL output suitable for CVPR/ICRA 2027.