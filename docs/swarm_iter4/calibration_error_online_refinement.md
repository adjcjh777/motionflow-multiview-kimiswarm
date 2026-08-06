# Robustness to Calibration Error and Online Camera Refinement

## 1. Topic survey

The current `ray_attention` fusion model (`motionflow_mv/fusion/ray_attention_model.py`) is built around a fixed, perfectly calibrated camera rig: it back-projects 2D keypoints with `K⁻¹`, rotates rays by `Rᵀ`, uses camera centers `-Rᵀt`, and feeds a differentiable weighted DLT triangulator. This design has a strong geometric inductive bias and gives near-perfect results when the calibration is exact, but it has no mechanism to correct `K`, `R`, or `t` when they are noisy or drift. Attention weights can only suppress a bad *view*; they cannot fix a bad *camera*.

Real-world multi-view datasets such as Shelf and Campus are captured with one-time calibrations that contain small but non-negligible errors. Cross-dataset evaluation in `docs/design_v3.md` shows that learned plugins trained on Shelf still fail to generalize to Campus, partly because the model memorizes dataset-specific camera geometry rather than learning calibration-invariant fusion. A practical ICRA/CVPR 2027 submission therefore needs to demonstrate robustness to calibration error and, ideally, the ability to refine cameras online from 2D keypoints alone.

Key prior threads are already present in the project:

* `docs/swarm_iter3/calibration_refinement.md` frames the problem but is incomplete.
* `docs/swarm_iter3/epipolar_constraints.md` identified reprojection/epipolar losses as useful regularizers.
* `experiments/eval_ray_attention_robustness.py` evaluates occlusion/outlier robustness, but not calibration noise.

The missing piece is a concrete plan for *calibration-aware training and online refinement* that fits the existing `ray_attention` architecture.

## 2. Concrete recommendations

1. **Add a differentiable camera-refinement head.**  After the attention block, add a small MLP predicting per-camera corrections `ΔK`, `ΔR`, `Δt`.  Parameterize `ΔR` with axis-angle / Lie algebra and intrinsics with log focal length and principal-point offsets.  Initialize to zero and regularize toward the original calibration so the model can fall back when the input calibration is already good.

2. **Train with a reprojection-consistency loss.**  Project the triangulated 3D joints back into every view and compute a robust reprojection error (Huber or Geman–McClure) weighted by the predicted per-view weights.  This loss directly couples camera parameters, attention weights, and 3D points, forcing the network to discover consistent geometry rather than memorizing a single rig.

3. **Inject synthetic calibration errors during training.**  Augment each training sample by perturbing `K` (focal length ±5%, principal point ±10 px), `R` (small rotation noise), and `t` (position noise).  Use the perturbed cameras as input and the unperturbed cameras as the supervision target for the refinement head.  This yields a controlled robustness ablation without requiring new real data.

4. **Provide an lightweight online bundle-adjustment (BA) fallback.**  At inference time, keep the learned weights as observation covariances and run a short temporal BA that optimizes camera intrinsics/extrinsics and 3D joints to minimize the weighted reprojection error over a small window.  This is the classical safety net: when calibration is unknown or drifts, the system can still self-correct without retraining.

5. **Define a calibration-robustness evaluation protocol.**  Measure MPJPE as a function of injected calibration-noise magnitude on (a) the synthetic generator, (b) Shelf, and (c) Campus.  Compare: (i) plain `ray_attention`, (ii) `ray_attention` + refinement head, and (iii) DLT baseline.  Report reprojection error before/after refinement and cross-dataset zero-shot numbers.

## 3. Potential risks

* **Degenerate refinement.**  Without constraints, the refinement head may collapse all cameras to a single viewpoint or drift to a degenerate configuration.  Mitigate via the regularizer toward the initial calibration and via the reprojection-consistency loss.
* **Overfit to training rigs.**  A `ΔK` head trained only on synthetic rigs with similar focal lengths may not transfer to very different real cameras.  Normalize inputs by focal length and principal point before embedding.
* **Ambiguity between calibration error and bad 2D detections.**  A mis-calibrated camera and an outlier keypoint can produce similar residuals.  The model already predicts per-joint per-view weights; use them as soft masks so that the refinement head focuses on cameras with consistent residuals.
* **Inference cost.**  Online BA per sequence adds non-trivial compute.  Keep it optional/fallback and cap iterations; use the learned head as the default fast path.
* **Training instability.**  Jointly optimizing pose and camera parameters is harder than fixed-camera triangulation.  Start with the existing `ray_attention` weights frozen, add the refinement head, and use a smaller learning rate.

## 4. Fit with the paper plan

This work package directly supports the ICRA/CVPR 2027 story: *the proposed ray-aware attention fusion is not only accurate with perfect calibration, but also robust enough for real-world deployment*.  It adds a "Robustness to calibration error and online refinement" section with two kinds of evidence:

1. **Controlled synthetic ablations** showing graceful degradation under calibration noise and recovery via the refinement head.
2. **Real-data cross-dataset validation** (Shelf → Campus) demonstrating that the refined model retains geometric accuracy even when the input calibration is imperfect.

Together with the existing occlusion/outlier experiments, this will position `ray_attention` as a practical, deployable multi-view fusion plugin rather than a calibration-dependent method.
