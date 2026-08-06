# MotionFlow-MultiView: The ICRA/CVPR 2027 Paper Story

> A living story document for the next submission cycle.  
> Last updated: 2026-08-06 while the anchor run
> `scripts/run_bayesian_tri_v2_large_scale_wsl.sh` is still training on the RTX 4090.

---

## 1. The one-sentence thesis

**We make calibrated multi-view 3D human pose estimation robust by treating geometry as the predictor and learning only the structured error that geometry cannot fix: a Bayesian precision-weighted triangulation step, an adaptive Gauss-Newton refinement, and a small residual MLP—wrapped in a plug-in module that drops into the existing MotionFlow pipeline.**

---

## 2. Why this problem matters

Multi-view video is the capture modality of choice wherever metric, world-grounded human pose matters: human-robot collaboration, sports analytics, immersive video, and robot policy training. The standard pipeline is:

1. Run a 2D keypoint detector in each view.
2. Triangulate with Direct Linear Transform (DLT).

DLT is simple, fast, and exact under perfect calibration and clean 2D observations. In practice it collapses under:

- **Occlusion / dropped views** – some rays are missing or wrong.
- **Detector noise and outliers** – 2D keypoints are biased, especially at limbs.
- **Calibration drift** – small principal-point / focal-length / rotation errors make rays miss.

End-to-end learned fusion can absorb noise, but regresses 3D joints directly and therefore throws away the metric, camera-consistent inductive bias of triangulation. We want the best of both worlds: **geometry first, learning second**.

---

## 3. Core insight: triangulate, then learn the leftover error

Our central design principle is a strict decomposition:

| Step | What it does | Why it is needed |
|------|--------------|------------------|
| **Intrinsic self-correction** | Predicts per-view principal-point (and optionally focal-length) offsets from the input 2D/confidence pattern. | Fixes the dominant real-world failure mode: small calibration drift. |
| **Ray-aware cross-view spatio-temporal attention** | Embeds 2D points as camera rays and exchanges information across views, joints, and time. | Produces context features for weighting and residual correction. |
| **Bayesian precision-weighted triangulation** | Predicts an anisotropic 2D covariance per view/joint, converts it to a precision, and feeds a differentiable weighted DLT. | Lets the model down-weight noisy/occluded views while keeping geometric triangulation. |
| **Adaptive Gauss-Newton refinement** | Runs 1–2 differentiable Gauss-Newton steps with learned per-joint damping. | Refines the DLT solution using the same camera model, not a black box. |
| **Residual refinement** | A small MLP adds a learned correction to the refined 3D pose, conditioned on pooled features and the geometric estimate. | Captures structured biases (detector bias, mild calibration drift, skeleton prior) that pure geometry misses. |

Because the residual head starts from a camera-consistent, metric estimate, it only has to learn a **small, structured correction**. This is the reason a 243 k–1.06 M parameter model can reach single-digit millimetre MPJPE on MPI-INF-3DHP.

---

## 4. Method snapshot

### 4.1 Model: Bayesian triangulation v2

The current anchor implementation is in
`motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`.
It extends the cross-view residual + principal-point correction model with:

1. **Anisotropic covariance head.** For every view and joint the head predicts the Cholesky factor of a 2×2 image-space covariance. The determinant gives a precision weight:
   ```
   precision = 1 / (l_xx * l_yy)
   ```
   The DLT weight is the product of detector confidence, predicted precision, and an soft visibility multiplier.

2. **Adaptive Gauss-Newton step.** After DLT, a learned per-joint damping factor controls 1–2 Gauss-Newton refinement iterations in world space. The Jacobian is derived from the pinhole projection equations, so the step is geometrically grounded.

3. **Epipolar-consistency auxiliary loss.** The predicted covariances also weight pairwise symmetric epipolar distances, giving an auxiliary loss that regularises the covariance head without extra 3D labels.

4. **Principal-point / intrinsic correction head.** A lightweight MLP predicts bounded per-view `(dx, dy)` (and optionally focal scale) before triangulation. It is supervised with the inverse of the applied perturbation, making the model self-calibrating at inference time.

5. **Cross-view spatio-temporal attention + residual MLP.** The backbone is a Transformer operating jointly over the `(time, view)` grid for each joint, followed by a tiny residual MLP:
   ```
   X = X_gn + MLP([pool(feat), X_gn])
   ```

### 4.2 Training objective

```
L = L_3D_MSE + λ_epipolar * L_epipolar + λ_pp * L_pp_offset + λ_reproj * L_reproj
```

- `L_3D_MSE` is the Euclidean distance to 3D ground truth in millimetres.
- `L_epipolar` regularises the covariance head via multi-view epipolar consistency.
- `L_pp_offset` supervises the predicted principal-point offset against the inverse of the training perturbation.
- `L_reproj` is an optional reprojection term.

Training-time camera perturbation is a core augmentation: per-clip rotation, translation, focal length, and principal-point noise. Validation is always run on the unperturbed calibration.

### 4.3 Plug-in integration

The fusion module is exposed as a `MultiViewFusionPlugin` inside MotionFlow. It consumes calibrated multi-view 2D keypoints and outputs a `HumanMotionIR` containing:

- `pose`: world-coordinate 3D joints `(T, J, 3)`.
- `uncertainty`: per-view weights, predicted precision, and per-joint confidence.
- `provenance`: source manifest, camera calibration hash, and plugin version.

This decouples multi-view fusion from the upstream single-view estimator and downstream robot retargeting.

---

## 5. Experimental narrative

### 5.1 Datasets and metrics

- **MPI-INF-3DHP** (14 views, 28 joints). Train on subjects 1 and 3, validate on subject 2 sequence 1. Primary metric: **MPJPE** (mm). We also report PA-MPJPE, PCK@50/100/150 mm, and AUC.
- **Human3.6M** (4 views, 17 joints). Train on subject 1, validate on subject 5 action 2.
- **Robustness protocol.** Evaluate under 2D Gaussian noise, random joint occlusion, 2D outliers, and controlled camera calibration perturbations (rotation, translation, focal length, principal point).
- **Runtime.** Report latency/throughput on a single RTX 4090.

### 5.2 Key results to date

| Dataset / Condition | Model | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---------------------|-------|-----------:|---------------:|-------|
| MPI-INF-3DHP S2/Seq1 | Raw DLT | 25.21 | 24.08 | geometric baseline, no learning |
| MPI-INF-3DHP S2/Seq1 | Temporal ray-attention (no residual) | 25.21 | 24.14 | learned weights alone do not help |
| MPI-INF-3DHP S2/Seq1 | Cross-view residual + PP (d=64, h=128, 20 ep) | **9.32** | **5.37** | current best single-model result |
| MPI-INF-3DHP S2/Seq1 | Cross-view residual + PP small (d=32, h=64) | 10.34 | 6.28 | 66 k params |
| Human3.6M S5/Act2 | CamPE + GraphJR (d=64, h=128) | **0.62** | **0.70** | 4-view rig |
| Human3.6M S5/Act2 | Cross-view residual + PP (d=64, h=128) | **5.24** | **4.84** | with PP correction |

The full result table is maintained in `docs/results_icra_cvpr_2027.md`.

### 5.3 Calibration robustness story

The biggest practical contribution is making multi-view pose robust to small calibration drift. On the current best principal-point correction model:

| Condition | MPJPE (mm) | PA-MPJPE (mm) |
|-----------|-----------:|---------------:|
| clean | 9.32 | 5.37 |
| rot_0.5° | 16.89 | 8.11 |
| rot_1.0° | 27.45 | 13.50 |
| trans_5mm | 10.61 | 5.20 |
| focal_1% | 19.13 | 8.07 |
| cxcy_3px | 11.41 | 5.75 |
| cxcy_5px | 13.87 | 6.61 |

Principal-point drift, which is catastrophic for raw DLT (>2 m), is reduced to a few millimetres by the learned correction head. Rotation and focal-length drift are still the largest remaining gaps and are the focus of the ongoing calibration-curriculum + Bayesian triangulation v2 run.

### 5.4 The anchor run

The current large-scale experiment is:

```
scripts/run_bayesian_tri_v2_large_scale_wsl.sh
```

Configuration:
- Model: `RayAttentionFusionModelBayesianTriV2`
- `d=128`, `residual_hidden=256`, `n_st_layers=3`
- 50 epochs, 2 000 training samples, batch size 8
- Principal-point loss weight 0.2, epipolar loss weight 0.05
- Training perturbation: principal point ±5 px, focal ±1 %
- Output: `outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.pth`

**Goal:** push MPI-INF-3DHP validation MPJPE below the ICRA/CVPR 2027 publishable threshold of **8.75 mm**.

The expected outcome of this run will determine the final numbers in the paper story. If it reaches <8.75 mm, the paper's primary accuracy claim is satisfied. If it lands slightly above, the remaining sections describe the ablations and extensions that close the gap.

---

## 6. Contribution list (for the abstract and introduction)

1. **Bayesian precision-weighted triangulation.** An anisotropic covariance head converts per-view 2D uncertainty into differentiable DLT weights, giving a principled way to fuse noisy/occluded views.
2. **Adaptive Gauss-Newton refinement.** A learned per-joint damping factor controls lightweight geometric refinement after DLT, keeping the solution metric and camera-consistent.
3. **Self-calibrating intrinsic correction.** A small head predicts per-view principal-point and focal-length corrections, making the system robust to realistic calibration drift.
4. **Geometry-first residual architecture.** The residual MLP only corrects the structured leftover error after triangulation and refinement, yielding compact models (<1.1 M params).
5. **MotionFlow plug-in and HumanMotionIR integration.** The fusion module is packaged as a drop-in plugin with explicit uncertainty and provenance, ready for downstream robot retargeting and policy training.

---

## 7. Paper arc

### Abstract
We present a compact, calibrated multi-view 3D human pose estimator that combines a differentiable weighted-DLT triangulation step with a learned anisotropic covariance, adaptive Gauss-Newton refinement, and a small residual correction. On MPI-INF-3DHP it reaches **X.XX mm** MPJPE (target <8.75 mm), and on Human3.6M it reaches **0.62 mm** MPJPE. The method is calibration-robust, runs at 12–195 clips/s on an RTX 4090, and is exposed as a `MultiViewFusionPlugin` inside MotionFlow.

### Introduction (narrative)
1. Multi-view capture is everywhere in robotics and immersive video.
2. DLT is the geometric baseline but brittle; end-to-end learning discards geometry.
3. We show that *triangulate first, then learn the residual* is the right decomposition.
4. Preview of results and contributions.

### Related Work
- Classical triangulation and robust statistics.
- Learnable triangulation / ray-attention methods.
- Temporal pose models.
- Residual learning and calibration-aware pose.
- MotionFlow / robot retargeting pipelines.

### Method
- 4.1 Ray-aware cross-view spatio-temporal attention.
- 4.2 Anisotropic covariance and Bayesian precision weights.
- 4.3 Weighted DLT.
- 4.4 Adaptive Gauss-Newton refinement.
- 4.5 Residual refinement head.
- 4.6 Principal-point / intrinsic correction.
- 4.7 Training losses and plug-in interface.

### Experiments
- Datasets and metrics.
- MPI-INF-3DHP accuracy and ablations.
- Human3.6M accuracy.
- Robustness matrix (noise, occlusion, outliers, calibration).
- Runtime and MotionFlow integration demo.

### Discussion
- What works: residual geometry decomposition, PP correction, compact size.
- What does not: discrete view gating, unfocused reprojection loss, naive camera embeddings.
- Future: multi-person association, SMPL fitting stage, cross-dataset domain adaptation.

---

## 8. Figures and tables to produce

1. **Architecture diagram.** Data flow: 2D keypoints → intrinsic correction → ray embedding → (T×V) attention → covariance + weight heads → weighted DLT → adaptive GN → residual MLP → 3D pose + uncertainty.
2. **MPI-INF-3DHP MPJPE bar chart.** Raw DLT, temporal ray-attention, residual model, PP correction, and Bayesian tri v2.
3. **Calibration robustness heatmap.** Same model under rot/trans/focal/PP perturbations.
4. **Human3.6M comparison.** Residual vs. CamPE+GraphJR.
5. **Variable-view curve.** MPJPE@k for k=2..14 on the best checkpoint.
6. **Runtime/latency plot.** RTX 4090 batch 1..16.
7. **GVHMR projection demo.** Clean vs. noisy view dropout.
8. **Failure-case heatmaps.** Per-joint/per-view error and PP correction magnitude.

---

## 9. Open questions and risk register

| Risk | Current status | Mitigation |
|------|----------------|------------|
| Anchor run does not reach <8.75 mm | Training in progress | Queue visibility-gated fusion v2, focal/rotation curriculum, and repeated-seed runs. |
| Rotation robustness still gap | rot_0.5° = 16.89 mm | Stronger extrinsic perturbation curriculum; separate rotation-aware correction head. |
| Focal-length robustness gap | focal_1% = 19.13 mm | Dedicated focal-scale loss; bound correction with dataset-specific ranges. |
| Cross-dataset transfer (H36M ↔ MPI) | Mixed-dataset H36M is poor (101 mm) | Domain adaptation wrapper, per-dataset pose heads, or larger mixed training. |
| Real-time throughput | 12–195 clips/s on RTX 4090 | FlashAttention / SDPA, distilled student, batch-size tuning. |
| Principal-point correction saturation | Diagnostic found saturation in an earlier checkpoint | Explicit reprojection supervision and pre-training in the current v2 run. |

---

## 10. Submission checklist

- [ ] Finalise numbers after `bayesian_tri_v2_large_scale` run completes.
- [ ] Update `docs/results_icra_cvpr_2027.md` with final MPJPE/PA-MPJPE/PCK/AUC.
- [ ] Generate architecture and robustness figures.
- [ ] Produce variable-view MPJPE@k curve.
- [ ] Run 3–5 repeated seeds and report mean±std.
- [ ] Fill in the abstract with concrete numbers.
- [ ] Write the introduction around the “triangulate, then residual” story.
- [ ] Add related-work section with the references from `docs/phase0_literature_audit.md`.
- [ ] Write the plug-in integration and robot-retargeting angle for ICRA.
- [ ] Create a 2-minute supplementary video showing noisy/occluded view handling.

---

## 11. Related files

- Implementation:
  - `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`
  - `motionflow_mv/fusion/principal_point_correction.py`
  - `motionflow_mv/fusion/epipolar_attention_bias.py`
  - `motionflow_mv/fusion/triangulation.py`
- Training:
  - `scripts/run_bayesian_tri_v2_large_scale_wsl.sh`
  - `experiments/train_bayesian_tri_v2_full_mpiinf3dhp.py`
- Evaluation and results:
  - `docs/results_icra_cvpr_2027.md`
  - `docs/experiment_log_icra_cvpr_2027.md`
  - `docs/paper_draft_icra_cvpr_2027.md`
- Roadmap:
  - `docs/iter_next_swarm_plan.md`
  - `docs/next_iteration_plan_swarm.md`
