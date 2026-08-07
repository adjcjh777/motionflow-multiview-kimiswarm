# Literature Novelty Positioning — MotionFlow-MultiView (ICRA/CVPR 2027)

> Status: read-only research / positioning proposal. No code or running experiments were modified.
> Date: 2026-08-07
> Repo: `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`

---

## 1. Current Technical State

### 1.1 Anchor results

The strongest reproducible results as of today are in `docs/results_icra_cvpr_2027.md` and `docs/icra_cvpr_2027_paper_story.md`:

| Setting | MPJPE | PA-MPJPE | Model / Notes |
|---|---:|---:|---|
| MPI-INF-3DHP S2/Seq1 | **8.35 mm** | 5.29 mm | Bayesian Tri v2 ensemble (stabilized + aug, d=128) |
| MPI-INF-3DHP S2/Seq1 | 9.03 mm | — | Bayesian Tri v2 single (d=128, stabilized) |
| H36M S5/Act2 | 0.62 mm | 0.70 mm | CamPE + GraphJR (d=64, h=128) |
| H36M S5/Act2 | 5.24 mm | 4.84 mm | Cross-view residual + PP (d=64, h=128) |

The accuracy threshold for a strong ICRA/CVPR 2027 story — **<8.75 mm on MPI-INF-3DHP S2/Seq1** — has already been met by an ensemble. The open problem is no longer *whether* the system can reach that threshold, but *how* to package the components into a defensible novelty claim.

### 1.2 Code-level architecture summary

The current code stack has three layers:

1. **Bayesian Tri v2** (`motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`)
   - Anisotropic 2-D covariance head → precision weights for weighted DLT.
   - Adaptive Gauss-Newton refinement with learned per-joint damping.
   - Principal-point (and optional focal) correction head.
   - Epipolar-consistency auxiliary loss (`epipolar_loss_weight=0.05`).

2. **OmniMultiViewFusion v2** (`motionflow_mv/fusion/omniview_fusion_v2.py`)
   - Adds explicit per-view/per-joint visibility gating (`_visibility_multiplier`).
   - Replaces dense joint self-attention with sparse graph-joint attention (`GraphJointAttentionV2`, `motionflow_mv/fusion/graph_joint_attention_v2.py`).
   - Keeps the Bayesian Tri v2 triangulation and refinement pipeline intact.

3. **OmniMultiViewFusion v3** (`motionflow_mv/fusion/omniview_fusion_v3.py`)
   - Adds hierarchical multi-scale temporal/cross-view/joint fusion (`_HierarchicalMultiscaleFusion`).
   - Adds camera-parameter conditioning (`_CameraConditioning`).
   - Adds epipolar-biased spatio-temporal transformer (`EpipolarBiasedTransformerEncoderLayer`).
   - Implemented, with a smoke-test trainer at `experiments/train_omniview_fusion_v3_mpiinf3dhp.py`, but not yet trained at scale.

Trainers: `experiments/train_omniview_fusion_v2_mpiinf3dhp.py` and `experiments/train_omniview_fusion_v3_mpiinf3dhp.py` share the same loss mix (3-D MSE, visibility BCE, uncertainty NLL, temporal velocity, bone-length, epipolar). Evaluation scripts include `experiments/eval_omniview_fusion_v2_mpiinf3dhp.py`, `eval_omniview_fusion_v2_variable_views.py`, and `eval_omniview_fusion_v2_camera_perturbation.py`.

---

## 2. Literature Map and the Gap We Actually Occupy

### 2.1 Classical triangulation and robust statistics

Classical multi-view pose is:
1. Detect 2-D keypoints per view.
2. Triangulate via Direct Linear Transform (DLT) or bundle adjustment (BA).

DLT is exact under perfect calibration and clean 2-D observations, but collapses under occlusion, detector noise, outliers, and calibration drift. Robust statistics (RANSAC, M-estimators, Huber) reduce outlier sensitivity but still treat each view equally and do not learn detector-specific or skeleton-aware corrections.

### 2.2 Learnable triangulation

**Isakov et al., *Learnable Triangulation of Human Pose*, ICCV 2019** — the canonical baseline — learns per-view confidence weights for triangulation. It keeps triangulation at the center, which is philosophically close to our approach, but:
- It uses isotropic scalar confidence, not anisotropic covariance.
- It has no explicit occlusion/visibility gating.
- It has no intrinsic/calibration correction.
- It has no temporal or skeleton-aware reasoning.

**VoxelPose / Fast and Robust Multi-Person 3D Pose Estimation and Tracking from Multiple Views (Dong et al., T-PAMI 2021)** — voxel-based 3-D heatmap fusion. Strong for multi-person, but computationally heavy and not naturally differentiable end-to-end with camera geometry.

### 2.3 Transformer-based multi-view fusion

**TransFusion (arXiv 2110.09554)** and related cross-view transformers fuse per-view 2-D features directly into 3-D pose. They typically:
- Regress 3-D joints or volumetric features end-to-end, discarding the metric triangulation inductive bias.
- Do not explicitly model camera parameters inside the attention or triangulation step.
- Report MPJPE on clean benchmarks but rarely report calibration-robustness or variable-view curves.

**Occlusion-aware multi-view fusion (arXiv 2408.15810)** and visibility-gating work are close to our v2 visibility head, but they are usually trained with synthetic occlusion masks and do not also combine graph-joint attention, uncertainty-weighted triangulation, and adaptive Gauss-Newton refinement.

### 2.4 Geometry-aware transformers

Several recent papers inject camera parameters or epipolar geometry into transformers:
- Camera positional encoding (CamPE) variants.
- Epipolar attention bias for feature matching.
- Ray embeddings for triangulation.

Our v3 is adjacent to this literature, but most prior work either:
- Conditions features on camera parameters without using them in the triangulation step, or
- Uses epipolar constraints only as a loss or post-processing step, not as an attention bias inside the transformer.

**RapidPoseTriangulation (arXiv 2503.21692)** focuses on speed for multi-person triangulation, not on accuracy under calibration drift or occlusion.

### 2.5 What is genuinely missing — and where we are positioned

The literature has strong methods for each sub-problem in isolation, but the *combination* is still rare:

| Capability | Classical | Learnable Tri | TransFusion | VoxelPose | Omni v2 | Omni v3 |
|---|---|---|---|---|---|---|
| Differentiable triangulation at core | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| Anisotropic per-view uncertainty | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Explicit occlusion/visibility gating | ✗ | ✗ | partial | ✗ | ✓ | ✓ |
| Skeleton-aware graph attention | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Intrinsic/calibration self-correction | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Temporal + cross-view transformer | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |
| Multi-scale temporal/joint fusion | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Geometry-biased attention | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Robot-ready plugin / IR | ✗ | ✗ | ✗ | ✗ | partial | partial |

Our defensible novelty claim is therefore:

> **A compact, geometry-first multi-view fusion module that unifies calibration self-correction, visibility gating, skeleton-aware graph attention, anisotropic uncertainty-weighted triangulation, adaptive Gauss-Newton refinement, and geometry-regularized spatio-temporal attention — packaged as a plug-in robot-ready component.**

This is not a single new layer; it is a *systematic decomposition* of the problem that lets each learning component stay small and interpretable. That is the story that differentiates us from both end-to-end transformers and pure triangulation papers.

---

## 3. Concrete Implementable Next Steps

These steps are grounded in the existing code and can be implemented without touching the running training jobs.

### 3.1 Harden v2 to a reproducible single-model anchor

**Goal:** Reach a single-model MPJPE ≤ 8.5 mm on MPI-INF-3DHP S2/Seq1 without ensemble.

**Files:**
- `experiments/train_omniview_fusion_v2_mpiinf3dhp.py`
- `motionflow_mv/fusion/omniview_fusion_v2.py`
- `docs/results_icra_cvpr_2027.md`

**Actions:**
1. Warm-start from `outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth` (d=128).
2. Freeze encoder/ST-transformer for 5 epochs, train only graph + visibility heads; then unfreeze for end-to-end training (`--warm_start_freeze_epochs 5`).
3. Use the same perturbation schedule as the Bayesian Tri v2 stabilized run: principal point ±5 px, focal ±1 %.
4. Train with full MPI-INF-3DHP train (subjects 1 and 3), validate on S2/Seq1, and save to `outputs/omniview_fusion_v2_mpiinf3dhp.pth`.

**Why:** The v2 design doc already predicts that visibility + graph should improve over the Bayesian Tri v2 single model. A strong single-model result removes the need to rely on an ensemble in the paper.

### 3.2 Launch controlled v3 ablation matrix on A800

**Goal:** Measure the marginal contribution of each v3 component.

**Files:**
- `experiments/train_omniview_fusion_v3_mpiinf3dhp.py`
- `motionflow_mv/fusion/omniview_fusion_v3.py`
- `experiments/eval_omniview_fusion_v2_mpiinf3dhp.py` (can be adapted for v3)

**Actions:**
1. Add a small adapter script `experiments/eval_omniview_fusion_v3_mpiinf3dhp.py` that loads `OmniMultiViewFusionV3` and runs the same validation as v2.
2. Run the ablation matrix from `docs/omniview_fusion_v3_design.md` §5:
   - A (v3 full): multiscale + camera cond + epipolar bias.
   - B: no multiscale.
   - C: no camera conditioning.
   - D: no epipolar bias.
   - E (v2 baseline): all three off.
3. Use d=64/h=128 for fast turn-around, then scale the winning config to d=128.
4. Warm-start all v3 runs from the best v2 checkpoint to reduce training time.

**Expected deliverable:** A table of MPJPE/PA-MPJPE for A–E, plus per-component Δ.

### 3.3 Build a calibration-robustness benchmark that is publishable

**Goal:** Convert the existing robustness logs into a paper-quality table/figure.

**Files:**
- `experiments/eval_omniview_fusion_v2_camera_perturbation.py`
- `docs/results_icra_cvpr_2027.md`
- New: `experiments/eval_calibration_robustness_v2v3.py`

**Actions:**
1. Wrap the existing per-perturbation evaluation into a single script that:
   - Loads a checkpoint,
   - Applies rotation ±0.5°/±1.0°, translation ±5 mm/±10 mm, focal ±1 %/±2 %, principal point ±3 px/±5 px,
   - Reports MPJPE, PA-MPJPE, and PCK@50/100/150 for each.
2. Run it on the best v2 and (once trained) best v3 checkpoints.
3. Generate a LaTeX-ready table and a heatmap figure in `docs/figures/`.

**Why:** Calibration robustness is the strongest differentiator against learnable-triangulation and transformer baselines. The current numbers are scattered across multiple logs; consolidating them into a reproducible script makes the paper claim defensible.

### 3.4 Implement a variable-view inference protocol

**Goal:** Publish MPJPE@k for k = 2..14 views, as in the existing smoke results.

**Files:**
- `experiments/eval_omniview_fusion_v2_variable_views.py`
- `docs/figures/variable_views_crossview_residual_smoke.png`

**Actions:**
1. Generalize the existing variable-view script to accept `OmniMultiViewFusionV2` and `V3`.
2. For each k, sample N view subsets, run inference, and average MPJPE.
3. Produce a plot: x-axis = k, y-axis = MPJPE, with error bars.
4. Compare with the raw-DLT baseline (should be far worse at low k) and with the best v2/v3 models.

**Why:** Variable-view performance is a practical concern for robot deployment (cameras can fail). Most literature only reports full-view MPJPE.

### 3.5 Write the related-work and method sections with explicit contrasts

**Goal:** Make the paper’s novelty claim precise and reviewer-proof.

**Files to create/update:**
- `docs/paper_method_section_draft.md`
- New: `docs/paper_related_work_draft.md`

**Actions:**
1. Draft a paragraph for each of the following, with explicit contrasts:
   - vs. Isakov et al. (scalar confidence vs. anisotropic covariance + visibility + calibration).
   - vs. TransFusion / end-to-end transformers (regression vs. geometry-first triangulation).
   - vs. VoxelPose (voxel heatmaps vs. ray-aware feature + DLT).
   - vs. recent epipolar/camera-PE work (geometry as attention bias inside our transformer, not only as a loss or pose encoding).
2. Add a method subsection: "Geometry-first decomposition" with the table from §4.1 of `docs/paper_story_system_v2.md`, updated to include v3.
3. Include the HumanMotionIR / plugin angle for the ICRA systems track.

---

## 4. Paper Framing Options

Given the above, there are two strong and non-exclusive framing options:

### Option A: Method paper (CVPR / ICRA)

**Title style:** *Geometry-First Multi-View Human Pose Estimation with Anisotropic Uncertainty and Geometry-Regularized Attention*

**Core claim:** A compact, interpretable fusion module that keeps triangulation at the center and learns only structured residuals (calibration, visibility, uncertainty, skeleton, attention).

**Key experiments:**
- MPI-INF-3DHP accuracy + ablations.
- H36M cross-dataset transfer.
- Calibration-robustness matrix.
- Variable-view MPJPE@k.
- Runtime on RTX 4090.

### Option B: Systems / robotics paper (ICRA)

**Title style:** *MotionFlow-MultiView: A Reproducible, Calibration-Robust Multi-View Pose Pipeline for Robot Learning*

**Core claim:** The fusion module is one replaceable plugin inside a larger reproducible workflow (HumanMotionIR, quality gates, robot profiles).

**Key experiments:**
- Same accuracy/robustness numbers as Option A.
- Downstream robot retargeting/policy metrics (if available).
- Docker reproducibility / runtime / deployment story.

**Recommendation:** Submit as a **method paper to CVPR** and as a **systems paper to ICRA**, with the same core numbers but different emphasis. The code is already structured to support both: `OmniMultiViewFusionV2/V3` are the method contribution; `HumanMotionIR` and the plugin registry support the systems contribution.

---

## 5. Risk Register and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| v3 multi-scale fusion does not beat v2 single-model | Medium | High | The ablation matrix will tell us quickly; if negative, v2 is still a strong standalone method. |
| Variable-view MPJPE@k still poor for k < 6 | Medium | Medium | Use visibility gating + uncertainty to down-weight missing views; report both raw numbers and relative improvement over DLT. |
| Calibration-robustness numbers are not better than Bayesian Tri v2 | Medium | Medium | The v3 epipolar-bias + camera conditioning is designed exactly for this; run the full matrix before deciding. |
| Literature reviewers say "this is just Isakov + transformers" | Low | High | Emphasize the *systematic combination* and the explicit geometry-first decomposition; cite each contrast. |
| Ensemble numbers cannot be reproduced as a single model | Low | Medium | The v2 single-model run is already planned; if it underperforms, use the ensemble as the final result and explain warm-starting. |

---

## 6. Top 5 Action Items (in order)

1. **Run the v2 d=128 full-data warm-start** (`experiments/train_omniview_fusion_v2_mpiinf3dhp.py`) from the stabilized Bayesian Tri v2 checkpoint, freezing old weights for 5 epochs; target single-model MPJPE ≤ 8.5 mm on MPI-INF-3DHP S2/Seq1.
2. **Implement and run the v3 ablation matrix** (`experiments/train_omniview_fusion_v3_mpiinf3dhp.py`) with the four configurations (full, no-multiscale, no-camera-cond, no-epipolar-bias), starting from the best v2 checkpoint; evaluate with a new `eval_omniview_fusion_v3_mpiinf3dhp.py`.
3. **Create a unified calibration-robustness evaluation script** (`experiments/eval_calibration_robustness_v2v3.py`) that produces a paper-ready table and heatmap for rotation, translation, focal, and principal-point perturbations.
4. **Generalize the variable-view inference protocol** to v2/v3, generate the MPJPE@k plot, and save it to `docs/figures/variable_views_omniv2v3_mpiinf3dhp.png`.
5. **Draft the related-work section** (`docs/paper_related_work_draft.md`) with explicit contrasts to Isakov et al., TransFusion, VoxelPose, and recent epipolar/camera-PE methods, and update `docs/paper_method_section_draft.md` with the v3 architecture.

---

## 7. Files Referenced

- `motionflow_mv/fusion/omniview_fusion_v2.py`
- `motionflow_mv/fusion/omniview_fusion_v3.py`
- `motionflow_mv/fusion/graph_joint_attention_v2.py`
- `motionflow_mv/fusion/visibility_gated_fusion_v2.py`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`
- `experiments/train_omniview_fusion_v2_mpiinf3dhp.py`
- `experiments/train_omniview_fusion_v3_mpiinf3dhp.py`
- `experiments/eval_omniview_fusion_v2_mpiinf3dhp.py`
- `experiments/eval_omniview_fusion_v2_variable_views.py`
- `experiments/eval_omniview_fusion_v2_camera_perturbation.py`
- `docs/results_icra_cvpr_2027.md`
- `docs/icra_cvpr_2027_paper_story.md`
- `docs/omniview_fusion_v3_design.md`
