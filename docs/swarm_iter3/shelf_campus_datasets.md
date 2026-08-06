# Shelf / Campus Datasets for Multi-View Human Pose Fusion

**Topic:** `shelf_campus_datasets`  
**Scope:** Multi-view 2D/3D human pose estimation, calibrated multi-camera fusion, SMPL-compatible intermediate representations.  
**Target venues:** CVPR / ICRA 2027.  
**Date:** 2026-08-04.

---

## 1. Problem Statement

The MotionFlow-MultiView project needs a small, calibrated, real-world multi-view dataset to validate its plugin-based fusion pipeline (DLT, attention fusion, robust triangulation, residual/temporal refinement) and its `HumanMotionIR` integration with single-view estimators such as GVHMR. The **Shelf** and **Campus** datasets are the natural candidates because they are compact, public, widely used in the multi-view pose literature, and already partially integrated into the codebase.

This topic addresses a specific gap in the current workflow: the project has strong geometric baselines and a growing family of learned fusion plugins, but real progress is currently measured only by **2D reprojection error** because full 3D ground truth (GT) is not loaded locally. The key question is how to use Shelf/Campus—and eventually Human3.6M—to move from a reprojection-sanity validation loop to a true 3D-supervised training and evaluation loop, and how to align the dataset conventions (mm vs. meters, 14-joint vs. 17-joint skeletons, camera parameter formats) with the `HumanMotionIR` standard.

---

## 2. Key Related Work and Methods

### 2.1 Shelf / Campus (Belagiannis et al., CVPR 2014)

Belagiannis et al. introduced the Shelf and Campus benchmarks for **3D pictorial structures** and multi-human pose estimation. They remain the de facto small-scale benchmarks for multi-view methods.

| Dataset | Views | People | Frames | Resolution | Unit |
|---------|-------|--------|--------|------------|------|
| Campus  | 3     | 3      | ~2,000 | 640×480    | mm   |
| Shelf   | 5     | 5      | ~3,200 | 1024×768   | mm   |

Both datasets ship with:
- Camera intrinsics `K = [fx, fy, cx, cy]`
- Extrinsics `R`, `T` (world-to-camera)
- 2D/3D GT annotations (14-joint annotation set; 17-joint COCO-like detections in VoxelPose-derived files)

These are the datasets on which the current `VoxelPoseShelfLoader` / `VoxelPoseCampusLoader` operate.

### 2.2 VoxelPose (Tu et al., ECCV 2020)

VoxelPose is the first strong **volumetric** baseline for Shelf/Campus. It discretizes 3D space, projects multi-view features into a common voxel grid, and regresses 3D joints. For our purposes, the more important artifacts are the pre-processed calibration files and 2D detections (`pred_shelf_maskrcnn_hrnet_coco.pkl`, `pred_campus_maskrcnn_hrnet_coco.pkl`) that the current loader consumes. VoxelPose establishes the **standard train/test split** and the expected 17-joint COCO layout used by the project's 2D predictions.

### 2.3 Learnable Triangulation (Iskakov et al., ICCV 2019)

Iskakov et al. showed that a neural network can predict **per-view confidence weights** that feed into a differentiable triangulation layer. Their insight—"let the geometry do the 3D reasoning, let the network select trustworthy views"—directly informs our `RobustTriangulationModel`, which predicts per-view weights and solves a weighted DLT system. Their work also establishes the principle that learned fusion should not regress 3D coordinates blindly; it should exploit the calibrated camera geometry.

### 2.4 Human3.6M (Ionescu et al., TPAMI 2014)

Human3.6M is the canonical large-scale 3D pose dataset. It provides true 3D GT for 11 actors (S1,5,6,7,8 train; S9,11 test), enabling 3D-supervised training of fusion heads. The current design documents identify Human3.6M as the **next dataset** once Shelf/Campus validation is stable. For CVPR/ICRA 2027, Human3.6M will be essential to claim that learned fusion generalizes beyond the tiny Shelf/Campus domain.

### 2.5 Geometry-Aware Transformers / State-Space Models

Recent work such as **MVGFormer** (Liao et al., CVPR 2024) and **MV-SSM** (Chharia et al., CVPR 2025) fuses multi-view evidence with explicit camera-parameter embeddings and temporal state-space modeling. The project's `AttentionFusionModelV2` (camera-projection embedding) and `TemporalRefinerModel` (Bi-GRU window) are early steps in this direction, but currently underperform the DLT baseline, indicating that the architecture/loss design must mature.

---

## 3. Relation to the Current Codebase

The following files already implement Shelf/Campus support and the surrounding fusion stack:

- `motionflow_mv/data/voxelpose_loader.py` — `VoxelPoseShelfLoader` and `VoxelPoseCampusLoader` load calibration (`calibration_shelf.json`, `calibration_campus.json`) and 2D predictions (`pred_*_maskrcnn_hrnet_coco.pkl`). The loader converts VoxelPose's `R, T` convention into the project's `Camera(K, R, t)` with `t = -R @ T`.
- `motionflow_mv/calibration/camera.py` — lightweight pinhole `Camera` exposing `projection_matrix`.
- `motionflow_mv/fusion/triangulation.py` — confidence-weighted DLT.
- `motionflow_mv/fusion/fusion_module.py` — plugin interface; `DLTFusion` is the geometric baseline.
- `motionflow_mv/fusion/robust_triangulation.py` — learned per-view weights + differentiable weighted DLT.
- `motionflow_mv/fusion/attention_model_v2.py` / `attention_fusion_v2_module.py` — geometry-aware attention fusion (currently experimental/unstable).
- `motionflow_mv/fusion/residual_refiner.py` / `temporal_refiner.py` — post-DLT refinement modules.
- `experiments/eval_all_plugins_shelf.py` / `eval_all_plugins_campus.py` — evaluate all registered plugins by **reprojection error**.
- `experiments/train_attention_fusion_shelf.py` — trains `AttentionFusionModel` using DLT triangulated output as pseudo-GT.
- `experiments/prepare_shelf_dataset.py` — precomputes matched frames with DLT pseudo-targets.

### Current empirical status (from `docs/design_v3.md` and `README.md`)

On Shelf frames 300–600 (5 views):

| Plugin | Mean reprojection (px) | Median (px) |
|--------|------------------------|-------------|
| `dlt` | 9.88 | 5.52 |
| `attention` (Shelf-finetuned) | 80.42 | 58.90 |
| `robust_triangulation` | 10.65 | 5.97 |
| `residual_refiner` | 13.11 | 9.79 |
| `temporal_refiner` | ~9.9 | ~5.5 |

Key observations:
- DLT is already excellent on this metric.
- `robust_triangulation` is close but still does not beat DLT.
- `attention` lags far behind, and `attention_v2` is unstable.
- Cross-dataset zero-shot on Campus shows `dlt`/`robust_triangulation`/`temporal_refiner` transfer well (~1.5 px), but `attention` does not (318 px).

This confirms that the project is in a **reprojection-only validation regime**. Without true 3D GT, the learned modules cannot be trained to minimize real-world 3D error, and the geometric baseline is near-optimal for the reprojection metric.

---

## 4. Concrete Recommendations

### 4.1 Immediate: Load True 3D Ground Truth for Shelf/Campus

The most impactful next step is to extend the loader and eval scripts to read the actual `annotation_3d.json` files and report **MPJPE / PA-MPJPE** in addition to reprojection error.

**Action items:**
- Add `motionflow_mv/data/shelf_campus_3d_loader.py` that parses `annotation_3d.json` and aligns 3D GT to the 17-joint COCO predictions via joint-name mapping.
- Extend `experiments/eval_all_plugins_shelf.py` and `eval_all_plugins_campus.py` to compute `mpjpe` and `pa_mpjpe` against GT when available.
- Update `experiments/train_attention_fusion_shelf.py` to use real 3D GT as the target (with reprojection loss as auxiliary) instead of DLT pseudo-labels.

This will finally allow the project to ask: "does the learned fusion improve **real 3D accuracy**, not just reprojection?"

### 4.2 Short-Term: Standardize the Skeleton and Units

The codebase currently mixes 14-joint annotation GT, 17-joint COCO detections, and mm/meter units. This is a recurring source of scale bugs.

**Action items:**
- Define a canonical 17-joint skeleton for `HumanMotionIR` and map Shelf/Campus GT to it.
- Encode the `length_unit` field in `HumanMotionIR.coordinate_system` (already part of the dataclass) and enforce meters downstream.
- Add unit tests that verify a known 3D point reprojects correctly to 2D under the loaded calibration.

### 4.3 Short-Term: Train `RobustTriangulation` and `AttentionFusionV2` with 3D Supervision

`RobustTriangulation` is geometrically principled and already near-DLT. With real 3D GT, it is the most likely learned plugin to show an improvement. `AttentionFusionV2` is currently unstable because it concatenates raw projection-matrix values to 2D coordinates without normalization; stabilize it first.

**Action items:**
- For `RobustTriangulationModel`, add `L_MPJPE + λ_reproj * L_reproj + λ_bone * L_bone`.
- For `AttentionFusionModelV2`, normalize the projection matrix embedding by focal length and principal point, or use ray-direction features instead of raw P-matrix entries.
- Use Shelf/Campus as the fast dev loop; move the best architecture to Human3.6M.

### 4.4 Medium-Term: Integrate with `HumanMotionIR` End-to-End

The `MultiViewAdapter` in `motionflow_mv/ir/multiview_adapter.py` is thin and currently only works when `per_view_2d` is populated. It does not yet reproject SMPL joints from per-view IRs.

**Action items:**
- Implement SMPL reprojection inside `multiview_adapter.py` so that GVHMR/ScoreHMR per-view IRs can be fused without external 2D keypoints.
- Run `experiments/demo_gvhmr_multiview_projection.py` on Shelf/Campus frames to compare single-view GVHMR with multi-view fused output.

### 4.5 Publication Angle

For CVPR/ICRA 2027, the strongest story is not "we beat DLT on reprojection." It is:

> "A modular, SMPL-compatible multi-view fusion stack that systematically compares DLT, learned triangulation, residual refinement, and temporal refinement under a common `HumanMotionIR`, with uncertainty-aware outputs for downstream robotics."

Shelf/Campus provide the fast, calibrated benchmark; Human3.6M provides the 3D-supervised scale; the robot downstream task provides the ICRA motivation.

---

## 5. Open Questions / Risks

| Risk / Question | Impact | Mitigation |
|-----------------|--------|------------|
| **No local 3D GT** for Shelf/Campus currently loaded. | High — blocks 3D-supervised training and credible publication claims. | Parse `annotation_3d.json`; if unavailable, register Human3.6M. |
| **DLT is already near-optimal on reprojection.** | Medium — learned methods cannot demonstrate value on this metric alone. | Switch to MPJPE/PA-MPJPE; train with 3D GT + bone-length + temporal losses. |
| **Scale/unit mismatch** (mm in Shelf/Campus vs. meters in `HumanMotionIR`). | Medium — causes silent errors in fusion and SMPL fitting. | Enforce `length_unit` metadata; add unit tests. |
| **`attention_v2` instability.** | Medium — geometry-aware attention is a key publication direction but currently fails. | Normalize camera embeddings; start from `RobustTriangulation` geometry. |
| **Skeleton mismatch** between 14-joint GT and 17-joint detections. | Medium — joint alignment affects metrics. | Build and document a canonical joint mapping. |
| **Domain gap** when moving from Shelf/Campus to Human3.6M/3DPW. | Medium — small datasets risk overfitting. | Use synthetic AMASS pre-training and strong augmentation. |
| **Multi-person scenes** are out of scope for v3. | Low-Medium — limits real-world applicability. | Defer to v4; keep current single-person `select_best_person_group` assumption. |

---

## 6. References

1. Belagiannis et al., "3D Pictorial Structures for Multiple Human Pose Estimation," CVPR 2014.
2. Tu et al., "VoxelPose: Towards Multi-camera 3D Human Pose Estimation in Wild Environment," ECCV 2020.
3. Iskakov et al., "Learnable Triangulation of Human Pose," ICCV 2019.
4. Ionescu et al., "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments," TPAMI 2014.
5. Chharia et al., "MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation," CVPR 2025.
6. Liao et al., "Multiple View Geometry Transformers for 3D Human Pose Estimation," CVPR 2024.