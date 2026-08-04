# Occlusion Handling for Multi-View Human Pose Estimation in MotionFlow

> Status: Iteration 3 research note. Target venues: CVPR / ICRA 2027.  
> Sources: local codebase (`motionflow_mv/fusion`, `motionflow_mv/ir`, `experiments/`), design notes in `docs/`, and general multi-view pose literature. No external web resources were accessed for this report.

## 1. Problem Statement

In multi-view human pose estimation, **occlusion** is the failure mode where one or more camera views lose sight of a body joint because of self-occlusion, other people, scene objects, or camera placement. The result is a missing or low-confidence 2D keypoint in some views. Standard triangulation treats all views equally, so a single occluded-but-high-confidence view can bias the estimated 3D position. Learned fusion methods that do not explicitly model occlusion can overfit to the visible-view distribution and collapse when views drop out.

This topic addresses three questions for the MotionFlow multi-view extension:

1. **Detection:** How do we know a joint is occluded in a view?
2. **Robust fusion:** How do we fuse the remaining unoccluded views without letting the occluded view corrupt the 3D estimate?
3. **Recovery:** How do we use temporal, skeletal, and appearance priors to fill in missing information?

## 2. Key Related Work

### 2.1 Learnable Triangulation with Confidence Weighting (Iskakov et al., ICCV 2019)

Iskakov et al. introduce a **volumetric aggregation** approach where each view predicts a 2D heatmap, unprojected heatmaps are fused into a 3D volume, and the 3D joint is decoded from the volume. A simpler derivative is **confidence-weighted DLT**: each ray is weighted by the detector’s confidence, so low-confidence (likely occluded) views contribute less. This is the geometric backbone already used in MotionFlow’s `triangulate_confidence_weighted` and in `RobustTriangulationModel`.

*Relevance:* It is the strongest deterministic baseline; occlusion handling is implicit through confidence weighting.

### 2.2 TransFusion: Cross-view Fusion with Transformer for 3D Human Pose Estimation (Ma et al., BMVC 2021)

TransFusion uses a transformer to fuse per-view 2D pose features. The cross-view attention learns view-reliability scores, which naturally suppress features from views where the joint is occluded. It demonstrates that explicit **view-level attention** is more flexible than fixed confidence weights when occlusion patterns vary.

*Relevance:* Directly informs the `AttentionFusionModel` / `ViewAttentionFusion` path. The current model treats each joint independently; TransFusion suggests adding cross-view attention over all joints jointly.

### 2.3 Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation (Bragagnolo et al., ECCVW 2024)

This workshop paper fuses per-view **3D skeletons** (not just 2D keypoints) and refines them by minimizing reprojection error subject to **limb-length symmetry constraints**. Occluded views are down-weighted by reprojection residuals, and the skeleton prior enforces anatomical plausibility.

*Relevance:* The most directly applicable method to the current MotionFlow stack. Its optimization objective can be added to `ResidualRefinerModel` or `TemporalRefinerModel`.

### 2.4 Multiple View Geometry Transformers (MVGFormer) (Liao et al., CVPR 2024)

MVGFormer interleaves **learning-free geometry modules** with **learnable appearance modules**. The appearance module hallucinates features for occluded views, while the geometry module enforces multi-view consistency. This hybrid design addresses the “DLT ceiling” by adding learned occlusion reasoning without discarding geometry.

*Relevance:* Suggests upgrading `AttentionFusionModelV2` from a shallow camera-embedding model to a geometry-aware transformer that predicts occlusion masks.

### 2.5 Fast and Robust Multi-Person 3D Pose Estimation and Tracking from Multiple Views (Dong et al., IEEE T-PAMI 2021)

ZJU mvpose proposes **epipolar geometry-based association and robust triangulation** for multi-person scenes. It handles occlusion through cross-view matching and temporal tracking, rather than per-frame fusion alone.

*Relevance:* Useful for extending `pipeline_utils.py::select_best_person_group` beyond combinatorial matching to handle occlusion and multi-person scenarios.

## 3. Relation to the Current MotionFlow Codebase

### 3.1 Where occlusion is already handled implicitly

- **`motionflow_mv/fusion/triangulation.py` / `DLTFusion`**  
  Confidence-weighted DLT down-weights low-confidence detections. If the 2D detector sets the confidence of an occluded joint near zero, the occluded view is naturally ignored.

- **`motionflow_mv/fusion/robust_triangulation.py`**  
  `RobustTriangulationModel` learns per-view weights via attention and uses them in a differentiable DLT solver. In principle this can learn to suppress occluded views, but it is currently trained only on reprojection / DLT pseudo-labels.

### 3.2 Where occlusion is not yet explicitly modeled

- **`AttentionFusionModel` / `AttentionFusionModelV2`**  
  These fuse 2D keypoints with view attention but do not use an explicit occlusion mask, visibility flag, or skeleton prior. The attention is learned from pseudo-3D targets, so it may not generalize to occlusion patterns outside the training set.

- **`ResidualRefinerModel` and `TemporalRefinerModel`**  
  They refine a baseline 3D skeleton but do not explicitly enforce limb-length or symmetry constraints. Large outlier maxima in the Shelf evaluation (up to ~1044 px) are often caused by occluded-view contamination.

- **`HumanMotionIR`**  
  The IR has `uncertainty` and `quality` fields, but they are currently underutilized. They could store per-joint visibility or occlusion probability for downstream consumers.

- **`experiments/eval_all_plugins_shelf.py`**  
  The per-joint/per-view breakdown introduced here is the right tool to identify occlusion-related failures, but the script does not yet flag which joints are occluded.

## 4. Concrete Recommendations

### 4.1 Add explicit visibility / occlusion detection

Implement a module `motionflow_mv/fusion/visibility.py` that produces a per-joint, per-view visibility score:
