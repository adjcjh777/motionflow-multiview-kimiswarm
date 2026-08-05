# Cross-View Matching for Multi-View Human Pose Estimation

## 1. Problem statement

In a calibrated multi-view setup, the input to 3D human pose estimation is rarely a clean set of per-frame, per-view detections of the *same* person. Before any triangulation or learned fusion can take place, the system must answer:

> Given V views and up to P detections per view, which 2D poses belong to the same physical person?

This is the **cross-view matching** problem. It sits upstream of the `FusionModule` plugins in `motionflow-multiview`, yet it is not represented in the current plugin interface. Failure modes include:

- **Person-ID switches**: a person is correctly detected in each view but associated with different IDs across views, producing a fused skeleton from mismatched body parts or two different people.
- **Missing/occluded detections**: some views have no detection for a person, so a simple positional or bipartite match is under-constrained.
- **Scale/calibration drift**: matching in 2D image space ignores depth; matching purely in 3D ray space requires accurate camera calibration and a common ground plane.
- **Multi-person scenes**: even a two-person scene creates combinatorial ambiguity (P^V possible assignment combinations) if only 2D bounding boxes or poses are used.

For CVPR/ICRA 2027, treating cross-view matching as a first-class research problem rather than a data-loader detail is important because:

1. Real-world multi-view datasets (Shelf, Campus, CMU Panoptic, 3DPW multi-view) contain person-ID noise, occlusion, and partial views.
2. Current learned fusion methods (`attention`, `attention_v2`, `robust_triangulation`) assume the input tuples `(points_2d[t, v, j], confidences[t, v, j])` are already aligned across `v`. A matching error is an immediate failure mode that no fusion network can recover from.
3. SMPL-based pipelines (GVHMR → HumanMotionIR) project a single person into each view; to turn per-view single-person results into a multi-person scene, a principled matcher is needed.

## 2. Key related work

1. **Dong et al., "Fast and Robust Multi-Person 3D Pose Estimation and Tracking from Multiple Views", T-PAMI 2021 (MVPose).**
   - Geometric matching based on epipolar constraints and 3D consistency to associate 2D poses across views.
   - Separates cross-view association from 3D lifting and uses bipartite matching.

2. **Habibian et al., “3D Human Pose Estimation from a Single Image through Cross-View Consistency”, CVPR 2022.**
   - Uses reprojection/epipolar consistency to score cross-view correspondences without 3D GT.
   - Useful for weakly-supervised matching objectives.

3. **Zhang et al., “VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment”, ECCV 2020.**
   - Aggregates 2D heatmaps into a 3D voxel volume; the voxel grid implicitly resolves cross-view correspondence.
   - Relevant as a baseline but expensive in memory.

4. **Liu et al., “Cross-View Tracking for Multi-Person 3D Pose Estimation: A New Benchmark”, CVPR 2023.**
   - Introduces explicit cross-view tracking and identity linking with metrics (IDF1, MOTA).
   - Directly on-topic for publication-quality framing.

5. **Iskandar et al., “MV-P3D: Multi-View 3D Human Pose and Shape Estimation from Sparse Views”, CVPR 2023 / 2024.**
   - Transformer consuming unordered per-view detections to output a consistent 3D shape.
   - Implicit matching via attention over view tokens; natural extension of `ViewAttentionFusion`.

## 3. Relation to the current motionflow-multiview codebase

### 3.1 Where the problem appears

- `motionflow_mv/ir/multiview_adapter.py` currently assumes `irs` are already aligned: it stacks `points_2d_list` and `confidence_list` per view directly.
- `experiments/eval_all_plugins_shelf.py` uses `select_best_person_group` to choose one person group per frame, but Shelf already provides pre-grouped detections.
- `experiments/demo_gvhmr_multiview_projection.py` projects a *single* GVHMR result into all views, sidestepping multi-person matching entirely.
- The plugin interface in `motionflow_mv/fusion/fusion_module.py` accepts `(points_2d: T×V×J×2, confidences: T×V×J)` and therefore requires the caller to have solved the correspondence problem.

### 3.2 What is already in place

- `Camera` (`motionflow_mv/calibration/camera.py`) exposes `projection_matrix`, enabling epipolar geometry for matching.
- `DLTFusion` and `RobustTriangulationModel` are geometrically grounded; once correspondence is correct, they produce strong 3D poses.
- `ViewAttentionFusion`/`AttentionFusionModel` provide a learned per-joint attention mechanism over views, which could be extended to reason over *detections* rather than *views*.
- `HumanMotionIR` carries per-view `per_view_2d` and `per_view_confidence`, plus `person_id`, making it the natural place to store cross-view identity links.

### 3.3 What is missing

- No module takes `detections_per_view: List[List[Detection]]` and outputs a coherent `person_id`-aligned set of 2D observations.
- No cost function that combines 2D appearance similarity, epipolar distance, and 3D triangulation consistency.
- No evaluation metric for matching quality (IDF1, MOTA, or a per-frame person-consistency score).
- The current plugins are evaluated on Shelf after manual grouping; cross-view matching is therefore an untested gap between the data loader and the fusion plugins.

## 4. Concrete recommendations

### 4.1 Short-term: explicit geometry-and-appearance matcher

Implement a standalone `CrossViewMatcher` with the following pipeline:

1. **Input**: per-view per-person 2D keypoints and bounding boxes, plus calibrated cameras.
2. **Pairwise cost** between detection `a` in view `i` and detection `b` in view `j`:
   - **Epipolar term**: average symmetric epipolar distance between matched joints.
   - **Appearance term**: cosine distance between per-person appearance embeddings (e.g. from a small Re-ID network or simple color histogram).
   - **Pose-shape term**: Procrustes-like 2D pose similarity after rectification.
3. **Global assignment**: Hungarian algorithm per frame, then track identities over time with a simple Kalman filter or online Hungarian tracker.
4. **Output**: aligned `(T, V, J, 2)` arrays and a `person_id` map, feeding directly into `fuse_multiple_irs`.

This can be validated on Shelf/Campus without any network training, using the existing per-frame 2D detections.

### 4.2 Medium-term: learnable cross-view attention over detections

Extend `AttentionFusionModelV2` so that each *detection* is a token rather than each *view*:

- Input tokens: `(x, y, confidence, view_id_embed, detection_id_embed)` for every detection.
- Self-attention reasons over all detections from all views, allowing implicit many-to-many matching and occlusion reasoning.
- A matching head predicts a soft assignment matrix `detection → person_id`, and the fusion head uses the assignment to pool per-person 3D joints.

This mirrors MV-P3D and METRO-style transformers and could be trained end-to-end on synthetic data with known person IDs.

### 4.3 Training and data recommendations

- **Synthetic pre-training**: augment the existing synthetic generator to produce multi-person scenes with occlusion and person overlap. Include explicit matching labels (person_id per detection) and 3D GT.
- **Shelf fine-tuning**: use Shelf 2D detections and the provided 3D pseudo-GT from DLT to validate matching accuracy, not just 3D error.
- **Metrics**: report
  - 3D pose metrics: MPJPE, PA-MPJPE, Procrustes MPJPE.
  - Matching metrics: MOTA, IDF1, and a new **per-frame person-consistency rate** (fraction of frames where all views map to the same person).
- **Ablation study**: compare the proposed matcher against the current pre-grouped baseline on the same fusion plugins. If the matcher is perfect, plugin errors should match the pre-grouped numbers; the gap measures matching loss.

## 5. Open questions / risks

| Risk / Question | Impact | Mitigation |
|----------------|--------|-----------|
| Shelf/Campus provide pre-grouped detections; real-world footage may not. | The current strong DLT numbers could be hiding matching failures. | Run an end-to-end experiment with raw detections to quantify the gap. |
| Learned matching requires multi-person labels, which are scarce. | Training data bottleneck. | Use synthetic multi-person data with occlusion and rely on weakly-supervised epipolar losses. |
| `ViewAttentionFusion` is currently per-joint over views, not over detections. | Extending to variable detections requires redesigning the input tokenization and output heads. | Prototype with a simple transformer head before committing to a full architecture change. |
| Person re-identification across views is hard under strong viewpoint change. | Appearance features may be unreliable. | Weight epipolar/pose terms higher and keep appearance as a soft prior. |
| The plugin API assumes fixed V and no missing views. | A matcher that outputs missing detections needs a mask/placeholder convention. | Standardize a `visibility_mask: (T, V, J)` field in `HumanMotionIR`. |
| Temporal tracking vs. per-frame matching trade-off. | Online applications need low latency; batch trackers may be too slow. | Start with per-frame Hungarian + cheap temporal smoothing, then graduate to online trackers. |

## 6. Bottom line

Cross-view matching is the **missing upstream module** in `motionflow-multiview`. The fusion plugins (DLT, attention, robust triangulation) are geometrically strong, but they assume perfectly aligned per-view observations. A publication-quality next step is to make matching explicit: implement a geometry-and-appearance matcher, evaluate it with multi-person synthetic data and real Shelf/Campus raw detections, and optionally fold it into a learned transformer that unifies matching and fusion. This directly addresses a real research gap and differentiates the project from prior work that either assumes pre-aligned views or buries matching inside a monolithic volume network.