# ICRA 2027 Positioning, Contribution, and Experimental Plan

## 1. Brief Survey

Calibrated multi-view human pose estimation typically relies on two families of methods: classical geometric triangulation (e.g., DLT) and learned attention-based fusion.  DLT is geometrically principled and metrically accurate when views are clean, but it has no mechanism to handle occlusion, noisy detections, or calibration uncertainty.  Pure attention fusion can learn to down-weight corrupted views, yet it regresses 3D coordinates directly and therefore lacks a strong geometric inductive bias; as the project’s v3 design notes show, a Shelf-trained attention plugin still reports ~80 px reprojection error while DLT stays near 10 px, and the attention model fails to generalize across datasets.

The recently implemented `RayAttentionFusionModel` (`motionflow_mv/fusion/ray_attention_model.py`) bridges this gap: it embeds 2D keypoints together with camera centers and ray directions, uses multi-head self-attention to predict per-view weights, and feeds those weights into a differentiable weighted DLT layer.  Synthetic validation already shows a drop from 3.68 m MPJPE (pure attention) to 0.0021 m MPJPE for the new model, confirming that the geometric triangulation layer enforces the correct metric structure while the attention head learns robust view weighting.

## 2. Positioning for ICRA 2027

ICRA values methods that are (a) grounded in real robotic perception problems, (b) geometrically faithful, and (c) empirically validated on public benchmarks.  The ray-aware fusion module fits this profile: it is a *calibrated* multi-view fusion layer that can be dropped into a human-motion perception pipeline for human-robot interaction, teleoperation, or motion prediction.  Unlike end-to-end regressors, it produces interpretable per-view weights and metric 3D poses, which robotics downstream tasks can consume directly.

### Proposed Core Contributions

1. **Geometry-aware neural fusion**: a plugin that combines ray embeddings, self-attention, and differentiable weighted DLT.
2. **Interpretable robustness**: per-view, per-joint attention weights that explain which cameras are trusted under occlusion or outliers.
3. **Metric multi-view evaluation**: empirical validation on Shelf, Campus, and synthetic rigs, with ablations showing when learning improves upon DLT.

## 3. Actionable Recommendations

1. **Lock the experimental protocol now.**  Use `experiments/train_ray_attention_real.py` to train on Shelf and Campus, and report both MPJPE and reprojection error.  Mirror the existing per-plugin evaluation in `experiments/eval_all_plugins_shelf.py` so results are directly comparable to the DLT/temporal-refiner baselines in `docs/design_v3.md`.

2. **Add a reprojection / epipolar auxiliary loss.**  The current trainer uses only 3D MSE.  Adding a reprojection loss will make the model robust to the inevitable scale mismatches (mm vs. m) and camera-transfer scenarios noted in the v3 risk register, and will help beat the DLT baseline on real data.

3. **Run controlled real-data ablations.**  The synthetic ablations are promising but insufficient.  Train and evaluate: (a) pure DLT, (b) ray-attention with frozen DLT weights, (c) full ray-attention, (d) pure attention, all on the same Shelf/Campus split.  This directly answers ICRA reviewers’ “why not just DLT?” question.

4. **Obtain and integrate Human3.6M.**  `docs/design_v3.md` identifies 3D-GT training as the only path to clearly beat DLT.  Add `motionflow_mv/data/human36m_loader.py`, align its joint schema with Shelf/Campus, and use it either for pre-training or as an additional benchmark.

5. **Draft the robotics narrative early.**  Frame the paper around “reliable multi-view human pose for robot perception,” not just pose-estimation accuracy.  Tie the differentiable triangulation layer to downstream use cases (human motion prediction, policy preview, HMR-to-IR conversion) already present in the project.

## 4. Potential Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Data access** — Shelf/Campus/H36M not in workspace; A800-D is read-only. | Blocks real-data training. | Use the Google Drive mirror noted in `docs/swarm_iter2/shelf_campus_source.md`; mirror data locally before experiments. |
| **Generalization across calibrations** — learned weights may overfit to one rig. | Weak cross-dataset results hurt ICRA credibility. | Scale-aware augmentation, reprojection loss, and evaluation on Campus after Shelf training. |
| **DLT is already strong** — learning may not improve enough to justify a neural method. | Weak contribution story. | Focus experiments on occlusion/outlier scenarios where DLT has no mechanism to recover. |
| **Timeline** — ICRA 2027 deadline is likely September 2026. | Risk of incomplete experiments. | Finalize data loader and baseline protocol within this iteration; reserve time for ablations and writing. |

## 5. Fit into the Paper Plan

The ray-aware attention fusion module is positioned as the **core technical contribution** of the multi-view extension.  It replaces the unstable `attention_v2` flattening approach identified by the swarm and is the method around which the experimental section should be organized.  The plugin architecture (`dlt`, `attention`, `robust_triangulation`, `residual_refiner`, `temporal_refiner`) provides the necessary baselines and makes the work reproducible.  If real-data training succeeds, this plugin becomes the headline result; if not, the controlled synthetic ablations and the clear geometric design still provide a publishable contribution, albeit a narrower one.
