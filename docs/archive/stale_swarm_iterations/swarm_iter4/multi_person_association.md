# Multi-Person Scenes and Person Association Across Calibrated Views

## 1. Topic survey

Most of the current MotionFlow-MV pipeline is built around a **single-person assumption**: `shelf_loader.py` extracts one `person_id` and `train_ray_attention_real.py` triangulates a single subject. Real multi-camera datasets such as Shelf and Campus, however, contain several interacting people, and the GVHMR-to-IR converter currently produces independent single-view sequences. For CVPR/ICRA 2027, the system must therefore solve two coupled problems:

1. **Detection & pose per view**: each camera yields 2D keypoint detections for zero, one, or multiple people.
2. **Cross-view association**: group detections that belong to the same physical person across views before/during triangulation.
3. **Consistent 3D reconstruction**: produce a stable set of 3D skeletons, ideally in world units, with correct identities over time.

The literature usually addresses this with **geometric matching + appearance/re-id cues**. Classic methods build a cost matrix from epipolar distances, 3D reprojection residuals, or bipartite matching across view pairs (e.g., VoxelPose, MvP, and follow-ups). More recent approaches add learned association: a graph neural network or transformer that scores compatibility between detections, or a multi-person Pictorial Structure model that reasons jointly over views and people. For calibrated rigs, a strong baseline is **triangulability**: candidate 2D detections from different views that truly correspond will triangulate to a 3D point with low reprojection error and physically plausible depth; non-matches will not.

The ray-aware attention fusion model (`RayAttentionFusionModel`) is currently per-person. Its geometric inductive bias—ray directions, camera centers, and weighted DLT—makes it an excellent building block, but the model has no notion of multiple people. Extending it to multi-person scenes is therefore a natural next step.

## 2. Current project context

- `motionflow_mv/data/shelf_loader.py` loads Shelf/Campus annotations for a single `person_id` and projects GT 3D joints to 2D.
- `motionflow_mv/fusion/ray_attention_model.py` implements `RayAttentionFusionModel`, which predicts per-view weights per joint and feeds a differentiable weighted DLT layer.
- `experiments/train_ray_attention_real.py` trains the model on a single extracted person from Shelf/Campus, comparing against a DLT baseline.

Thus, the multi-person gap is not in the fusion head itself but in **data loading, association, and downstream evaluation**.

## 3. Actionable recommendations

1. **Add a multi-person loader that returns detection sets, not a single person.**
   - Extend `shelf_loader.py` to read all `poses` per frame and emit `(T, M, V, J, 3)` arrays of 2D keypoints + confidences for `M` people, plus visibility masks.
   - Expose a flag `multi_person=True` to keep existing single-person experiments backward-compatible.
   - Also expose per-person 3D GT where available, so association accuracy can be evaluated.

2. **Implement a baseline association module using triangulation consistency.**
   - For each frame, take the set of 2D detections per view and enumerate candidate tuples across `V` views.
   - Use the existing `triangulate_dlt` to compute a 3D hypothesis for each tuple and score it by mean reprojection error.
   - Use **greedy or Hungarian assignment** to select the best `M` consistent tuples, constrained to one detection per person per view.
   - This gives a strong, interpretable baseline and a target for learned association to beat.

3. **Use ray-attention weights as an soft association signal.**
   - Train a lightweight per-joint **affinity head** that, given two views, predicts whether two detections belong to the same person from ray/appearance features.
   - Alternatively, feed multi-person detection tensors into a modified `RayAttentionFusionModel` where attention is computed jointly over people and views, producing per-person weights and 3D poses in one forward pass.
   - Either approach should be trained with a multi-person 3D loss + re-identification loss where identity labels exist.

4. **Leverage temporal and appearance consistency for robust tracking.**
   - Run a lightweight 3D tracker (e.g., Kalman filter or simple LSTM) on the triangulated skeletons to maintain identities across frames.
   - Add optional re-id embedding input (or use GVHMR appearance features) to resolve identity switches after occlusion.
   - Evaluate using CLEAR MOT-style metrics (MOTA, IDF1) in addition to pose accuracy.

5. **Build a controlled multi-person synthetic benchmark.**
   - Extend `experiments/generate_synthetic_multiview_dataset.py` to produce scenes with 2–5 SMPL bodies, overlapping fields of view, and realistic occlusion patterns.
   - This provides ground-truth identities and 3D poses, allowing the association module to be validated before real-data collection is complete.

## 4. Potential risks

- **Combinatorial explosion**: naive enumeration of detection tuples is exponential in the number of views and people. The baseline must use pruning (e.g., epipolar-only pairs, confidence thresholds) to remain practical.
- **Occlusion and identity switches**: when a person leaves the shared frustum or is heavily occluded, association can drift. Temporal tracking helps but adds latency and complexity.
- **Metric scale and calibration consistency**: `shelf_loader.py` converts from millimeters to meters; multi-person association must preserve the same unit convention across all people and views.
- **Lack of labeled multi-person 3D GT**: real datasets may not always provide per-person 3D annotations for all frames. Synthetic data and 2D-only reprojection losses will be essential for bootstrapping.

## 5. Fit into the paper plan

Multi-person association is a natural extension of the single-person ray-aware fusion story. It positions the paper beyond a single-person triangulation method and toward a **practical multi-view pose system**. The planned narrative arc is:

1. Single-person ray-aware fusion matches/exceeds DLT on controlled synthetic and real data (current state).
2. Multi-person association extends the method to realistic crowded scenes, validated on Shelf/Campus multi-person frames and a new synthetic multi-person benchmark.
3. The combined system is evaluated against DLT + Hungarian association and, where possible, against recent multi-view multi-person baselines.

By ICRA/CVPR 2027, this deliverable will provide both the **association module** and the **evaluation protocol** needed to demonstrate multi-person generalization.
