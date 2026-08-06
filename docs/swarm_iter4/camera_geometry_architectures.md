# Camera-Geometry-Aware Architectures Beyond Attention

## Brief Survey

The current `RayAttentionFusionModel` (`motionflow_mv/fusion/ray_attention_model.py`) already moves beyond raw keypoint regression: it embeds per-view observations `(x, y, conf)` together with ray features `(camera_center, ray_dir)` and uses multi-head self-attention across views to predict per-view weights for a differentiable weighted DLT triangulator.  On synthetic data this beats the older `attention` plugin by orders of magnitude, confirming that feeding geometry into the network is the right direction.  However, the architecture still treats each joint and each view as an unordered set of tokens and only models interactions through a vanilla attention layer over views.  The topic asks what lies *beyond* this: structured camera-graph representations, full transformer blocks, and richer ray embeddings.

Recent multi-view pose literature points to three complementary directions:

1. **Camera-graph neural networks.**  Instead of pooling attention across an implicit complete graph, explicitly model views/cameras as nodes in a graph and propagate messages along calibrated epipolar or ray-intersection edges.  Work such as MVPose and MPSNet uses graph attention or message passing to aggregate multi-view evidence while respecting camera topology.
2. **Transformers with camera positional embeddings.**  A standard transformer encoder over `(view, joint)` tokens can be made camera-aware by injecting positional embeddings derived from extrinsics/intrinsics, analogous to NeRF-style positional encodings or the learned camera embeddings in multi-view 3D human pose.
3. **Rich ray embeddings.**  Plücker coordinates `(o, d)` or their variants encode a ray in a more geometrically meaningful way than concatenated centers and directions.  Ray embeddings can be fused with 2D observations before attention, providing a stronger inductive bias.

A camera-geometry-aware architecture should therefore exploit the calibrated rig structure explicitly rather than hoping attention discovers it implicitly.

## Concrete Recommendations

1. **Add camera-conditioned positional embeddings to the existing attention layer.**  Compute a per-view embedding from `R_v`, `t_v`, and the focal length/principal point (e.g., flatten the rotation or use a small MLP on `c_v` and `K_v` parameters) and add it to the per-joint tokens before self-attention.  This is a minimal change in `ray_attention_model.py` and should improve cross-dataset generalization because the network is no longer camera-agnostic.

2. **Implement a graph-attention fusion plugin `graph_ray_attention`.**  Build a bipartite graph where nodes are views and joints, and edges connect a joint in a view to the same joint in another view with an edge feature based on the angle between their rays or the epipolar line.  Run a small Graph Attention Network (GAT) or message-passing neural network (MPNN) and read off per-view weights.  This directly encodes the calibrated camera topology and can generalize to variable numbers of views.

3. **Replace the `(center, ray_dir)` concatenation with Plücker ray coordinates.**  A 3D ray can be represented by Plücker coordinates `(m = o × d, d)` or simply the normalized `(o, d)` pair projected through a learned ray encoder.  A dedicated `RayEmbed` module would make the geometry explicit and simplify downstream extensions such as ray-ray distance losses.

4. **Add skeleton-aware cross-joint attention.**  The current model attends only across views for a single joint.  A second transformer layer over joints within each view (or a full `(joint, view)` transformer) lets the model share information across anatomically connected joints, reducing the impact of single-view joint detection failures before triangulation.

5. **Introduce an epipolar consistency loss during training.**  When 3D GT is unavailable, use pseudo-GT from the DLT baseline but also regularize the predicted per-view weights with an epipolar loss: for a pair of views, the predicted 3D point should project close to the epipolar line of the other view.  This loss is cheap to compute and reinforces the camera-geometry prior without requiring labeled 3D data.

## Potential Risks

- **Training instability.**  The `attention_v2` attempt was abandoned precisely because flattening projection matrices caused instability.  Any new geometry-aware module must be carefully normalized (standardize ray origins to unit scale, use layer norm, avoid regressing 3D coordinates directly) and validated on synthetic data first.
- **Over-engineering vs. the DLT baseline.**  The plain DLT baseline already achieves single-pixel reprojection on Shelf.  A richer architecture must show a measurable improvement (e.g., robustness to occlusion/outliers or cross-dataset transfer) rather than margin.
- **Data hunger.**  Transformers and GNNs require more training data than the small Shelf/Campus sequences.  The synthetic generator should be scaled up, and Human3.6M should be integrated, before these architectures are expected to beat the geometric baselines.
- **Inference cost.**  Adding graph or cross-joint attention increases latency.  For a real-time pipeline, keep the final model lightweight or make the heavy architecture an optional refinement stage.

## Fit into the Paper Plan

This work package sits between the existing `ray_attention` proof-of-concept and a mature ICRA/CVPR 2027 submission.  It provides the "geometry-aware architecture" contribution: going beyond simple attention to structured camera-graph and ray-embedding designs.  The ablation matrix in the paper would compare (i) vanilla DLT, (ii) current `ray_attention`, (iii) graph-attention fusion, and (iv) the full camera-geometry transformer.  Success is measured by MPJPE on Shelf/Campus, zero-shot transfer between Shelf and Campus, and controlled robustness to occlusion/outliers using the synthetic benchmark already in place.  If the recommendations above are implemented incrementally, each produces a concrete experiment and an ablation row for the paper.
