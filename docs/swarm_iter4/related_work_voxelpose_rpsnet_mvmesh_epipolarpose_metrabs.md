# Related Work Review: VoxelPose, RPSNet, MvMESH, EpipolarPose, MeTRAbs

**Target venues:** CVPR / ICRA 2027

---

## 1. Short survey

**VoxelPose** (Tu et al., ECCV 2020) avoids 2D-to-3D lifting by operating directly in 3D space. A Cuboid Proposal Network localizes people in a common voxel grid and a Pose Regression Network refines joints. It is robust to occlusion but heavy and tied to a fixed calibrated rig.

**EpipolarPose** (Kocabas et al., CVPR 2019) trains a 3D pose estimator without 3D ground truth: multi-view 2D keypoints are triangulated via epipolar geometry to generate pseudo-3D labels. It shows that geometry can substitute for scarce annotations, but accuracy is bounded by the 2D detector and triangulation quality.

**MeTRAbs** (Sárándi et al., IEEE T-BIOM 2021) predicts absolute 3D joints in metric space from a single image using metric-scale volumetric heatmaps. It removes scale ambiguity and handles truncation, but it does not exploit calibrated multi-view cues.

**RPSNet / MvMESH** represent two extension directions beyond skeleton-only, single-frame fusion: recurrent temporal pose modeling and multi-view parametric mesh (e.g., SMPL) recovery. These are increasingly relevant for downstream robotics, but canonical public citations for methods with these exact names are sparse.

---

## 2. Relation to the current MotionFlow-multiview stack

The current `RayAttentionFusionModel` (`motionflow_mv/fusion/ray_attention_model.py`) embeds camera rays and centers, predicts per-view weights, and triangulates with a differentiable weighted DLT. `shelf_loader.py` supplies 2D keypoints, confidences, and calibrated cameras, and `train_ray_attention_real.py` fine-tunes on Shelf/Campus.

- **VoxelPose** motivates volumetric confidence aggregation, but our pipeline uses sparse 2D keypoints; a lightweight analogue is to treat per-view confidences as ray-based attention weights.
- **EpipolarPose** suggests an explicit epipolar loss can reduce reliance on 3D GT; the current trainer uses only 3D MSE.
- **MeTRAbs** stresses metric-scale outputs; the DLT layer is metric, but the learned attention head can drift without strict unit handling. The design_v3 plugin contract (`input_scale`/`output_scale`) is the right hook.
- **RPSNet / MvMESH** point to temporal refinement and SMPL/mesh output, which the existing GVHMR-to-`HumanMotionIR` converter can seed.

---

## 3. Concrete recommendations

1. **Add an epipolar loss to `train_ray_attention_real.py`.** Compute the symmetric point-to-epipolar-line distance per joint across view pairs using `Camera.projection_matrix` and add `L = L_3D_MSE + λ_epi * L_epi`. This follows EpipolarPose, reduces dependence on scarce 3D GT, and should improve cross-dataset transfer.

2. **Harden metric scale via bone-length and unit contracts.** Enforce the design_v3 `input_scale`/`output_scale` contract in all trainers and add a bone-length consistency term inspired by MeTRAbs, preventing confusion between millimeter-scale Shelf and meter-scale synthetic/GVHMR data.

3. **Prototype a VoxelPose-style volumetric confidence plugin.** For dense heatmap inputs, lift per-view 2D confidences into a shared 3D grid using calibration and extract joint locations. Compare it with the sparse `ray_attention` plugin to quantify the cost/benefit on Shelf/Campus.

4. **Build a temporal `ray_attention` variant (RPSNet-style).** Stack `RayAttentionFusionModel` outputs over 5–7 frames and add a lightweight temporal transformer or Bi-GRU to smooth real-data jitter. Register it as `temporal_ray_attention` without changing the existing interface.

5. **Extend to SMPL multi-view mesh output (MvMESH-style).** Use the GVHMR adapter and `demo_gvhmr_multiview_projection.py` to fuse per-view SMPL parameters into a single metric `HumanMotionIR` mesh. Start with shape-parameter averaging and a small multi-view reprojection refinement of pose/trans.

---

## 4. Potential risks

- **3D GT scarcity.** Shelf/Campus are small; epipolar losses help but do not fully replace Human3.6M-scale 3D supervision.
- **Scale/unit mismatch.** Epipolar geometry is projective and scale-free, while DLT and SMPL outputs are metric. Keep the per-plugin scaling contract and `length_unit` metadata.
- **Computational cost.** Volumetric and temporal plugins are heavier than sparse DLT; `ray_attention` should remain the default fast path.
- **Citation uncertainty.** “RPSNet” and “MvMESH” are not clearly established multi-view pose names in public indices. Verify exact titles/venues before including them in a camera-ready paper, or replace with canonical references.

---

## 5. Fit with the paper plan

For ICRA/CVPR 2027, this survey frames the contribution as bridging heavy volumetric fusion (VoxelPose), self-supervised geometry (EpipolarPose), monocular metric pose (MeTRAbs), and sequence/mesh extensions (RPSNet/MvMESH). The related work can argue that prior work needs dense 3D volumes, large 3D annotations, or monocular assumptions, whereas our `RayAttentionFusionModel` is a calibrated, plugin-based 2D-keypoint fusion that learns ray-based attention and triangulates through a differentiable DLT layer—geometry-aware, modular, and metric, with a clear path to temporal and SMPL extensions.

---

## References

1. Tu et al., *VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment*, ECCV 2020. [arXiv:2004.06239](https://arxiv.org/abs/2004.06239)
2. Kocabas et al., *Self-Supervised Learning of 3D Human Pose using Multi-view Geometry*, CVPR 2019. [arXiv:1903.02330](https://arxiv.org/abs/1903.02330)
3. Sárándi et al., *MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation*, IEEE T-BIOM 2021. [arXiv:2007.07227](https://arxiv.org/abs/2007.07227)
