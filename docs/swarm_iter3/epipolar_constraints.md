I’ve researched the topic against the local codebase and relevant literature. The report is below; it should be saved as `docs/swarm_iter3/epipolar_constraints.md`.

---

# Epipolar Constraints for Multi-View Human Motion Fusion

## 1. Problem statement

In calibrated multi-view human pose estimation, the same 3D joint is observed from several synchronized cameras. For a pinhole rig, the projection of a 3D point `X` in view `i` is

```
x_i = P_i X / (P_i X)_3
```

where `P_i ∈ ℝ^{3×4}` is the camera projection matrix. Geometrically, the back-projected ray from `x_i` must pass through `X`; for any other view `j`, the corresponding image point `x_j` must lie on the epipolar line induced by `x_i`. This is the *epipolar constraint*:

```
x_j^T F_{ij} x_i = 0
```

with `F_{ij}` the fundamental matrix between views `i` and `j`.

Why this matters for MotionFlow:

* **Correspondence quality.** The current pipeline triangulates per-view 2D keypoints under the assumption that all detections belong to the same person and the same joint. When a detector is occluded, noisy, or mis-detects, the epipolar constraint is violated before any 3D reconstruction is formed.
* **Generalization.** Learned fusion models trained only with 3D MSE or reprojection loss have no explicit inductive bias that rays must intersect. This partly explains the large cross-dataset gap seen in `docs/design_v3.md`: attention fusion drops from ~80 px on Shelf to ~318 px/110 px on Campus.
* **Outlier rejection.** DLT and its weighted variants treat every view as equally plausible (modulo confidence). An epipolar consistency check can reject or down-weight views whose detections do not lie on the corresponding epipolar lines.
* **SMPL fitting.** Per-view SMPL estimators such as GVHMR/ScoreHMR produce independent camera-relative results. Enforcing that projected SMPL joints satisfy epipolar constraints across views is a principled way to refine `global_orient`, `transl`, and `betas`.

The research question for the next iteration is therefore: **how should epipolar geometry be encoded—through losses, attention masks, or explicit triangulation—so that learned fusion becomes both more accurate and more generalizable?**

## 2. Key related work / methods

### 2.1 Geometric triangulation and epipolar geometry

The Direct Linear Transform (DLT) builds a linear system from `N ≥ 2` projection equations and solves for the 3D point in the least-squares sense. Confidence weighting, as in Hartley & Zisserman and implemented in `motionflow_mv/fusion/triangulation.py`, is the standard baseline. However, DLT does not explicitly penalize deviation from the epipolar constraint; it only minimizes algebraic error. A statistically cleaner formulation minimizes the *Sampson distance* or symmetric epipolar distance between corresponding points, which is the basis of most robust multi-view geometry estimators.

### 2.2 Learnable triangulation

Iskakov et al. (*Learnable Triangulation of Human Pose*, ICCV 2019) showed that a neural network can predict per-view weights that are then fed into a differentiable DLT layer. The key insight is to make the triangulation operation itself differentiable, so the model learns to discount occluded or noisy views. In MotionFlow, the `RobustTriangulationModel` (`motionflow_mv/fusion/robust_triangulation.py`) follows this pattern. The original work does not add an explicit epipolar term, but its differentiable triangulation can be augmented with one.

### 2.3 Epipolar transformers

He et al. (*Epipolar Transformer for Multi-view Human Pose Estimation*, CVPRW 2020) proposed an attention mechanism restricted by epipolar geometry: features are gathered along the epipolar line in a target view rather than from the whole image. This reduces the search space for cross-view correspondence and naturally handles occlusion. For a sparse keypoint pipeline like MotionFlow, the analogous idea is to restrict view-to-view attention to pairs whose 2D detections are epipolar-consistent.

### 2.4 Cross-view fusion transformers

Ma et al. (*TransFusion: Cross-view Fusion with Transformer for 3D Human Pose Estimation*, BMVC 2021) introduced an “epipolar field” to encode 3D positional information into the transformer. They fuse 2D predictions across views before triangulation, using epipolar geometry as a positional cue. Their work suggests that camera-aware attention can outperform late triangulation, but only when geometry is used to guide the attention map. This directly informs the unstable `AttentionFusionModelV2` in the codebase, which currently embeds flattened projection matrices but does not compute epipolar-line features.

### 2.5 Geometry-aware state-space and transformer fusions

Liao et al. (*Multiple View Geometry Transformers for 3D Human Pose Estimation*, CVPR 2024, arXiv:2311.10983) combined transformers with explicit multi-view geometry, using attention over epipolar correspondences to improve 3D pose estimation. More recently, MV-SSM (*Multi-View State Space Modeling for 3D Human Pose Estimation*, CVPR 2025) uses geometry-guided scanning. For MotionFlow, the take-away is that the fusion network should consume **rays** (camera center + direction) rather than raw 2D coordinates or flattened projection matrices; this makes the representation camera-invariant and epipolar-aware.

### 2.6 Multi-person robust estimation

Dong et al. (*Fast and Robust Multi-Person 3D Pose Estimation and Tracking from Multiple Views*, IEEE T-PAMI 2021) use epipolar constraints to match detections across views before triangulation. Their framework builds an epipolar graph that resolves ambiguous person-to-person correspondences. MotionFlow is currently single-person, but any move toward multi-person scenes will require a similar matching step.

## 3. Relation to the current MotionFlow-multiview codebase

The current repository already contains the necessary hooks for epipolar geometry, but they are not used as an explicit constraint.

| Component | Current behavior | Epipolar gap |
|---|---|---|
| `motionflow_mv/fusion/triangulation.py` | Confidence-weighted DLT. | No explicit epipolar error; treats each view independently. |
| `motionflow_mv/fusion/robust_triangulation.py` | Learns per-view weights, then solves DLT. | No epipolar loss; weights are learned from 3D/reprojection targets. |
| `motionflow_mv/fusion/attention_model_v2.py` | Embeds flattened `P` matrices alongside 2D points. | Does not compute epipolar lines or ray directions; noted as “unstable” in `__init__.py`. |
| `motionflow_mv/fusion/fusion_module.py` | Plugin contract with `fuse(points_2d, confidences, cameras)`. | No epipolar residual exposed in the contract. |
| `motionflow_mv/ir/multiview_adapter.py` | Fuses per-view IRs and writes `uncertainty`. | `uncertainty` has no epipolar consistency field. |
| `experiments/train_attention_fusion_shelf_v2.py` | Trains with MSE loss only. | No geometry-aware regularization. |
| `docs/design_v3.md` | Reports DLT ~10 px, learned attention ~80 px after fine-tuning, but poor cross-dataset transfer. | Suggests geometry-aware attention is needed. |

Key observations from the code:

1. **Projection matrices are available.** `Camera.projection_matrix` (`motionflow_mv/calibration/camera.py`) provides everything needed to compute epipolar lines and fundamental matrices.
2. **Training ignores geometry.** `train_attention_fusion_shelf_v2.py` minimizes only `MSELoss(pred, target_3d)`. A reprojection or epipolar term would add geometric consistency without requiring 3D GT at test time.
3. **`attention_v2` is disabled.** `motionflow_mv/fusion/__init__.py` comments out `register_attention_v2_fusion_module()` because the geometry-aware variant is unstable. This is the most natural place to introduce epipolar guidance.
4. **No correspondence matching.** `select_best_person_group` in `experiments/eval_all_plugins_shelf.py` is single-person and assumes views are aligned; epipolar matching would be required for multi-person extension.

## 4. Concrete recommendations

### 4.1 Add an epipolar loss to the learned fusion training

Modify `train_attention_fusion_shelf_v2.py` (and the synthetic trainers) to include an geometry term:

```
L = L_3d + λ_reproj * L_reproj + λ_epi * L_epi
```

For each pair of views `(i, j)` and each joint:

```
L_epi = dist_epi(x_j, F_{ij}, x_i)^2
```

where `dist_epi` is the (symmetric) point-to-epipolar-line distance. This can be computed cheaply from the camera projection matrices already loaded in the dataloader. It requires no extra ground truth and directly regularizes the network to respect multi-view geometry.

### 4.2 Replace flattened projection-matrix embeddings with ray embeddings

In `attention_model_v2.py`, instead of embedding the 12 entries of `P`, encode each 2D keypoint as a 3D ray:

```
o_i = camera_center_i
d_i = R_i^T K_i^{-1} [u_i, v_i, 1]^T
```

The attention module can then compute ray–ray proximity (or angle) and epipolar-line distances, which are invariant to camera resolution and distance units. This should stabilize the geometry-aware attention model and let it generalize across datasets with different scales (mm vs. meters).

### 4.3 Implement an epipolar outlier/rejection pre-step

Before DLT/attention fusion, test each 2D detection against the epipolar lines from the other views. A detection that is far from its expected epipolar lines (e.g., Sampson distance above a threshold) should have its confidence down-weighted or be marked as occluded. This can be implemented as a small utility in `motionflow_mv/fusion/` and reused by `DLTFusion`, `RobustTriangulationFusion`, and `AttentionFusionV2Module`.

### 4.4 Add an epipolar residual to `HumanMotionIR.uncertainty`

Extend `motionflow_mv/ir/multiview_adapter.py` to populate:

```python
uncertainty["epipolar_residual"] = (T, J)  # mean point-to-epipolar-line distance
```

This gives downstream robot retargeting a per-joint geometric consistency score and helps identify frames where fusion is unreliable.

### 4.5 Use epipolar constraints for SMPL parameter refinement

The GVHMR/ScoreHMR adapters produce per-view SMPL poses. Add an optional optimization step in the multi-view adapter that minimizes:

```
E(θ, β, T) = Σ_i Σ_j ρ( x_i^obs, π_i(X_j(θ, β, T)) ) + λ_epi * Σ_i Σ_{j≠i} dist_epi(x_i^obs, F_{ij}, x_j^obs)
```

where `X_j(·)` are SMPL joint positions. This is a small bundle-adjustment-style refinement that uses epipolar geometry to resolve scale and global translation ambiguities.

### 4.6 Dataset and evaluation plan

* **Validation:** Continue using Shelf (frames 300–600) and Campus for fast reprojection sanity. Report both reprojection error *and* the new epipolar residual.
* **3D-supervised training:** Register Human3.6M and train with the epipolar loss. Use the standard S9/S11 test split.
* **Cross-dataset ablation:** Train on Shelf, validate on Campus, exactly as in `docs/design_v3.md`. The epipolar loss should reduce the cross-dataset gap because it is scale- and dataset-invariant.
* **Synthetic pre-training:** Use AMASS-generated sequences with virtual rigs. Deliberately inject occlusion and noise to measure epipolar outlier rejection.

## 5. Open questions / risks

1. **Does the epipolar loss add value beyond reprojection?** Reprojection error already encodes multi-view consistency in a projective sense. The ablation must show that `L_epi` improves cross-dataset or occlusion performance, not just training stability.
2. **Numerical stability with degenerate views.** The fundamental matrix becomes ill-defined when cameras share the same optical center or have parallel optical axes. The Shelf/Campus rigs are well-separated, but synthetic rigs may not be.
3. **Scale/unit handling.** Epipolar geometry is projective and scale-invariant, but DLT outputs and SMPL parameters are metric. The existing per-plugin scaling in `eval_all_plugins_shelf.py` must be preserved.
4. **Distortion.** Current `Camera` assumes a pinhole model. Real-world wide-angle or fisheye cameras will need undistortion before epipolar constraints hold.
5. **Multi-person combinatorics.** Epipolar matching across many people and views is NP-hard in general; the current single-person pipeline avoids this, but any extension must add Hungarian or graph-based matching.
6. **Attention model stability.** `AttentionFusionModelV2` is already unstable. Adding epipolar features may make training harder before it makes it better; start with the loss-only variant, then move to attention masking.

## 6. References

1. Hartley, R. & Zisserman, A., *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2003.
2. He, Y., Yan, R., Fragkiadaki, K., & Yu, S.-I., “Epipolar Transformer for Multi-view Human Pose Estimation,” *CVPR Workshops*, 2020. DOI: [10.1109/cvprw50498.2020.00526](https://doi.org/10.1109/cvprw50498.2020.00526)
3. Iskakov, K., Lempitsky, V., & Malkov, Y., “Learnable Triangulation of Human Pose,” *ICCV*, 2019. DOI: [10.1109/iccv.2019.00781](https://doi.org/10.1109/iccv.2019.00781) — arXiv: [1905.05754](https://arxiv.org/abs/1905.05754)
4. Ma, H., et al., “TransFusion: Cross-view Fusion with Transformer for 3D Human Pose Estimation,” *BMVC*, 2021. arXiv: [2110.09554](https://arxiv.org/abs/2110.09554)
5. Dong, J., et al., “Fast and Robust Multi-Person 3D Pose Estimation and Tracking from Multiple Views,” *IEEE T-PAMI*, 2021. DOI: [10.1109/tpami.2021.3098052](https://doi.org/10.1109/tpami.2021.3098052)
6. Liao, Z., Zhu, J., Wang, C., & Hu, H., “Multiple View Geometry Transformers for 3D Human Pose Estimation,” *CVPR*, 2024. arXiv: [2311.10983](https://arxiv.org/abs/2311.10983)

---

**Summary:** The MotionFlow-multiview stack already has camera calibration, DLT triangulation, and a geometry-aware attention plugin, but it does not yet use epipolar constraints explicitly. The highest-impact next step is to add an epipolar loss to the learned fusion training, encode 2D keypoints as 3D rays rather than flattened projection matrices, and expose an epipolar residual in the IR uncertainty. These changes should improve cross-dataset generalization and provide a strong angle for a CVPR/ICRA 2027 paper.