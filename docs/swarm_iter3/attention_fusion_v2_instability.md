# Instability of Geometry-Aware Attention Fusion (`AttentionFusionV2`)

## 1. Problem Statement

In MotionFlow Multi-View, `AttentionFusionModelV2` was introduced to make the learned fusion model *geometry-aware*: instead of only seeing per-view 2D keypoints and confidences, the network also receives the camera projection matrices. The expectation was that camera parameters would let the model reason about occlusions, view reliability, and multi-view geometry, and thereby outperform the camera-agnostic `AttentionFusionModel` (V1).

The actual results on the real Shelf dataset are the opposite:

| Method | Mean reproj. error (px) | Median reproj. error (px) |
|---|---|---|
| DLT (`dlt`) | 9.88 | 5.52 |
| Attention V1 (`attention`) | 80.42 | 58.90 |
| Attention V2 (`attention_v2`) | **184.29** | **173.01** |

*Table from `README.md` and `docs/design_v3.md`, Shelf frames 300–600, 5 views.*

Adding camera parameters made the learned fusion model **substantially worse** than the simpler V1 and orders of magnitude worse than the deterministic DLT baseline. This note documents why the current geometry-aware attention formulation is unstable, how it relates to the broader multi-view pose literature, and what should be tried next.

## 2. Key Related Work and Methods

**Direct 2D→3D regression is fragile.** Iskakov et al. (*Learnable Triangulation of Human Pose*, ICCV 2019) showed that regressing 3D coordinates directly from 2D observations performs poorly compared to methods that use geometry explicitly. Their solution predicts *per-view confidences* and then performs algebraic triangulation; the network does not regress the 3D point itself. This is a key lesson for `attention_v2`: the current model directly regresses `(B, J, 3)` world coordinates from a fused feature, bypassing the multi-view constraints that make DLT strong.

**Camera-aware transformers.** Moliner et al. (*Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction*, arXiv 2312.17106) and Liao et al. (*Multiple View Geometry Transformers for 3D Human Pose Estimation*, CVPR 2024, MVGFormer) inject camera geometry into transformers. They do not flatten the raw projection matrix and add it to 2D features; instead they use epipolar embeddings, ray directions, or learned camera tokens that attend to joint tokens. The current `attention_v2` implementation is far coarser: it flattens `P = K[R|t]` to 12 scalars, normalizes by 1000, and *adds* the camera embedding to the lifted 2D feature.

**Variable-view attention.** Shuai et al. (*Adaptive Multi-view and Temporal Fusing Transformer for 3D Human Pose Estimation*, arXiv 2110.05092, MFT) introduced relative attention to handle a variable number of views. The current `ViewAttentionFusion` uses a fixed per-joint query and softmax over views, which is related but lacks multi-head self-attention, positional view encoding, and the capacity to model complex view interactions.

**Recent neural optimizers.** Matsubara et al. (*HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery*, CVPR 2025) and Chharia et al. (*MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation*, CVPR 2025) show that strong results come from combining deep learned priors with explicit geometric projection. They are relevant long-term references, but the immediate priority is to fix the geometry representation before scaling model capacity.

## 3. Relation to the Current Codebase

The `attention_v2` plugin is implemented in three places:

- **`motionflow_mv/fusion/attention_model_v2.py`** defines `AttentionFusionModelV2`. The forward pass is:
  ```python
  x = self.lift(x)              # 2D + confidence -> d
  cam = self.cam_embed(cameras) # 12-dim flattened P -> d
  x = x + cam                   # additive fusion
  x = self.attention(x)
  return self.head(x)           # directly regress (B, J, 3)
  ```
  This fuses heterogeneous signals by element-wise addition and directly regresses 3D world coordinates without an explicit triangulation step.

- **`motionflow_mv/fusion/attention_fusion_v2_module.py`** feeds flattened projection matrices into the model. It normalizes neither the entries of `P` nor the camera coordinate frame beyond the `/1000` scaling done by the training scripts.

- **`experiments/train_attention_fusion_v2.py` / `train_attention_fusion_shelf_v2.py`** train with MSE (+ small MPJPE term) against DLT pseudo-GT. The training target is therefore the output of the geometric baseline the model is trying to beat. No reprojection, bone-length, or temporal losses are used.

Several concrete weaknesses stand out:

1. **Projection-matrix flattening is not geometry-preserving.** A flattened `P` mixes focal length, principal point, rotation, and translation in a single 12-dim vector. Entries have wildly different magnitudes (focal length ~1000, rotation ~1, translation ~1000) and different units. Even after `/1000`, the representation is not camera-invariant.

2. **Additive fusion of 2D and camera embeddings is weak.** Adding the camera embedding to the per-view 2D feature (`x = x + cam`) assumes the two embeddings live in the same semantic space. A cleaner design is cross-attention or concatenation, possibly followed by gating.

3. **No geometric inductive bias.** The model never forms camera rays, epipolar lines, or triangulation residuals. It is asked to *memorize* the mapping from `(2D, P)` to 3D, which is hard to generalize across camera rigs.

4. **DLT pseudo-GT ceiling.** Training against DLT means the best the model can do is copy DLT. It cannot learn to outperform DLT, and any instability in the camera branch can push it far below DLT.

5. **Shallow attention.** `ViewAttentionFusion` (`motionflow_mv/fusion/attention.py`) is a single scaled dot-product with a per-joint query. It has no layer norm, no residual, no multi-head self-attention, and no view positional encoding.

The empirical evidence matches these weaknesses: `attention_v2` not only fails to beat DLT, it performs worse than the camera-agnostic `attention_v1`, indicating that the camera branch is injecting harmful noise rather than useful geometric information.

## 4. Concrete Recommendations

### 4.1 Replace direct regression with attention-over-weights + differentiable triangulation

The most reliable way to beat DLT is **not** to regress 3D directly, but to let the network predict per-view per-joint weights and then triangulate with those weights. The codebase already has a prototype in `RobustTriangulationModel` (`motionflow_mv/fusion/robust_triangulation.py`). `attention_v2` should be refactored into a similar design:

- Input: `(x, y, confidence)` per view + camera parameters.
- Output: per-view per-joint reliability weights `w_{v,j}`.
- Final 3D: differentiable weighted DLT using `w_{v,j}`.

This keeps the strong geometric inductive bias of DLT while letting the network down-weight occluded or noisy views. It is also interpretable: `uncertainty.view_weights` can be written back into `HumanMotionIR`.

### 4.2 Use a geometrically meaningful camera representation

Do not flatten `P` directly. Instead, encode one of the following:

- **Ray representation:** for each view, compute the camera center `c = -R^T t` and the ray direction through the principal point. Encode `c` and the ray direction as separate tokens.
- **Intrinsics/extrinsics separation:** feed `K` and `(R, t)` through separate MLPs.
- **Plücker line coordinates:** represent each 2D point as the 3D ray it defines; this naturally couples the image observation with the camera geometry.

All values should be normalized to unit scale or camera-relative coordinates before being fed into the network.

### 4.3 Add geometric losses and real 3D supervision

If 3D ground truth is available (e.g., Shelf/Campus 3D annotations, Human3.6M, CMU Panoptic), train with a 3D loss such as MPJPE. Also add:

- **Reprojection loss:** `||Π_v(X_j) - x_{v,j}||` to keep the 3D estimate consistent with the 2D observations.
- **Bone-length consistency:** penalize deviation from a learned or dataset skeleton prior.
- **Temporal smoothness:** second-order finite difference on 3D joints.
- **Symmetry loss:** equal left/right limb bone lengths.

Training only against DLT pseudo-GT should be treated as a weak fallback, not the primary objective.

### 4.4 Upgrade the attention block

Replace `ViewAttentionFusion` with a standard transformer block:

- Multi-head self-attention over views.
- Positional encoding for view index or camera angle.
- Residual connections and layer normalization.
- Per-joint queries if joint-specific attention is desired.

This is a modest capacity increase and removes a single point of failure in the current single-query design.

### 4.5 Systematic ablation matrix

Before committing to a large model, run a controlled ablation to isolate the cause of instability:

| Factor | Options |
|---|---|
| Camera representation | raw flattened `P` / ray directions / `(K, R, t)` tokens |
| Fusion operation | additive / concatenation / cross-attention |
| Output head | direct 3D regression / per-view weights + DLT / volumetric soft-argmax |
| Supervision | DLT pseudo-GT / 3D GT / reprojection + 3D loss |
| Dataset | Shelf only / Shelf→Campus / synthetic AMASS pre-training |

The first goal is to reproduce or exceed the V1 baseline (≈80 px) with a camera-aware model; only then should the team try to close the gap to DLT (≈10 px).

### 4.6 Shift the paper story if DLT remains unbeaten

If the best learned fusion still ties or loses to DLT on reprojection, the CVPR/ICRA 2027 story should emphasize:

- **Uncertainty-aware fusion:** learned view weights and per-joint uncertainty for downstream robotics.
- **Occlusion robustness:** controlled occlusion experiments where DLT degrades and learned methods recover.
- **SMPL-aware multi-view fusion:** fusing per-view SMPL outputs (`HumanMotionIR`) rather than 2D keypoints, which is a more realistic robot-use setting.

## 5. Open Questions and Risks

- **Root cause of instability:** Is the poor performance of `attention_v2` due to (a) the raw projection-matrix representation, (b) the additive fusion, (c) the direct 3D regression head, or (d) all of the above? The ablation matrix in §4.5 is needed to tell.

- **Does camera-aware attention help at all on this data?** DLT is already excellent on Shelf (5.52 px median). The benefit of learned fusion may only appear under occlusion, outlier views, or noisy detections, which the current evaluation does not isolate.

- **Cross-dataset generalization:** V1 trained on Shelf fails on Campus (318 px). V2 is likely even more dataset-specific because it memorizes camera parameters. Any camera-aware model must be validated across multiple camera rigs, not just a single dataset.

- **Data bottleneck:** Shelf has no publicly released 3D GT for these scripts, so training is limited to DLT pseudo-labels. Real 3D GT (Human3.6M, Panoptic, 3DPW) is needed to train a learned fusion model that can beat DLT.

- **Compute constraint:** The A800-D is read-only. Large-scale training must run on the local RTX 4090 or another accessible GPU, limiting batch size and model capacity.

- **Risk of over-engineering:** The simplest fix—using the existing `RobustTriangulationModel` as the camera-aware head—may already outperform both V1 and V2 and should be prioritized over building a brand-new transformer architecture.

## 6. References

- Iskakov et al., *Learnable Triangulation of Human Pose*, ICCV 2019.
- Moliner et al., *Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction*, arXiv:2312.17106.
- Liao et al., *Multiple View Geometry Transformers for 3D Human Pose Estimation*, CVPR 2024.
- Shuai et al., *Adaptive Multi-view and Temporal Fusing Transformer for 3D Human Pose Estimation*, arXiv:2110.05092.
- Matsubara et al., *HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery*, CVPR 2025.
- Chharia et al., *MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation*, CVPR 2025.