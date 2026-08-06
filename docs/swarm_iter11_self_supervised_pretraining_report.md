# Iter11+ Self-Supervised Pretraining Roadmap

**Scope:** Add a large-scale, label-free pretraining stage for the MotionFlow-MultiView ray-attention fusion models, then fine-tune on the small labeled MPI-INF-3DHP set. The goal is to push the current best MPI-INF-3DHP validation MPJPE (~11.17 mm, cross-view residual model) toward CVPR/ICRA 2027 competitiveness while improving data efficiency and cross-dataset generalization.

## 1. Current state (what already exists)

* **Top models:** `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` in `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py` combines ray-attention, spatio-temporal (time + view) attention, per-view uncertainty, weighted DLT, differentiable Gauss-Newton refinement, and a residual MLP. It is trained in `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py`.
* **Losses available:** 3D MSE, optional reprojection loss (`motionflow_mv/losses/reprojection.py`), and skeleton-aware losses in `experiments/train_utils.py` (bone-length L1, temporal bone-length consistency, bone symmetry).
* **Data:** Canonical `.npz` files are produced by `motionflow_mv/data/webbridge_loader.py` for Human3.6M, MPI-INF-3DHP, AIST++, Shelf/Campus, and synthetic SMPL/AMASS (`experiments/generate_synthetic_multiview_dataset.py`). H36M WebBridge conversion is in progress at `data/webbridge/h36m`.
* **Precedent:** Earlier work (e.g. `experiments/train_ray_attention_temporal_residual_v2_mpiinf3dhp.py`, `experiments/train_ray_attention_temporal_finetune_mpiinf3dhp.py`) already uses two-stage *supervised* pretraining (single-frame V4 → temporal finetune), but the first stage still requires 3D ground truth.

## 2. What to add: a label-free SSL pretraining stage

The core idea is to pretrain the encoder/spatio-temporal backbone on **multi-view consistency** rather than 3D ground truth. The geometric triangulation head is naturally self-supervised: a 3D estimate can be reprojected into every view and compared with the input 2D keypoints.

### 2.1 Proposed SSL objectives

For an clip `x` of shape `(B, T, V, J, 3)` (2D keypoints + confidence):

1. **Masked-view reprojection loss** — randomly zero out (mask) a subset of views/time steps, triangulate from the remaining views, then reproject the predicted 3D pose into the masked views and minimize 2D error. This forces the model to fuse information across views instead of memorizing per-view biases.
2. **Visible-view reprojection loss** — standard reprojection of the full fused 3D pose back into all *observed* views.
3. **Temporal smoothness loss** — L2 penalty on frame-to-frame 3D joint acceleration; no labels needed.
4. **Skeleton consistency loss** — use the existing `temporal_bone_length_consistency_loss` and `bone_symmetry_loss` from `experiments/train_utils.py` to encourage plausible skeletons.
5. **Uncertainty-aware NLL** — the advanced model already predicts per-view log-variance; its Gaussian reprojection NLL can be trained without 3D GT because the target is the observed 2D keypoint.

These losses do not require `joints_3d` ground truth, so they can be applied to **H36M WebBridge train, AIST++, Shelf/Campus, and synthetic SMPL sequences** even where 3D labels would otherwise be scarce or off-limits.

### 2.2 Architecture / stage plan

Keep the advanced model unchanged; wrap a new pretrainer around it.

1. **Stage A — SSL pretraining:**
   * Load unlabeled multi-view clips from H36M train (subjects 1,5,6,7,8,9,11), AIST++ train, synthetic SMPL/AMASS, and optionally Shelf/Campus.
   * Train with the mixed SSL loss for many epochs (e.g. 50–100).
   * Save encoder + spatio-temporal transformer weights.

2. **Stage B — supervised fine-tuning:**
   * Initialize `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` from the SSL checkpoint (encoder + temporal/cross-view backbone).
   * Fine-tune on labeled MPI-INF-3DHP (S1 Seq1/2) with the existing 3D MSE + reprojection + uncertainty NLL objective.

This mirrors the current two-stage training scripts, but replaces the supervised single-frame pretrain stage with a label-free SSL stage on much more data.

## 3. Concrete implementation steps

1. Create `experiments/pretrain_ray_attention_ssl.py` that reuses the advanced model and adds the SSL loss.
2. Add a masking augmentation function `mask_views()` in the trainer:
   ```python
   def mask_views(x, mask_ratio=0.25, mask_value=0.0):
       # x: (B, T, V, J, 3)
       B, T, V, J, _ = x.shape
       mask = torch.rand(B, T, V, 1, 1, device=x.device) > mask_ratio
       x_masked = x.clone()
       x_masked[..., :2] = x[..., :2] * mask[..., :2]
       x_masked[..., 2] = x[..., 2] * mask[..., 0, 0]  # confidence
       return x_masked, mask[..., 0, 0]  # (B, T, V)
   ```
3. In the training loop, compute `pred_3d, weights, log_var, nll = model(x_masked, K=K, R=R, t=t)`. Then compute:
   * `loss_reproj_visible` on visible views using `reprojection_loss` with the observed confidence mask.
   * `loss_reproj_masked` on the masked views by reprojecting `pred_3d` and comparing only where `view_mask == True` for the *masked* set.
   * `loss_smooth` as mean frame-to-frame 3D acceleration.
   * `loss_bone` from `temporal_bone_length_consistency_loss(pred_3d, parents=H36M_17_PARENTS)` and optionally symmetry.
   * `loss = loss_reproj_visible + λ_masked * loss_reproj_masked + λ_smooth * loss_smooth + λ_bone * loss_bone + nll`.
4. Save the best checkpoint by visible-view reprojection loss on a held-out SSL validation set.
5. Modify the supervised fine-tuning script to accept `--pretrained_checkpoint` and load with `strict=False`.

## 4. Experiments to run

| Experiment | Data | Goal | Expected signal |
|---|---|---|---|
| SSL-A: Pretrain encoder on H36M train (no 3D labels) | `data/webbridge/h36m/s_*_acts_*_multiview.npz` | Learn view-invariant features | Lower MPI-INF-3DHP val MPJPE after fine-tune |
| SSL-B: Add AIST++ train | AIST++ converted `.npz` | More motion diversity | Improved PA-MPJPE / PCK on MPI-INF-3DHP |
| SSL-C: Add synthetic SMPL | `outputs/synthetic_multiview_dataset.npz` | Robustness to occlusion/outliers | Better robustness curves (noise 20 px) |
| SSL-D: Masked-view ablation | Same as SSL-A but mask_ratio ∈ {0, 0.15, 0.25, 0.4} | Optimal masking rate | Choose mask_ratio that maximizes downstream MPJPE gap |
| SSL-E: Data-efficiency fine-tune | Fine-tune on 10%, 25%, 50%, 100% of MPI-INF-3DHP labels | Quantify SSL benefit for paper | SSL should match or beat supervised baseline with fewer labels |
| SSL-F: Longer pretraining clips | T ∈ {13, 27, 49} | Temporal consistency benefit | Lower smoothness loss, better PCK AUC |

Run each SSL pretraining with `d=64, n_st_layers=2, residual_hidden=128` to match the current best supervised model. After pretraining, fine-tune on MPI-INF-3DHP S1 Seq1+2 → validate S2 Seq1 for 30 epochs with early stopping.

## 5. Metrics to track

* **Primary:** MPI-INF-3DHP val MPJPE, PA-MPJPE, PCK@50/100/150 mm, PCK AUC (0–150 mm).
* **SSL pretraining:** visible-view reprojection error (px), masked-view reprojection error (px), temporal smoothness loss, bone-length consistency loss, per-view uncertainty calibration (predicted vs. observed residual).
* **Data efficiency:** MPJPE as a function of labeled MPI-INF-3DHP fraction.
* **Robustness:** Gaussian noise 5/20 px, 50% joint occlusion, 20% outliers (reuse `experiments/eval_residual_robustness_mpiinf3dhp_v1.py`).
* **Cross-dataset generalization:** H36M S5 test, Shelf/Campus zero-shot (if labels available for evaluation only).

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Camera domain gap:** H36M (4 cam, mm), MPI-INF-3DHP (14 cam, m), AIST++ (9 cam, cm) | Normalize all camera rigs to unit scale and embed camera intrinsics/extrinsics explicitly (already done in ray embedding). Convert all datasets to meters before SSL. |
| **Scale/unit mismatch:** Some data are in mm, others in m | Preprocess with a `--unit_scale` argument in the SSL loader so every sequence is in meters before loss computation. |
| **SSL objectives conflict with 3D MSE during fine-tune** | Use a small initial learning rate and freeze the encoder for the first few epochs of fine-tuning (as in `train_ray_attention_temporal_finetune_mpiinf3dhp.py`). |
| **Overfit to synthetic motion** | Cap synthetic fraction at 20–30% of each mini-batch; apply stronger augmentation to synthetic data. |
| **Slow wall-clock time** | Pretrain on the largest available GPU (A800 if accessible); use gradient accumulation and mixed precision. Cache canonical `.npz` files. |
| **Masked-view loss under-determines 3D scale** | Retain the visible-view reprojection + bone-length + smoothness terms; they jointly constrain scale and skeleton structure. |

## 7. Expected impact on paper quality

A strong SSL stage turns the project from a “small labeled-dataset” story into a **scalable multi-view pose learning** story: pretrain on abundant unlabeled multi-view video, then fine-tune cheaply on the target dataset. This directly addresses reviewer concerns about generalization and data efficiency, and it gives CVPR/ICRA-level results on MPI-INF-3DHP with the potential to push MPJPE below 10 mm.
