# Self-Supervised Multi-View Pose Pretext Design

## Core Idea

Train the existing temporal ray-attention residual model **without 3D labels** by asking it to reconstruct randomly masked views from the remaining views.  The pretext is *masked-view ray reprojection*: given a temporal clip of 2D keypoints and calibrated cameras, hide a subset of views/time-steps, triangulate a 3D pose from the visible evidence, and enforce that the predicted 3D joints reproject correctly into the masked views.

This turns abundant unlabeled multi-view video (H36M WebBridge train, AIST++, Shelf/Campus, synthetic SMPL) into a pretraining stage, then fine-tune on the small labeled MPI-INF-3DHP set.

## Key Design Decisions

1. **Reuse the production architecture.**
   - `RayAttentionFusionModelTemporalResidual` in `motionflow_mv/fusion/ray_attention_temporal_residual_model.py` already outputs per-view weights and a 3D pose.  No model changes are required; the SSL trainer simply supplies a new loss.
   - The differentiable weighted-DLT layer gives a geometrically sound 3D estimate even when some views are masked.

2. **Masked-view reprojection as the primary signal.**
   - Sample a binary mask `M ∈ {0,1}^{B,T,V}`.
   - Feed `x_masked` into the model with confidences zeroed for masked slots.
   - Reproject the predicted 3D pose into **all** views, but compute the loss only on masked views.
   - This prevents the network from memorizing per-view biases and forces cross-view fusion.

3. **Auxiliary label-free regularizers.**
   - **Visible-view reprojection loss** keeps the solution consistent with observed 2D keypoints.
   - **Temporal smoothness loss** penalizes implausible frame-to-frame accelerations.
   - **Bone-length consistency / symmetry losses** from `experiments/train_utils.py` inject a skeleton prior.

4. **Metric-scale normalization.**
   - Convert every dataset to meters before SSL (H36M `÷1000`, MPI-INF-3DHP already in meters).  This lets one pretrained checkpoint transfer across camera rigs.

5. **Two-stage training.**
   - Stage A: SSL pretrain on unlabeled multi-view clips.
   - Stage B: supervised fine-tune on MPI-INF-3DHP S1 Seq1/2 with 3D MSE + reprojection.

## Loss Formulation

For a clip `X ∈ R^{B×T×V×J×3}` (2D keypoints + confidence) and a masked view set `M`:

- Reprojection loss:
  \[
  \mathcal{L}_{\text{reproj}}(X, \hat{X}) = \sum_{v \in \mathcal{V}} \mathbb{1}_{v \notin M} \|\pi_v(\hat{X}) - x_v\|_2^2
  \]
- Masked-view loss is the same reprojection error evaluated on `v ∈ M`.
- Temporal smoothness:
  \[
  \mathcal{L}_{\text{smooth}} = \sum_t \|\hat{X}_{t+1} - 2\hat{X}_t + \hat{X}_{t-1}\|_2^2
  \]
- Total SSL loss:
  \[
  \mathcal{L} = \lambda_{\text{vis}}\mathcal{L}_{\text{vis}} + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}} + \lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}} + \lambda_{\text{bone}}\mathcal{L}_{\text{bone}}
  \]

## References

- Architecture / model: `motionflow_mv/fusion/ray_attention_temporal_residual_model.py`
- Reprojection loss: `motionflow_mv/losses/reprojection.py`
- Skeleton consistency losses: `experiments/train_utils.py`
- Supervised temporal trainer: `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py`
- Canonical data: `motionflow_mv/data/webbridge_loader.py` and `data/webbridge/h36m/...`
