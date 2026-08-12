# Self-Supervised Masked-View Completion for Multi-View 3D Pose

## One-sentence hypothesis

Adding an explicit **masked-view 2D completion head**—trained only on reprojection error of masked-out views/timesteps—forces the temporal cross-view fusion model to learn a physically consistent 3D skeleton that generalises better to missing or degraded camera views.

## Related existing files/modules

- **Anchor model**: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`
- **Parent residual model**: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py`
- **Base spatio-temporal model**: `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py`
- **Reprojection loss utilities**: `motionflow_mv/losses/reprojection_consistency.py`
- **SSL dataset / masking utilities**: `motionflow_mv/data/ssl_dataset.py`
- **Existing SSL pre-training script**: `experiments/pretrain_ray_attention_ssl.py`
- **Training utilities**: `experiments/train_utils.py`

## Proposed code changes

1. **New loss module**: `motionflow_mv/losses/masked_view_completion.py`
   - Class/function: `masked_view_completion_loss(pred_2d, target_2d, mask, confidences=None, eps=1e-6)`
   - Computes per-slot 2D distance on masked `(B, T, V, J)` positions only.
   - Returns both the masked loss and a diagnostics dict.

2. **New model subclass**: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_completion_model.py`
   - Class: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCompletion`
   - Inherits from `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
   - Adds a lightweight per-view 2D completion head (`completion_mlp`) that refines the reprojection of the fused 3D pose into each view using per-view features.
   - `forward` returns `(pred_3d, weights, pred_2d_completed)` when `return_completion=True`; otherwise returns `(pred_3d, weights)` to stay backward-compatible.
   - Reuses the corrected intrinsics `K_corrected` from the principal-point branch so calibration alignment is preserved.

3. **Loss package registration** (minimal):
   - Add `from .masked_view_completion import masked_view_completion_loss` to `motionflow_mv/losses/__init__.py`.

4. **Training script** (skeleton, not executed):
   - Extend `experiments/pretrain_ray_attention_ssl.py` to optionally load the completion model and add `lambda_completion` term:
     ```
     loss = λ_vis·loss_vis + λ_mask·loss_mask + λ_completion·loss_completion + λ_smooth·loss_smooth + λ_bone·loss_bone
     ```
   - The completion loss is evaluated on masked slots only, while the standard reprojection loss is evaluated on visible slots.

## Training/smoke plan

- **Dataset**: Use existing H36M multi-view `.npz` clips (`data/h36m_hf/s_01_acts_*.npz`) already used by `pretrain_ray_attention_ssl.py`.
- **Masking**: `mask_ratio=0.25`, `mask_mode="mixed"` (views and time).
- **Smoke** (≤5 epochs, RTX 4090):
  ```bash
  python experiments/pretrain_ray_attention_ssl.py \
      --train data/h36m_hf/s_01_acts_02_06_multiview.npz \
              data/h36m_hf/s_05_acts_02_06_multiview.npz \
      --val data/h36m_hf/s_09_acts_02_06_multiview.npz \
      --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
      --epochs 5 --batch_size 8 --train_samples 2000 \
      --mask_ratio 0.25 --lambda_completion 1.0 \
      --output outputs/ray_attention_ssl_completion_smoke.pth
  ```
- **Estimated runtime**: ~45–90 minutes for 5 epochs on a single RTX 4090 with batch size 8 and clip length 13 (model has ~1.2 M parameters and 4 views).
- **Checkpointing**: Save best by total validation reprojection loss; evaluate MPJPE on MPI-INF-3DHP S2/Seq1 in a separate supervised fine-tuning / evaluation run (reuse existing eval scripts).

## Success metrics

- **Primary**: Clean MPJPE on MPI-INF-3DHP S2/Seq1 ≤ **9.32 mm** (match or beat current anchor; target improvement ≥ 0.20 mm is meaningful).
- **Robustness axis**: Relative MPJPE degradation when 1–2 views are randomly dropped during inference (e.g., view dropout rate 0.25 and 0.5). Completion model should degrade less steeply than the anchor.
- **Self-supervised diagnostics**: Masked-view reprojection loss on validation should decrease across the 5 smoke epochs and be comparable or lower than the visible reprojection loss.
- **Calibration alignment**: Reprojection error (pixels) on held-out visible views should improve or remain stable, confirming the 3D skeleton stays physically consistent.

## Risk and fallback

- **Risk 1 – Completion head dominates reprojection loss and calibration drifts.**
  - Mitigation: start with `lambda_completion=0.1` and only scale to 1.0 after reprojection loss converges; monitor reprojection error on *visible* slots.
  - Fallback: disable completion head (`lambda_completion=0`) and use the model as a plain principal-point model.
- **Risk 2 – Extra parameters slow smoke runs beyond 5 epochs.**
  - Mitigation: completion head is tiny (one hidden layer of 64 dims, ~15 k parameters); if still too slow, reduce `clip_len` to 9 or `d` to 32.
  - Fallback: run the smoke on CPU with a synthetic 4-view clip (existing `smoke_pretrain_ray_attention_ssl.py` already supports CPU).
- **Risk 3 – No MPJPE improvement on MPI-INF-3DHP.**
  - Fallback: the masked-view completion objective is still valuable as a self-supervised pre-training proxy; use the trained weights to initialise supervised fine-tuning, or publish as an ablation showing robustness gains without 3D labels.
