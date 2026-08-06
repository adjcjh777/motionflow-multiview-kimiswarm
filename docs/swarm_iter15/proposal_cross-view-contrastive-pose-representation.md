# Cross-View Contrastive Pose Representation

**Date:** 2026-08-06
**Author:** MotionFlow-MultiView research engineering
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP S2/Seq1 clean **9.32 mm** MPJPE
**Status:** Proposal, skeleton implemented, not yet trained

---

## 1. One-sentence hypothesis

Adding an auxiliary **cross-view joint-level contrastive loss** on the per-joint spatio-temporal features — pulling the same joint across views together and pushing different joints apart — will make the multi-view representation more view-invariant and physically consistent, improving robustness across views while preserving or improving the anchor’s clean MPJPE.

---

## 2. Related existing files/modules

- **Anchor model:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`
  - `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` (clean MPJPE 9.32 mm)
- **Parent architecture:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py`
- **Base cross-view encoder:** `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py`
- **Principal-point correction:** `motionflow_mv/fusion/principal_point_correction.py`
- **Anchor training script:** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
- **Existing auxiliary losses:** `motionflow_mv/losses/reprojection.py`, `motionflow_mv/losses/velocity.py`, `motionflow_mv/losses/bone_length.py`

---

## 3. Proposed code changes

### 3.1 New loss module

**Create:** `motionflow_mv/losses/crossview_pose_contrast.py`

- New class: `CrossViewJointContrastiveLoss(nn.Module)`
  - Args:
    - `d`: input feature dimension
    - `projection_dim`: embedding dimension for contrastive learning (default 64)
    - `temperature`: softmax temperature (default 0.07)
  - Forward:
    - Input: per-joint multi-view features `feat` of shape `(N, V, J, d)`
    - Projects and L2-normalises each token.
    - Builds an all-pairs similarity tensor `sim[n, v, j, vp, jp]`.
    - Defines positives as same-joint/cross-view tokens and negatives as different-joint tokens.
    - Returns a multi-positive InfoNCE loss as a scalar.

### 3.2 New model subclass

**Create:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_crossview_contrast_model.py`

- New class: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCrossViewContrast`
  - Inherits from `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
  - Constructor adds:
    - `contrastive_dim: int = 64`
    - `contrastive_temperature: float = 0.07`
    - `contrastive_loss_weight: float = 0.1`
  - Adds a `CrossViewJointContrastiveLoss` instance operating on the spatio-temporal transformer features after principal-point correction.
  - Provides:
    - `compute_contrastive_loss(x, ...) -> Tensor` — explicit feature extraction + loss.
    - `forward_with_contrastive_loss(x, ...) -> (pred_3d, weights, c_loss)` — single-pass anchor forward with a forward-hook capture of the transformer features.
  - Does **not** modify the anchor triangulation, weight head, or residual refinement path.

### 3.3 Training integration (one-line follow-up)

**Modify:** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`

- Add argument `--contrastive_loss_weight` (default `0.0`).
- Instantiate the new model class when the weight is positive.
- Replace `outputs = model(xb, K=K, R=R, t=t)` with the hook-based variant or call `model.compute_contrastive_loss` and add `args.contrastive_loss_weight * c_loss` to the total loss.

No changes are required to the principal-point correction, triangulation, or evaluation code.

---

## 4. Training / smoke plan (≤ 5 epochs, RTX 4090)

### Smoke test (minutes)

Run the provided unit test:

```bash
python tests/test_crossview_pose_contrast.py
```

Expected output: `cross-view contrastive tests passed` in ~5–10 seconds on CPU.

### Short train (≤ 5 epochs)

Use the anchor training script with the new model class.  A typical smoke split on `data/webbridge/mpi_inf_3dhp/`:

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 --epochs 5 \
  --pp_loss_weight 0.1 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --model_type <mapped to contrastive model>
```

Estimated runtime on RTX 4090: ~8–12 min/epoch for `d=32`, `batch_size=8`, `clip_len=13`. Total ≤ 60 min for 5 epochs (slightly slower than anchor because of the extra similarity computation).

### A800 read-only validation

Reuse existing eval outputs/checkpoints; do not submit new jobs. Compare against the anchor checkpoint on MPI-INF-3DHP S2/Seq1 and the existing robustness matrices.

---

## 5. Success metrics

| Metric | Target | How measured |
|---|---|---|
| Clean MPJPE (MPI-INF-3DHP S2/Seq1) | ≤ 9.20 mm (improvement over 9.32 mm anchor) | existing eval script |
| PA-MPJPE | ≤ 5.50 mm | existing eval script |
| Robustness to view dropout | ≥ 5% relative improvement over anchor when 1–2 views are dropped at test time | reuse `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` |
| Robustness to joint occlusion | ≥ 5% relative improvement over anchor under joint-dropout | same robustness matrix |
| Calibration drift (±5 px PP noise) | ≥ 5% relative improvement over anchor | same robustness matrix with PP perturbation |
| Training stability | Smoke test passes; no NaNs/Inf; validation MPJPE decreases monotonically over 5 epochs | smoke + short train logs |

---

## 6. Risk and fallback

| Risk | Mitigation / Fallback |
|---|---|
| **Contrastive loss dominates early training** and destabilises the anchor. | Start with a low weight (e.g., `0.01`), use linear warm-up, or freeze the anchor weights for the first epoch and train only the projection head. |
| **No measurable MPJPE gain** despite improved representations. | Treat as a negative ablation and remove the loss; the change is a drop-in subclass, so reverting means deleting the new model file and using the anchor unchanged. |
| **Memory / runtime overhead** from the all-pairs similarity matrix. | The loss is `O(V² J²)`; if it becomes a bottleneck, subsample negatives (e.g., per-anchor hard negatives) or reduce `projection_dim`/`contrastive_dim`. |
| **Over-fitting to a specific camera rig** because positives/negatives are defined per-sample. | Add a small inter-sample bank (memory buffer) or sample negatives across the batch, not just within the same clip. |
| **Principal-point correction and contrastive objective conflict.** | Train the PP head first (existing `pp_pretrain_epochs`), then add the contrastive term. |
