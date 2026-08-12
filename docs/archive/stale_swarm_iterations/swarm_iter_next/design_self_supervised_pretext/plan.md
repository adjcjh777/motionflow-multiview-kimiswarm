# Implementation Plan: Self-Supervised Pretext for Temporal Ray-Attention Fusion

## Goal
Add a label-free SSL pretraining stage for `RayAttentionFusionModelTemporalResidual`, then fine-tune on MPI-INF-3DHP to improve data efficiency and cross-dataset generalization for the ICRA/CVPR 2027 submission.

## Phase 1 — Data Pipeline (1 day)

1. **Verify canonical `.npz` availability.**
   - H36M WebBridge: `data/webbridge/h36m/s_*_acts_*_multiview.npz`
   - MPI-INF-3DHP: `data/webbridge/mpi_inf_3dhp/s_01_seq_*_v14_multiview_m.npz`
   - Shelf/Campus: `data/shelf_campus/*/pseudogt.npz`

2. **Create `motionflow_mv/data/ssl_dataset.py`**
   - `SSLClipDataset(npz_paths, clip_len, stride)` returns `(x, K, R, t)` without requiring `joints_3d`.
   - `RandomClipDataset` for oversampling long sequences.

3. **Unit normalization helper**
   - Add `--unit_scale` argument (e.g., H36M `1.0/1000`, Shelf/Campus `1.0/100`) so all camera `t` and 2D inputs are in meters.

## Phase 2 — SSL Trainer (2 days)

1. **Create `experiments/pretrain_ray_attention_ssl.py`**
   - Load `RayAttentionFusionModelTemporalResidual(j, d, n_views, ...)`.
   - Implement `mask_views(x, mask_ratio, mask_mode)`:
     - `mask_mode="view"`: mask entire views for random frames.
     - `mask_mode="time"`: drop full frames (temporal gap filling).
     - `mask_mode="mixed"`: combine both.
   - Compute losses:
     - `loss_reproj_visible`: from `motionflow_mv/losses/reprojection.py` on unmasked slots.
     - `loss_reproj_masked`: reprojection error computed only on masked slots.
     - `loss_smooth`: frame-to-frame 3D acceleration L2.
     - `loss_bone`: `temporal_bone_length_consistency_loss` from `experiments/train_utils.py`.
   - Total:
     ```
     loss = λ_vis*loss_reproj_visible + λ_mask*loss_reproj_masked + λ_smooth*loss_smooth + λ_bone*loss_bone
     ```
   - Track best checkpoint by **visible-view reprojection error on a held-out SSL validation set**.

2. **Augmentations**
   - 2D Gaussian noise (0–5 px).
   - View dropout at training time (complementary to the SSL mask).
   - Scale jitter in metric meters (±5%).

## Phase 3 — Fine-Tuning Bridge (1 day)

1. **Extend `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py`**
   - Add `--pretrained_checkpoint PATH` argument.
   - Load checkpoint with `model.load_state_dict(..., strict=False)` and optionally freeze encoder for first `N` epochs.
   - Keep supervised 3D MSE + reprojection loss.

## Phase 4 — Evaluation (2 days)

1. **Data-efficiency curves**
   - Fine-tune on 10%, 25%, 50%, 100% of MPI-INF-3DHP labels.
   - Compare vs. training from scratch.

2. **Robustness evaluation**
   - Reuse `experiments/eval_residual_robustness_mpiinf3dhp_v1.py`.
   - Compare SSL-pretrained vs. supervised-only under noise / occlusion / outliers.

3. **Metrics to report**
   - MPI-INF-3DHP val MPJPE, PA-MPJPE, PCK@50/100/150 mm, PCK AUC.
   - SSL validation reprojection error (px).
   - Masked-view reprojection error (px).

## Deliverables

- `motionflow_mv/data/ssl_dataset.py`
- `experiments/pretrain_ray_attention_ssl.py`
- Updated `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py` (pretrained checkpoint loading)
- `docs/swarm_iter_next/design_self_supervised_pretext/pretrain_ssl_pretext_demo.py` (prototype)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| SSL scale under-constrained | Keep visible-view reproj. + bone-length + smoothness losses |
| Camera domain gap (H36M 4-cam vs MPI 14-cam) | Normalize all rigs to meters; use camera-conditioned embeddings already in the model |
| Fine-tune divergence | Freeze encoder for first 3 epochs; use 10× smaller initial LR |
| Overfit to synthetic motion | Cap synthetic fraction to 30% per batch |
