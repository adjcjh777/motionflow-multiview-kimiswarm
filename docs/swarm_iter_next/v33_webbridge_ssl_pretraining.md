# v33 — Large-Scale Self-Supervised Pretraining on WebBridge

**Slug:** `webbridge_ssl_pretraining`  
**Date:** 2026-08-08  
**Target:** ICRA/CVPR 2027  
**Owner:** Swarm agent  
**Status:** Design proposal  

---

## 1. Problem statement and motivation

Current best supervised training of `OmniMultiViewFusionV5` relies on 3-D ground truth from Human3.6M and MPI-INF-3DHP. WebBridge contains far larger multi-view video corpora (H36M train subjects, AIST++, Shelf/Campus, and pseudo-labeled in-the-wild sequences) whose 3-D labels are absent, noisy, or incompatible. The v5 architecture already predicts per-view weights and a metric-scale 3-D pose from calibrated 2-D keypoints, so it can be pretrained with **no 3-D labels** using a masked-view reprojection objective.

Large-scale self-supervised (SSL) pretraining on WebBridge is expected to:

1. Improve downstream 3-D pose accuracy, especially when fine-tuning on small labeled sets.
2. Make the model more robust to missing/occluded views and variable-view inputs.
3. Provide a strong initialization for all downstream v31/v32 components (geometry fusion, hierarchical encoders, physical-space losses, trajectory consistency).

---

## 2. Proposed architecture changes

We introduce a **new pretraining stage** without modifying the core supervised pipeline or existing source files.

### 2.1 New top-level flag / entry point

```text
--ssl_pretrain
```

When passed to the v5 trainer, the training loop switches to label-free masked-view reprojection mode. The existing supervised 3-D MSE loss is disabled and the model is trained on unlabeled WebBridge `.npz` files.

### 2.2 New training script

`experiments/train_omniview_fusion_v5_webbridge_ssl_pretrain.py`

- Mirrors `experiments/train_omniview_fusion_v5_webbridge_multi.py`.
- Reuses `OmniMultiViewFusionV5` and the existing `build_model_from_args` helper.
- Loads unlabeled clips via `motionflow_mv/data/ssl_dataset.py` (`SSLTemporalClipDataset`, `SSLRandomClipDataset`, and `MaskedViewReprojectionDataset`).

### 2.3 New loss module

`motionflow_mv/losses/ssl_masked_view_reprojection.py`

Functions:

- `masked_view_reprojection_loss(pred_3d, x, K, R, t, mask, visible_weight, masked_weight)`
- `temporal_smoothness_loss(pred_3d)`
- `ssl_bone_length_consistency_loss(pred_3d, parents)`

The total SSL loss is:

```text
L_ssl = λ_vis * L_reproj_visible
      + λ_mask * L_reproj_masked
      + λ_smooth * L_temporal_smoothness
      + λ_bone * L_bone_length_consistency
```

### 2.4 New trainer wrapper

`motionflow_mv/training/ssl_pretrainer_v33.py`

- Wraps `TrainerV2`.
- Applies `mask_views(...)` before each forward pass.
- Tracks **visible-view reprojection error** and **masked-view reprojection error** as validation metrics.
- Saves the best SSL checkpoint by visible-view reproj. error.

### 2.5 Fine-tuning bridge

Add to the existing supervised trainer:

```text
--ssl_checkpoint PATH
```

which performs `model.load_state_dict(..., strict=False)` and optionally freezes the encoder for the first `N` epochs. This is the only change inside the supervised script; all other changes are in the new pretraining script and loss module.

---

## 3. Training command / ablation flags

### 3.1 Smoke test (CPU / RTX 4090)

```bash
python experiments/train_omniview_fusion_v5_webbridge_ssl_pretrain.py \
    --smoke \
    --ssl_pretrain \
    --ssl_mask_ratio 0.25 \
    --ssl_mask_mode mixed \
    --ssl_visible_reproj_weight 1.0 \
    --ssl_masked_reproj_weight 1.0 \
    --ssl_temporal_weight 0.02 \
    --ssl_bone_weight 0.05 \
    --epochs 1
```

### 3.2 Full WebBridge SSL pretraining (A800-D style)

```bash
python experiments/train_omniview_fusion_v5_webbridge_ssl_pretrain.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --ssl_pretrain \
    --ssl_mask_ratio 0.30 \
    --ssl_mask_mode mixed \
    --ssl_visible_reproj_weight 1.0 \
    --ssl_masked_reproj_weight 1.0 \
    --ssl_temporal_weight 0.02 \
    --ssl_bone_weight 0.05 \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
    --variable_view_permute \
    --d 64 --residual_hidden 128 --n_st_layers 2 --n_joint_layers 1 --n_heads 4 \
    --clip_len 9 --ssl_epochs 50 --batch_size 16 --train_samples 4000 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --weight_decay 1e-4 \
    --output outputs/omniview_fusion_v33_ssl_webbridge.pth
```

### 3.3 Fine-tune on labeled data

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --ssl_checkpoint outputs/omniview_fusion_v33_ssl_webbridge.pth \
    --warm_start_freeze_epochs 3 \
    # ... v32 supervised flags ...
```

### 3.4 Ablation knobs

| Flag | Meaning | Typical values |
|------|---------|----------------|
| `--ssl_mask_ratio` | Fraction of views/time steps to mask | 0.15, 0.25, 0.40 |
| `--ssl_mask_mode` | Mask whole views, time frames, or mixed | `view`, `time`, `mixed` |
| `--ssl_visible_reproj_weight` | Weight for visible-view reprojection | 1.0 |
| `--ssl_masked_reproj_weight` | Weight for masked-view reprojection | 1.0 |
| `--ssl_temporal_weight` | Temporal smoothness loss weight | 0.0, 0.02 |
| `--ssl_bone_weight` | Bone-length consistency weight | 0.0, 0.05 |
| `--ssl_epochs` | Pretraining epochs | 30, 50, 100 |

---

## 4. Expected metrics and baseline to beat

### 4.1 Primary metrics

After fine-tuning on the standard H36M + MPI-INF-3DHP supervised split, compare the SSL-pretrained model against the current best supervised-only baseline (v32 combined, A800-D queue).

| Metric | Baseline (supervised only) | SSL-pretrained target |
|--------|---------------------------|-----------------------|
| MPI-INF-3DHP val MPJPE (mm) | current v32 best | ≤ 5% improvement |
| PA-MPJPE (mm) | current v32 best | ≤ 5% improvement |
| Data-efficiency (25% labels) | from-scratch fine-tune | ≥ 10% improvement |
| Variable-view 2–14 cam MPJPE | current v32 | ≥ 5% improvement |

### 4.2 SSL-specific metrics

- Visible-view reprojection error (px) on held-out WebBridge validation clips.
- Masked-view reprojection error (px).
- Fine-tuning convergence speed (epochs to best val MPJPE).

### 4.3 Robustness checks

Reuse existing robustness scripts to measure SSL vs. supervised-only under:

- 2-D keypoint noise (σ = 1, 3, 5 px).
- Occluded joints / views.
- Outlier views (`outlier_view_prob = 0.3`).

---

## 5. Risks / unknowns

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| **Scale ambiguity** | SSL reprojection alone does not constrain global metric scale | Use metric-normalized WebBridge `.npz`; add bone-length prior |
| **Domain gap** | H36M 4-cam vs. MPI 14-cam vs. in-the-wild pseudo-labels | Camera-conditioned view embedding + domain embedding already in v5 |
| **High compute cost** | WebBridge is much larger than labeled H36M/MPI | Train in stages; start with 50 epochs and scale up |
| **Overfit to reprojection** | Model may memorize per-camera biases | Masked-view objective forces cross-view reasoning; strong augmentations |
| **Fine-tune instability** | SSL weights may diverge when switching to supervised loss | Freeze encoder for 3 epochs; use 10× smaller LR initially |
| **Mask ratio sensitivity** | Too high ratio may make task unsolvable | Smoke sweep over {0.15, 0.25, 0.40} before full run |

---

## 6. Deliverables

1. `experiments/train_omniview_fusion_v5_webbridge_ssl_pretrain.py`
2. `motionflow_mv/losses/ssl_masked_view_reprojection.py`
3. `motionflow_mv/training/ssl_pretrainer_v33.py`
4. `configs/splits/webbridge_ssl_pretrain_train_val.yaml` (optional, pointing to all unlabeled WebBridge train subjects)
5. This proposal document

## 7. Dependencies on other v31/v32 work

- Reuses `motionflow_mv/fusion/omniview_fusion_v5.py` as-is.
- Reuses `motionflow_mv/data/ssl_dataset.py` for label-free loading.
- Can leverage `use_variable_view_training`, `use_camera_view_embedding`, and `use_set_view_aggregator` flags already in the v5 trainer.
- Fine-tuning stage depends on the existing supervised `warm_start` logic (add `--ssl_checkpoint` bridge).
