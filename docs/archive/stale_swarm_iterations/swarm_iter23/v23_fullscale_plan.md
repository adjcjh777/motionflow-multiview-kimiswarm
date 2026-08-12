# v23 Full-Scale A800 Launch Plan

**Model:** v23 = v18 deformable cross-view attention + v22 Kinematic Anthropometric Prior (KAP), *without* neural bundle adjustment.  
**Goal:** Train the full WebBridge/H36M/MPI mixed-dataset model on an A800 GPU and compare its final MPJPE/PA-MPJPE to the v18 and v22 baselines.  
**Reference scripts:** `scripts/run_v23_kap_no_ba_a800_fullscale.sh`, `scripts/launch_v23_a800_tmux.sh`.

## 1. Model & training config (full-scale)

| Item | Value | Notes |
|------|-------|-------|
| Backbone dims | `d=128`, `residual_hidden=256` | Same as v18/v22 full-scale. |
| ST layers | `n_st_layers=3` | Spatial-temporal transformer stack. |
| Graph layers | `graph_num_layers=1`, `n_joint_layers=1` | Lightweight graph branch. |
| Attention heads | `n_heads=4` | Used in deformable cross-view attention. |
| Dataset manifest | `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml` | WebBridge + H36M + MPI-INF-3DHP mix. |
| Training samples | `10000` per epoch | Full mixed-dataset epoch. |
| Validation stride | `val_stride=10` | Standard skip for faster validation. |
| Epochs | `60` | Same as v18 full-scale. |
| Batch size | `16` | Fits comfortably on A800 80 GB. |
| Optimizer | AdamW, `lr=1e-3`, cosine, 3-epoch warmup, `lr_min=1e-6` | Matches v18/v22 schedule. |
| EMA | `ema_decay=0.999` | For stable evaluation checkpoint. |
| Gradient clipping | `max_grad_norm=1.0` | Prevents KAP spike at early epochs. |
| KAP loss weight | `kap_loss_weight=0.01` | v22 default; can be tuned if validation wobbles. |

**Key v23-only flags (no `--use_neural_bundle_adjustment_v21`):**

```bash
--use_deformable_cross_view_attention_v18 \
--use_kinematic_anthropometric_prior_v22 \
--kap_loss_weight 0.01
```

All other flags mirror the v18 full-scale recipe (full-precision DLT, robust/IRLS reweighting, multiscale fusion, camera conditioning, epipolar bias, variable-view training, outlier augmentation, etc.).

## 2. Full-scale launch script

Save as `scripts/launch_v23_a800_fullscale.sh` and run on the A800-D host (`ssh a800-D`).  The script picks the GPU via the first argument so it can be dispatched to whichever A800 device is free.

```bash
#!/usr/bin/env bash
# Launch v23 full-scale on a single A800 GPU via tmux.
# Usage: bash scripts/launch_v23_a800_fullscale.sh [GPU_ID]
set -euo pipefail

GPU=${1:-0}
NAME="v23_kap_no_ba_full_gpu${GPU}"
OUTPUT="outputs/omniview_fusion_v23_kap_no_ba_fullscale_gpu${GPU}.pth"
LOG="outputs/omniview_fusion_v23_kap_no_ba_fullscale_gpu${GPU}.log"
REPO="/mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm-iter20"
VENV="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u"

cd "$REPO"
mkdir -p outputs

tmux kill-session -t "$NAME" 2>/dev/null || true

tmux new-session -d -s "$NAME" \
    "CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$REPO $VENV experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_kinematic_anthropometric_prior_v22 \
    --kap_loss_weight 0.01 \
    --num_workers 0 \
    --d 128 --residual_hidden 256 --n_st_layers 3 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 60 --batch_size 16 --train_samples 10000 --val_stride 10 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training \
    --variable_view_min_views 2 --variable_view_max_views 14 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 \
    --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 3 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output $OUTPUT > $LOG 2>&1"

echo "Launched v23 full-scale on GPU $GPU as tmux session $NAME"
tmux list-sessions
```

### Quick launch on the first free GPU

```bash
# Example: launch on GPU 0 once it is free
ssh a800-D
bash /mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm-iter20/scripts/launch_v23_a800_fullscale.sh 0
```

## 3. GPU resource estimate

### 3.1 Memory (per run)

| Component | Estimate | Source / rationale |
|-----------|----------|--------------------|
| Model weights (d=128, graph/attention/ST heads) | ~1.0–1.3 GB | Comparable to v18/v22 at same width. |
| Activations / gradients (batch 16, B=16, T≈9, V≤14, J=17/28) | ~18–24 GB | Variable-view training samples 2–14 views; worst case is V=14. |
| Optimizer states (AdamW) | ~2× model weights | ~2.0–2.5 GB. |
| CUDA / PyTorch overhead + fragmentation | ~2–3 GB | A800 driver/context overhead. |
| **Total per run** | **~24–30 GB** | Leaves headroom on an 80 GB A800; batch 16 is safe. |

*Actionable check after launch:*

```bash
ssh a800-D "watch -n 5 nvidia-smi"
```

If memory stays below ~45 GB, consider a future run with `batch_size=24` for throughput, otherwise keep `batch_size=16`.

### 3.2 Compute / wall-clock time (estimate)

| Run type | Samples/epoch | Epochs | Steps | Relative work vs. small |
|----------|----------------|--------|-------|--------------------------|
| v23 small (GPU4/6 baseline) | 2 000 | 20 | ~2 500 | 1× |
| **v23 full** | **10 000** | **60** | **~37 500** | **15×** |

Assuming the v23 small run completes in roughly 2–4 hours on A800, the full-scale run is estimated at **30–60 hours** (≈1.5–2.5 days).  Actual time depends on:

- Number of validation samples and `val_stride`.
- EMA update overhead.
- Variable-view sampling distribution (more views = more compute).
- Disk/loader latency on A800-D.

*Monitoring:*

```bash
# On a800-D
 tail -f /mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/omniview_fusion_v23_kap_no_ba_fullscale_gpu*.log
```

### 3.3 Recommended scheduling

- **GPU affinity:** Use a single A800 GPU per run.  v23 full-scale needs one full 80 GB device.
- **Current A800-D occupancy (as of this plan):**
  - GPU4, GPU6: v23 small (in progress, waiting for first-epoch `val_MPJPE`).
  - GPU5: v18 full.
  - GPU7: v11 full.
- **Recommendation:** Launch v23 full-scale on the first GPU that becomes free (GPU0–GPU3 when idle, or whichever of GPU4/5/6/7 frees up first after the current runs finish).  Do **not** kill any running job; queue the full-scale launch and start it when a device is available.

## 4. Success criteria

1. Job starts without OOM on A800 with `batch_size=16`.
2. First-epoch validation MPJPE is logged and is comparable to or better than the v23 small run.
3. Training completes 60 epochs with EMA checkpoint saved.
4. Final checkpoint is evaluated against:
   - v18 full-scale baseline (GPU5).
   - v22 full-scale (v18 + v21 + KAP) if available.
   - v23 small (GPU4/6) to confirm full-dataset scaling gain.

## 5. Next steps

1. Wait for an A800 GPU to free up (see occupancy above).
2. Copy `scripts/launch_v23_a800_fullscale.sh` to A800-D (read-only pull from local repo) and launch.
3. Post the resulting `val_MPJPE` and `train_MPJPE` curves to the v23 tracking issue.
