# Direction 5: Temporal consistency and long-term dependencies

**Date:** 2026-08-05  
**Baseline:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`  
**Current best:** MPI-INF-3DHP clean 9.32 mm / PA-MPJPE 5.37 mm  
**Priority:** P1

## Problem statement

The current best model triangulates per-frame 3D poses from multi-view 2D keypoints using a spatio-temporal Transformer, but it is trained with a per-frame MSE and a short default clip length of 13 frames. This encourages accurate independent frames but does not explicitly penalize high-frequency jitter or enforce temporal coherence across the clip. Adding a velocity-consistency term and training on longer clips should reduce jitter, improve the velocity MPJPE metric, and potentially push clean MPJPE below 9.0 mm while keeping the change small and compatible with the existing principal-point correction model.

## Simplest concrete next experiment

1. Add a reusable `velocity_loss` (L2 and L1 variants) in `motionflow_mv/losses/velocity.py`.  
2. Wire it into the existing PP training script via `--velocity_loss_weight`.  
3. Train with `clip_len=25` (vs. the current 13) and `velocity_loss_weight=0.05`, warm-starting from the best PP curriculum checkpoint.  
4. Compare clean/velocity MPJPE against the baseline.

## Files touched / diff sketch

### New: `motionflow_mv/losses/velocity.py`
```python
import torch

def velocity_loss(pred: torch.Tensor, gt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    if pred.shape[-3] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    pred_v = pred[..., 1:, :, :] - pred[..., :-1, :, :]
    gt_v = gt[..., 1:, :, :] - gt[..., :-1, :, :]
    diff = pred_v - gt_v
    loss = (diff ** 2).sum(dim=-1)
    return loss.mean() if reduction == "mean" else loss.sum()
```

### Modified: `motionflow_mv/losses/__init__.py`
Exports `velocity_loss` and `velocity_l1_loss`.

### Modified: `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
- Added CLI arg `--velocity_loss_weight` (default 0.0).  
- Added `from motionflow_mv.losses import ..., velocity_loss`.  
- In training loop:
  ```python
  if args.velocity_loss_weight > 0.0:
      loss = loss + args.velocity_loss_weight * velocity_loss(pred, yb)
  ```

### New: `scripts/run_temporal_velocity_longclip_wsl.sh`
GPU launcher (do not run while RTX 4090 is busy):
```bash
python -u experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 25 --d 64 --residual_hidden 128 --n_st_layers 2 --epochs 10 \
  --train_samples 1000 --batch_size 4 --val_stride 50 \
  --pp_loss_weight 0.1 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --velocity_loss_weight 0.05 \
  --warm_start outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth \
  --output outputs/ray_attention_temporal_crossview_residual_principal_point_velocity_longclip.pth
```

### New: `experiments/verify_velocity_loss.py`
CPU-only sanity check that synthetic sequences yield zero loss for perfect predictions and monotonically increasing loss under added jitter.

## CPU verification run

Command:
```bash
python experiments/verify_velocity_loss.py
```

Output:
```
Velocity loss sanity checks passed.
  perfect L2 loss: 0.000000e+00
  L2 losses vs jitter: [(0.0, 0.0), (0.001, 6.2925892052589916e-06), (0.01, 0.0006292590405791998), (0.1, 0.06292590498924255)]
  noisy L2: 1.573148e-02, noisy L1: 1.156070e-01
```

The loss behaves as expected: zero for perfect matches and strictly increasing with jitter magnitude.

## Expected success metric

- Clean MPI-INF-3DHP MPJPE ≤ 9.0 mm (baseline 9.32 mm).  
- Velocity MPJPE reduced by ≥ 10 % relative to the baseline on the same validation split.  
- No regression in PA-MPJPE.  
- If the velocity term harms per-frame accuracy, ablate `velocity_loss_weight` in {0.01, 0.05, 0.1}.

## Resource requirements

- Training: **GPU** (RTX 4090, queued).  
- CPU verification: completed above; no GPU needed.  
- A800-D: not used (read-only constraint respected).

## Notes / next steps

- Run `scripts/run_temporal_velocity_longclip_wsl.sh` when the RTX 4090 queue is free.  
- If `clip_len=25` causes memory issues, reduce batch size to 2 or use gradient accumulation.  
- A stronger variant would add multi-scale temporal convolutions (direction 6), but that should only be attempted after this minimal velocity experiment is validated.
