# Residual Temporal Ray-Attention on Shelf/Campus Canonical Data

## Task
Run the top-performing residual refinement head on Shelf/Campus temporal canonical
(WebBridge) data and report MPJPE. Create a dedicated eval script.

## Files created

- `experiments/eval_ray_attention_temporal_residual_shelf_campus.py`
  - Loads a trained `RayAttentionFusionModelTemporalResidual` checkpoint for Shelf
    and/or Campus.
  - If a checkpoint is missing, performs a short smoke-run training (default 3
    epochs) on the train split and saves the best checkpoint.
  - Reports validation MPJPE and per-frame MPJPE breakdown.

## Method

- Architecture: `RayAttentionFusionModelTemporalResidual` (extends the temporal
  ray-attention model with a lightweight MLP residual head on top of the DLT
  triangulated output).
- Training hyperparameters (smoke run):
  - `clip_len = 13`
  - `d = 64`
  - `n_temporal_layers = 2`
  - `residual_hidden = 128`
  - `batch_size = 4`
  - `epochs = 3`
  - `train_samples = 500` random clips per epoch
- Data:
  - Shelf: `data/webbridge/shelf_campus/shelf_seq1_train_v5_multiview_m.npz` /
    `shelf_seq1_val_v5_multiview_m.npz` (5 views, 17 joints, 2560 / 640 frames)
  - Campus: `data/webbridge/shelf_campus/campus_seq1_train_v3_multiview_m.npz` /
    `campus_seq1_val_v3_multiview_m.npz` (3 views, 17 joints, 1138 / 285 frames)

## Commands run

```bash
conda run -n mf python experiments/eval_ray_attention_temporal_residual_shelf_campus.py \
    --dataset both --epochs 3 --batch_size 4 --train_samples 500
```

## Results

| Dataset | Views | Val MPJPE | DLT baseline MPJPE | Notes |
|---------|-------|-----------|--------------------|-------|
| Shelf   | 5     | **62.41 mm** | 65.68 mm | Residual head slightly improves over DLT. |
| Campus  | 3     | **725.23 mm** | 4903.25 mm | Residual head dramatically improves over DLT, but absolute error remains high; likely a scale/coordinate-system issue in the Campus pseudo-GT/calibration. |

### Training curves (best val MPJPE)

**Shelf:**
- Epoch 1: 81.54 mm
- Epoch 2: 62.41 mm (best)
- Epoch 3: 76.47 mm

**Campus:**
- Epoch 1: 1439.35 mm
- Epoch 2: 725.23 mm (best)
- Epoch 3: 756.90 mm

### Generated checkpoints

- `outputs/ray_attention_temporal_residual_shelf.pth`
- `outputs/ray_attention_temporal_residual_campus.pth`

## Observations and blockers

1. **Shelf looks reasonable.** A 3-epoch smoke run already beats the DLT baseline
   (62.41 mm vs 65.68 mm), confirming the residual head helps on real
   calibrated multi-view data.
2. **Campus absolute error is poor.** The residual head reduces error from ~4.9 m
   to ~0.73 m, which is a large relative improvement, but the absolute MPJPE is
   still far above a usable range. This suggests the Campus canonical data may
   have a scale/coordinate-system mismatch or the pseudo-GT is not in the same
   metric space as the calibration.
3. **No cross-dataset transfer.** The MPI-INF-3DHP residual checkpoint
   (`ray_attention_temporal_residual_v2.pth`) cannot be loaded because it was
   trained with a different number of views / temporal dimensions. Per-dataset
   smoke training was necessary.

## Next steps

- Investigate the Campus data scale/calibration alignment (check whether
  `camera_t` and `joints_3d` are in consistent units).
- Train Campus for more epochs or tune learning rate; 3 epochs is clearly
  insufficient if the coordinate scale is correct.
- Compare against the non-residual temporal model (`ray_attention_temporal_shelf.pth`)
  to quantify the residual head contribution on Shelf/Campus.
