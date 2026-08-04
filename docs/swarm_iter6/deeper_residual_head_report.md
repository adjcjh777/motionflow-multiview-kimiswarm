# Swarm Iter 6 — Deeper Residual Refinement Head

## Task

Implement a **deeper residual refinement head** on top of the temporal ray-attention model: stack **3 residual MLP blocks**, where each block predicts a per-joint correction :math:`\Delta X` and adds it to the current 3D estimate. Smoke-train on MPI-INF-3DHP and compare against the original single-block residual head.

## Files added

| File | Purpose |
|------|---------|
| `motionflow_mv/fusion/ray_attention_temporal_residual_v4_model.py` | `RayAttentionFusionModelTemporalResidualV4` — temporal ray-attention model with a configurable N-block residual MLP head. |
| `experiments/train_ray_attention_temporal_residual_v4_mpiinf3dhp.py` | Training script that exposes `--n_residual_blocks` (1 or 3) and reuses the existing data loaders. |

## Architecture

`RayAttentionFusionModelTemporalResidualV4` inherits from `RayAttentionFusionModelTemporal` (same base as the v2 residual model). After the weighted-DLT triangulation, the new head:

1. Pools the temporal per-view features once per joint: `(B*T, J, d)`.
2. Feeds the concatenation `[pooled_feature, current_3d_estimate]` into each residual block.
3. Each block is a 2-layer ReLU MLP that outputs a 3D residual, which is added back to the current estimate.
4. Repeats for `n_residual_blocks` blocks (default 3).

Parameter count (smoke config: `d=64`, `residual_hidden=128`):

- 1 block: 243,428 params
- 3 blocks: 294,634 params

## Smoke-training protocol

Because GPU is shared, training was kept short:

- Dataset: MPI-INF-3DHP smoke `.npz` files
  - Train: `s_01_seq_01_v14_multiview_m_smoke.npz`, `s_01_seq_02_v14_multiview_m_smoke.npz`
  - Val: `s_02_seq_01_v14_multiview_m_smoke.npz`
- `--clip_len 9 --batch_size 4 --train_samples 100 --epochs 3`
- Same seed and augmentation for both runs.

## Results

| Model | Residual blocks | Best val MPJPE (mm) | Checkpoint |
|-------|-----------------|---------------------|------------|
| v2-style | 1 | 16.24 | `outputs/ray_attention_temporal_residual_v2_smoke.pth` |
| v4 deep | 3 | **13.24** | `outputs/ray_attention_temporal_residual_v4_smoke.pth` |

Raw console output:

**3-block (v4):**

```text
Device: cuda
n_views=14, j=28, clip_len=9, d=64, residual_hidden=128, n_residual_blocks=3
Model params: 294634
Epoch 1: train_loss=0.008376, val_MPJPE=16.27mm (saved)
Epoch 2: train_loss=0.000185, val_MPJPE=13.24mm (saved)
Epoch 3: train_loss=0.000160, val_MPJPE=20.59mm
Best val MPJPE: 13.24mm -> outputs\ray_attention_temporal_residual_v4_smoke.pth
```

**1-block (v2):**

```text
Device: cuda
n_views=14, j=28, clip_len=9, d=64, residual_hidden=128, n_residual_blocks=1
Model params: 243428
Epoch 1: train_loss=0.001604, val_MPJPE=18.84mm (saved)
Epoch 2: train_loss=0.000101, val_MPJPE=16.24mm (saved)
Epoch 3: train_loss=0.000113, val_MPJPE=20.89mm
Best val MPJPE: 16.24mm -> outputs\ray_attention_temporal_residual_v2_smoke.pth
```

## Interpretation

On the small smoke split, the 3-block deep residual head improved over the single-block head by about **3.0 mm** (16.24 mm → 13.24 mm). Both runs show signs of overfitting after epoch 2, which is expected given the tiny 3-epoch smoke schedule. A full training run is needed before drawing strong conclusions.

## How to reproduce / extend

Train the deep head on full MPI-INF-3DHP:

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_v4_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --batch_size 8 --train_samples 4000 --epochs 30 \
    --n_residual_blocks 3 \
    --output outputs/ray_attention_temporal_residual_v4_mpiinf3dhp.pth
```

Compare with the single-block baseline by setting `--n_residual_blocks 1`.

## Blockers / caveats

- None for code or environment.
- The comparison is a **smoke test only** (3 epochs on small smoke files). Overfitting is visible; full training is required for a rigorous comparison.
- Both checkpoints were saved on the same validation set; numbers should not be compared to the claimed ~13.84 mm on the full validation protocol.
