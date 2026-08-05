# Ablation / Smoke: Spatial Feature Pyramid

CPU-only smoke run of the new ``SpatialFeaturePyramid`` module.

## Setup

- Data: `data\webbridge\mpi_inf_3dhp\s_01_seq_01_v14_multiview_m_smoke.npz` (smoke=True)
- Frames: 250, Views: 14, Joints: 28, Clip len: 5
- Model d=32, num_scales=3
- Trainable parameters: 14,595
- Epochs: 2, max batches per epoch: 5, batch size: 2

## Pyramid feature-map shapes

Input clip shape: `(1, 5, 14, 28, 3)` (B, T, V, J, 3).

Per-scale conv output shapes (before upsampling) and final upsampled target:

| Scale | Shape (N, C_out, target_J) | Upsampled length |
|------:|----------------------------|-----------------|
| 1 | `(70, 32, 28)` | 28 |
| 2 | `(70, 32, 14)` | 28 |
| 3 | `(70, 32, 7)` | 28 |

## Smoke training loss

| Epoch | Train loss |
|------:|-----------|
| 1 | 920.551495 |
| 2 | 187.329149 |

Total elapsed time: 1.7 s
