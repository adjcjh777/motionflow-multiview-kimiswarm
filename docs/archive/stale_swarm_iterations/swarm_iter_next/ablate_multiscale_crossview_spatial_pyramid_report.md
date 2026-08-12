# Ablation / Smoke: Multi-Scale Cross-View Spatial Pyramid

Smoke run of the new spatial-pyramid anchor model.

## Setup

- Data: `data\webbridge\mpi_inf_3dhp\s_01_seq_01_v14_multiview_m_smoke.npz` (smoke=True)
- Frames: 250, Views: 14, Joints: 28, Clip len: 9
- Device: cpu
- Model d=32, scales=[1, 2, 4]
- Anchor trainable parameters: 73,558
- Pyramid trainable parameters: 103,414 (+29,856)
- Epochs: 1, max batches per epoch: 3, batch size: 2

## Smoke training loss

| Epoch | Train loss |
|------:|-----------|
| 1 | 0.005201 |

Total elapsed time: 1.4 s
