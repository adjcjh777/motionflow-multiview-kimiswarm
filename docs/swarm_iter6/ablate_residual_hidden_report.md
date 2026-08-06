# Ablation: residual_hidden size

Trains ``RayAttentionFusionModelTemporalResidual`` with different ``residual_hidden``
sizes and reports cross-subject val MPJPE (mm) and parameter count.

## Setup

- Data: `data/webbridge/mpi_inf_3dhp` (smoke=True)
- Epochs: 2
- Random clips per train seq: 50
- Batch size: 4
- d: 64, n_temporal_layers: 2
- Total elapsed time: 4.8 min

## Results

| residual_hidden | Parameters | Best val MPJPE (mm) | Best epoch | Checkpoint |
|----------------:|-----------:|--------------------:|-----------:|------------|
| 32 | 221,156 | 19.15 | 1 | `outputs\ablate_residual_hidden\residual_h32.pth` |
| 64 | 226,532 | 26.70 | 2 | `outputs\ablate_residual_hidden\residual_h64.pth` |
| 128 | 243,428 | 12.99 | 2 | `outputs\ablate_residual_hidden\residual_h128.pth` |
| 256 | 301,796 | 20.22 | 2 | `outputs\ablate_residual_hidden\residual_h256.pth` |

## Per-epoch history

### residual_hidden=32

| Epoch | Train loss | Val MPJPE (mm) |
|------:|-----------:|---------------:|
| 1 | 0.002943 | 19.15 |
| 2 | 0.000226 | 22.48 |

### residual_hidden=64

| Epoch | Train loss | Val MPJPE (mm) |
|------:|-----------:|---------------:|
| 1 | 0.004757 | 32.60 |
| 2 | 0.000271 | 26.70 |

### residual_hidden=128

| Epoch | Train loss | Val MPJPE (mm) |
|------:|-----------:|---------------:|
| 1 | 0.006318 | 16.06 |
| 2 | 0.000323 | 12.99 |

### residual_hidden=256

| Epoch | Train loss | Val MPJPE (mm) |
|------:|-----------:|---------------:|
| 1 | 0.008601 | 26.54 |
| 2 | 0.000183 | 20.22 |

## Sweet spot

Lowest val MPJPE = 12.99 mm with residual_hidden=128 (243,428 params).

## Notes

This is a smoke-run ablation; absolute MPJPE differs from the fully trained checkpoint, but the relative ranking of ``residual_hidden`` sizes is informative for architecture design.

GPU training was attempted first (`conda run -n mf python ...`), but the RTX 4090 was saturated by other concurrent swarm agents (nvidia-smi showed ~92% utilization and multiple `mf` python processes). To avoid resource conflicts and finish within the 30-minute budget, the run was executed on CPU with `CUDA_VISIBLE_DEVICES=""`. The full ablation completed in 4.8 minutes.
