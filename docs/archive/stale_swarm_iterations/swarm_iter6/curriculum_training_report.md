# Curriculum Training for Temporal Residual Model

## Goal
Implement curriculum training for the top-performing residual refinement head on
top of the temporal ray-attention model.  The idea is to progressively increase
the difficulty of training augmentations (occlusion/dropout, pixel noise, and
outliers) so that the model first learns a stable correction on clean inputs and
later adapts to noisy/occluded views.

## Files added/modified
- `experiments/train_ray_attention_temporal_residual_mpiinf3dhp_curriculum.py`
  - New training script extending `train_ray_attention_temporal_residual_mpiinf3dhp.py`.
  - Adds a curriculum that ramps augmentation difficulty over
    `--curriculum_epochs` epochs.
- `docs/swarm_iter6/curriculum_training_report.md`
  - This report.

## Curriculum design
For each of the four augmentation knobs we start from a "clean'' value and ramp
to a ``hard'' value:

| Augmentation | Start (easy) | End (hard) |
|---|---|---|
| Pixel noise std (px) | 0.0 | 1.5 |
| Confidence dropout (occlusion) | 0.0 | 0.25 |
| Outlier replacement rate | 0.0 | 0.03 |
| Outlier displacement half-range (px) | 50.0 | 200.0 |

Ramping is linear over `curriculum_epochs` and the end value is held for the
remaining epochs.  All values are exposed as command-line flags so the schedule
can be tuned without editing code.

## Smoke-run results

Validation was done on a 500-frame smoke subset of S2 Seq1
(`data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz`).

### 10-epoch curriculum smoke run

```bash
D:/anaconda3/envs/mf/python.exe -u experiments/train_ray_attention_temporal_residual_mpiinf3dhp_curriculum.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --d 64 --n_temporal_layers 2 --residual_hidden 128 \
    --epochs 10 --curriculum_epochs 5 --batch_size 4 --train_samples 200 \
    --seed 42 \
    --output outputs/ray_attention_temporal_residual_curriculum_10ep.pth
```

| Epoch | noise_std | dropout | outlier_rate | outlier_scale | train_loss | val_MPJPE |
|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.0 | 0.0 | 50.0 | 0.000589 | 23.71 mm (saved) |
| 2 | 0.375 | 0.0625 | 0.0075 | 87.5 | 0.000327 | 19.99 mm (saved) |
| 3 | 0.75 | 0.125 | 0.015 | 125.0 | 0.000073 | 21.86 mm |
| 4 | 1.125 | 0.1875 | 0.0225 | 162.5 | 0.000063 | 22.99 mm |
| 5 | 1.5 | 0.25 | 0.03 | 200.0 | 0.000049 | **19.14 mm (saved)** |
| 6 | 1.5 | 0.25 | 0.03 | 200.0 | 0.000046 | 21.63 mm |
| 7 | 1.5 | 0.25 | 0.03 | 200.0 | 0.000038 | 20.73 mm |
| 8 | 1.5 | 0.25 | 0.03 | 200.0 | 0.000031 | 19.61 mm |
| 9 | 1.5 | 0.25 | 0.03 | 200.0 | 0.000035 | 19.63 mm |
| 10 | 1.5 | 0.25 | 0.03 | 200.0 | 0.000031 | 20.54 mm |

**Best val MPJPE: 19.14 mm** at epoch 5.

### Baseline 10-epoch smoke run (same data, default constant augmentation)

```bash
D:/anaconda3/envs/mf/python.exe -u experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --d 64 --n_temporal_layers 2 --residual_hidden 128 \
    --epochs 10 --batch_size 4 --train_samples 200 --seed 42 \
    --output outputs/ray_attention_temporal_residual_baseline_smoke_10ep_v2.pth
```

| Epoch | train_loss | val_MPJPE |
|---|---|---|
| 1 | 0.000845 | 21.13 mm (saved) |
| 2 | 0.000097 | 20.30 mm (saved) |
| 3 | 0.000069 | 20.29 mm (saved) |
| 4 | 0.000061 | 25.69 mm |
| 5 | 0.000041 | 21.66 mm |
| 6 | 0.000043 | 20.39 mm |
| 7 | 0.000038 | 19.95 mm (saved) |
| 8 | 0.000029 | 21.83 mm |
| 9 | 0.000033 | **19.07 mm (saved)** |
| 10 | 0.000026 | 20.82 mm |

**Best val MPJPE: 19.07 mm** at epoch 9.

## Interpretation
- The curriculum schedule works as intended: noise/dropout/outlier difficulty ramps
  linearly and then holds at the final values.
- On the small smoke validation set, the curriculum-trained model reaches a
  comparable best MPJPE to the baseline with constant augmentation (19.14 mm vs.
  19.07 mm), with the curriculum best occurring earlier (epoch 5 vs. epoch 9).
- The smoke dataset is too small and the margin too small to conclude that
  curriculum improves the top-performing residual checkpoint on the full
  MPI-INF-3DHP S2 Seq1 validation set; a full-length run on full training data
  is needed for that.
- One baseline run did encounter a CUDA illegal memory access in the DLT solver
  during epoch-2 validation, likely triggered by a degenerate configuration of
  views after dropout/outliers.  The curriculum run was stable throughout,
  suggesting the gradual ramp may improve training stability.

## How to run a full curriculum training
```bash
D:/anaconda3/envs/mf/python.exe -u experiments/train_ray_attention_temporal_residual_mpiinf3dhp_curriculum.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --n_temporal_layers 2 --residual_hidden 128 \
    --epochs 30 --curriculum_epochs 10 --batch_size 8 --train_samples 4000 \
    --seed 42 \
    --output outputs/ray_attention_temporal_residual_curriculum_full30.pth
```

## Next steps / follow-up ideas
- Run the full 30-epoch curriculum on full MPI-INF-3DHP training data and
  compare MPJPE against the existing `outputs/ray_attention_temporal_residual_v2.pth`.
- Run a controlled robustness evaluation (varying occlusion/noise levels) to see
  if the curriculum model is actually more robust.
- Try cosine ramping or staged step schedules if linear ramping proves too
  aggressive/too slow.
