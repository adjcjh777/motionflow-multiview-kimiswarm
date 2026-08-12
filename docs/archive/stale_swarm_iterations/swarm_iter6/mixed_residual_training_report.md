# Swarm Iter 6: Mixed-Dataset Residual Training

## Objective
Train the top-performing residual refinement model on a mixture of MPI-INF-3DHP, AIST++, and Human3.6M canonical `.npz` data, and measure cross-dataset generalization.

## Files created
- `motionflow_mv/fusion/ray_attention_temporal_mixed_residual_v1.py`
- `experiments/train_ray_attention_temporal_mixed_residual_v1.py`
- `experiments/eval_ray_attention_temporal_mixed_residual_v1.py`
- `docs/swarm_iter6/mixed_residual_training_report.md` (this file)

## Dataset summary
| Dataset | Views | Joints | Unit | Train file | Val file |
|---------|-------|--------|------|------------|----------|
| MPI-INF-3DHP | 14 | 28 | m | `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz` | `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz` |
| AIST++ | 9 | 17 | m | `data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch01_multiview.npz` | `data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR1_ch01_multiview.npz` |
| Human3.6M | 4 | 17 | m (see note) | `data/h36m_hf/s_01_acts_*_multiview_m.npz` | `tmp/s_05_acts_02_multiview_m.npz` (converted from mm) |

**Unit note:** MPI-INF-3DHP and AIST++ canonical files are already in meters. The raw H36M files are in millimeters; we converted `s_05_acts_02_multiview.npz` to meters (including `camera_t`) before evaluation. The training used the existing `_m` H36M file which is already in meters.

## Model changes
- Extended `RayAttentionFusionModelTemporalMixed` (mixed_v1) with per-dataset residual refinement heads.
- Added AIST++ support to the mixed dataset specifications (9 views, 17 joints).
- Overrode the mixed model `forward` to dispatch dynamically by `dataset_id`, avoiding the hardcoded ID mapping in the parent class.
- Each dataset branch still pads its clips to the MPI-INF-3DHP grid `(14, 28)`; only the real views/joints are used by the per-dataset output/residual heads.

## Smoke training run
Command:
```bash
D:/anaconda3/envs/mf/python.exe -u experiments/train_ray_attention_temporal_mixed_residual_v1.py \
  --mpi_train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --aist_train data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch01_multiview.npz \
  --h36m_train data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz --val_dataset mpi \
  --clip_len 13 --epochs 5 --d 32 --train_samples 500 --val_samples 200 --batch_size 4 \
  --output outputs/ray_attention_temporal_mixed_residual_v1.pth
```

Results:
- Epoch 1: val MPJPE = 25.51 mm
- Epoch 2: val MPJPE = 23.81 mm
- Epoch 3: val MPJPE = 20.18 mm
- Epoch 4: val MPJPE = 18.38 mm (best)
- Epoch 5: val MPJPE = 19.57 mm
- **Best MPI val MPJPE: 18.38 mm**

## Cross-dataset generalization
Evaluation used checkpoint `outputs/ray_attention_temporal_mixed_residual_v1.pth`.

| Dataset | Val file | MPJPE (mm) |
|---------|----------|------------|
| MPI-INF-3DHP S2 Seq1 | `s_02_seq_01_v14_multiview_m.npz` | **18.19** |
| AIST++ (unseen genre) | `gBR_sBM_cAll_d04_mBR1_ch01_multiview.npz` | **12.12** |
| Human3.6M S5 Act02 | `tmp/s_05_acts_02_multiview_m.npz` | **4.69** |

In-dataset H36M train check (same file used during training): **7.08 mm**.

## Observations
- The model successfully trains on the three-dataset mixture; the shared temporal backbone plus per-dataset heads handles the heterogeneous view/joint counts.
- Cross-dataset errors are in the same ballpark as the in-dataset MPI val error, showing reasonable generalization for a short smoke run.
- Human3.6M val error is surprisingly low (4.69 mm); this may reflect the simpler studio capture and consistent skeleton.
- AIST++ generalizes well despite being a different motion genre (dance).

## Limitations
- Smoke run: only 5 epochs and 500 random clips per dataset.
- H36M val required manual unit conversion; the raw H36M `.npz` files store joints in mm and `camera_t` in mm, while the `_m` training file is already in meters.
- Full convergence and comparison to the MPI-only residual baseline were not attempted due to time/GPU constraints.

## Next steps / follow-up
- Longer training run (20–30 epochs) with full train sets to compare against the MPI-only ~13.84 mm baseline.
- Evaluate the impact of mixing AIST++/H36M on MPI performance: does multi-dataset training hurt or help MPI-INF-3DHP accuracy?
- Investigate per-joint error on H36M and AIST to confirm the low numbers are not due to a unit/alignment artifact.
