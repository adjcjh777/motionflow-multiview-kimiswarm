# Mixed-Dataset Temporal Training

## Goal
Train the temporal ray-attention fusion model on **MPI-INF-3DHP + Human3.6M**
clips in a single run, despite the two datasets having different numbers of
views and joints.

## What was built

| File | Purpose |
|------|---------|
| `motionflow_mv/fusion/ray_attention_temporal_model_mixed_v1.py` | Shared temporal backbone + per-dataset output heads |
| `experiments/train_ray_attention_temporal_mixed_v1.py` | Mixed-dataset training script with padded canonical loaders |
| `data/h36m_hf/s_01_subset_5k_m.npz` | 5k-frame subset of H36M S1 (meters) used for the smoke run |
| `outputs/ray_attention_temporal_mixed_v1.pth` | Best checkpoint from the smoke run |

## Approach

1. **Common grid.** MPI-INF-3DHP has 14 views / 28 joints; Human3.6M has 4 views /
   17 joints. All clips are padded to the larger MPI-INF-3DHP grid `(14, 28)`.
   Dummy views/joints are zero-filled and ignored during loss.
2. **Dataset-specific heads.** The shared temporal backbone is reused from
   `RayAttentionFusionModelTemporal`. After temporal attention, two lightweight
   heads project to the real view/joint set for each dataset:
   * `mpi`: 14 views, 28 joints
   * `h36m`: 4 views, 17 joints
3. **Loss masking.** The model returns a joint mask so that H36M samples are only
   supervised on their 17 real joints.
4. **Minimal changes.** No existing working files were modified; the original
   temporal model is treated as a feature extractor.

## Smoke test command

```bash
conda run -n mf python experiments/train_ray_attention_temporal_mixed_v1.py \
    --mpi_train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
    --h36m_train data/h36m_hf/s_01_subset_5k_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --epochs 1 --d 32 --train_samples 50 --batch_size 4 \
    --output outputs/ray_attention_temporal_mixed_v1.pth
```

The script also accepts multiple MPI train files, e.g.:

```bash
--mpi_train \
    data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
    data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz
```

## Result

```text
Device: cuda
Model params: 80435
Epoch 1: train_loss=0.001278, val_MPJPE=25.22mm (saved)
Best val MPJPE: 25.22mm -> outputs\ray_attention_temporal_mixed_v1.pth
```

The smoke run trained on MPI-INF-3DHP S1/Seq1 plus a 5k-frame Human3.6M S1
subset and validated on MPI-INF-3DHP S2/Seq1, reaching **25.22 mm MPJPE** after
a single epoch. This matches the baseline MPI-INF-3DHP-only smoke run reported
in the project state (25.25 mm after 2 epochs), while also consuming the H36M
mixed data.

## Notes / blockers

* No new Python dependencies were required; only NumPy and PyTorch were used.
* The full Human3.6M subject-1 file (`s_01_acts_*_multiview_m.npz`) is ~62k
  frames. A 5k-frame subset was created for the smoke test to keep runtime under
  the 30-minute limit.
* Dummy padded views use identity intrinsics/extrinsics and zero observations,
  with zero confidence; the model learns to ignore them.
* Larger 2-epoch runs with both MPI S1 sequences were attempted, but GPU
  contention with other swarm agents caused them to hang/timeout. The single
  epoch run above completed cleanly and demonstrates the mixed-dataset
  mechanism.
