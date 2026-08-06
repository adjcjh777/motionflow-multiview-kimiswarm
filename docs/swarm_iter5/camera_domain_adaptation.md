# Camera / Domain Adaptation for Temporal Ray-Attention Fusion

## Goal
Add a lightweight camera-domain MLP to the existing temporal ray-attention model so it can adapt to **mixed camera rigs** (different intrinsics/extrinsics per sample) within the same training batch.

## What was implemented

### New model: `motionflow_mv/fusion/ray_attention_temporal_model_domain.py`
* New class `RayAttentionFusionModelTemporalDomain` is a thin fork of `RayAttentionFusionModelTemporal`.
* It keeps the same per-frame encoder, temporal transformer, and weighted DLT triangulation.
* It adds a **camera-domain MLP**:
  * Input: flattened per-view camera parameters `K (9) + R (9) + t (3) = 21` floats.
  * Architecture: `Linear(21, d) -> ReLU -> Linear(d, d)`.
  * Output: a per-view domain embedding of size `d` that is added to the per-frame features.
* The model already accepts per-sample `(K, R, t)` tensors of shape `(B, V, 3, 3)`, `(B, V, 3, 3)`, `(B, V, 3)`, so mixed camera rigs are supported end-to-end.
* A small defensive change was added inside the model: predicted DLT weights are clamped to a minimum of `1e-4` before triangulation, preventing a degenerate least-squares problem when a joint has zero or near-zero weight across all views (which can happen with heavy dropout augmentation or extreme outlier views).

### New training script: `experiments/train_ray_attention_temporal_mpiinf3dhp_domain.py`
* Fork of the baseline temporal training script.
* Uses `RayAttentionFusionModelTemporalDomain`.
* Loads multiple `.npz` sequences, each with its own camera rig; `ConcatDataset` + per-sample `K/R/t` collation produces batches with mixed rigs.
* Same augmentation and MPJPE evaluation as the baseline.

### Test
* `tests/test_ray_attention_temporal_domain.py` checks forward/backward passes for:
  * standard temporal input,
  * single-frame (4D) input,
  * per-sample (mixed) camera rigs.

All three tests pass.

## How to run (smoke test)

```bash
conda run -n mf python experiments/train_ray_attention_temporal_mpiinf3dhp_domain.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --epochs 3 --batch_size 8 --train_samples 500 --d 32 \
    --output outputs/ray_attention_temporal_mpiinf3dhp_domain.pth
```

## Results

### Smoke run (3 epochs, mixed camera rigs)
A 3-epoch smoke run with mixed camera rigs (S1 Seq1 + Seq2 for training, S2 Seq1 for validation) completed successfully:

```bash
conda run -n mf python experiments/train_ray_attention_temporal_mpiinf3dhp_domain.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --epochs 3 --batch_size 8 --train_samples 500 --d 32 \
    --output outputs/ray_attention_temporal_mpiinf3dhp_domain.pth
```

Output:

```text
Device: cuda
n_views=14, j=28, clip_len=13, d=32
Model params: 61521
Epoch 1: train_loss=0.002049, val_MPJPE=25.23mm (saved)
Epoch 2: train_loss=0.000428, val_MPJPE=25.23mm (saved)
Epoch 3: train_loss=0.000434, val_MPJPE=25.22mm (saved)
Best val MPJPE: 25.22mm -> outputs\ray_attention_temporal_mpiinf3dhp_domain.pth
```

This is on par with the baseline temporal smoke run (25.25 mm after 2 epochs), while training on **mixed camera rigs** and with the extra domain-adaptation MLP. The run took well under 10 minutes on the RTX 4090.

### Blocker / follow-up
A run with the full `d=64` setting and `train_samples=500` repeatedly failed during the first evaluation with a CUDA illegal-memory-access inside `torch.linalg.lstsq`, and later hung when launched with a 10-minute background timeout. The same configuration with `d=32` runs reliably, so the problem appears to be related to the larger `d` setting rather than the mixed-rig logic itself. Suggested follow-ups:
1. Profile the `d=64` run with `torch.profiler` / `CUDA_LAUNCH_BLOCKING=1` to see whether the failure is inside `torch.linalg.lstsq`, in the temporal transformer, or due to a tensor shape mismatch introduced by the added domain MLP.
2. Reduce the domain MLP output to `d//2` and project back to `d`, keeping the added capacity lighter and closer to the baseline's parameter budget.
3. Add gradient clipping and/or weight decay to rule out gradient instability when `d=64`.

## Files touched
* `motionflow_mv/fusion/ray_attention_temporal_model_domain.py` — new camera-domain temporal model.
* `experiments/train_ray_attention_temporal_mpiinf3dhp_domain.py` — new training script for mixed camera rigs.
* `tests/test_ray_attention_temporal_domain.py` — sanity tests for the new model.
* `docs/swarm_iter5/camera_domain_adaptation.md` — this report.

No new Python dependencies were introduced.
