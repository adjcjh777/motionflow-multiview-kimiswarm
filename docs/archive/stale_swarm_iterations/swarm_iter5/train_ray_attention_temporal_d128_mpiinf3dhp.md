# Larger Temporal Ray-Attention Model on MPI-INF-3DHP

## Goal

Train a larger variant of `RayAttentionFusionModelTemporal` with **d=128** and
**n_temporal_layers=4** on MPI-INF-3DHP (train S1 Seq1+Seq2, val S2 Seq1) and
see whether cross-subject MPJPE drops below 22 mm with short training.

## What was created

- `experiments/train_ray_attention_temporal_d128_mpiinf3dhp.py`
  - Thin wrapper around the baseline temporal trainer.
  - Defaults: `d=128`, `n_temporal_layers=4`.
  - Includes a monkey-patch of the shared DLT triangulation solver to avoid a
    CUDA illegal-memory-access that the wider/deeper model triggers in
    `torch.linalg.lstsq`.

## Workaround for CUDA `torch.linalg.lstsq` failure

The larger model exposes a PyTorch/CUDA bug in the shared DLT triangulation
routine `motionflow_mv/fusion/ray_attention_model.py::_triangulate_joint`.
Symptoms observed:

- `RuntimeError: CUDA error: unknown error`
- `RuntimeError: CUDA error: an illegal memory access was encountered`

The new training script works around this by monkey-patching
`_triangulate_joint` with a regularised normal-equation solver
(`torch.linalg.solve` on a ridge-regularised 3×3 system), keeping all PyTorch
operations on the GPU and preserving gradients through the triangulation step.
No existing source files were modified.

## Smoke-test results

Command used:

```bash
conda run -n mf python experiments/train_ray_attention_temporal_d128_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --epochs 5 --batch_size 8 --train_samples 1000 \
    --output outputs/ray_attention_temporal_d128_mpiinf3dhp.pth
```

Observed cross-subject MPJPE on S2 Seq1:

| Epoch | train_loss | val_MPJPE |
|------:|-----------:|----------:|
| 1     | 0.000537   | 25.23 mm  |
| 2     | 0.000434   | 25.20 mm (best) |
| 3     | 0.000430   | 25.23 mm  |
| 4     | 0.000427   | 25.22 mm  |
| 5     | 0.000413   | 25.20 mm  |

Best validation MPJPE after 5 epochs: **25.20 mm**.

Training speed: roughly **150 s/epoch** with the above settings on the local
RTX 4090, mostly because the larger temporal transformer is substantially slower
than the baseline d=64 model.  A 5-epoch smoke run exceeds the 10-minute
background-task budget before finishing.

## Conclusion

With short training (5 epochs, ~1000 random clips per train sequence), the
larger temporal model reaches **25.20 mm** cross-subject MPJPE.  This is
comparable to the baseline (d=64, n_temporal_layers=2) after 2 epochs, but it
**does not drop below 22 mm** in this short run.  The model also plateaus
quickly (epochs 2-5 all hover around 25.2 mm), suggesting that simply increasing
capacity without additional tuning is not sufficient to push past the 22 mm
barrier.  Longer training, more data, or a better learning-rate/augmentation
schedule may be needed to exploit the extra capacity.

## Blockers / follow-up

1. **CUDA `torch.linalg.lstsq` instability** – the workaround in the training
   script is only a smoke-test patch.  A proper fix in the shared model code
   should be considered if d=128 models are to be trained routinely.
2. **Throughput** – the d=128 / 4-layer transformer is ~3× slower per epoch
   than the baseline, so a full 30-epoch run would need careful scheduling.
3. **No convergence below 22 mm** in 5-epoch short training; longer runs and
   hyper-parameter tuning are required to judge whether the extra capacity helps.
