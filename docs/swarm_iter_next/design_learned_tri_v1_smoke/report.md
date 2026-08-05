# Smoke Training for Learned Gauss-Newton Triangulation v1

## Summary

Created a minimal, self-contained smoke-training script for the learned
triangulation variant of the temporal residual model
(`RayAttentionFusionModelTemporalResidualLearnedTri`).  The script generates a
synthetic multi-view dataset on the fly, runs a short training loop, and saves
a checkpoint.

## Files Created

- `experiments/train_learned_tri_v1_smoke.py`
  - Builds a 4-view circular camera rig.
  - Generates random 3D poses, projects them to 2D, and adds Gaussian noise.
  - Trains the learned Gauss-Newton triangulation model for a few epochs on the
    synthetic data.
  - Saves the best checkpoint to `outputs/learned_tri_v1_smoke.pth`.
- `docs/swarm_iter_next/design_learned_tri_v1_smoke/report.md`
  - This report.

## How to Run / Validate

```bash
conda run -n mf python experiments/train_learned_tri_v1_smoke.py
```

Optional flags:

```bash
conda run -n mf python experiments/train_learned_tri_v1_smoke.py \
    --epochs 3 --batch_size 2 --clip_len 9 --d 32
```

A successful run should print device info, model statistics, and per-epoch
train/validation losses and MPJPE, finishing with a saved checkpoint.

## Smoke Test Results

Example output (2 epochs, batch size 2, CPU):

```
Device: cpu
n_views=4, j=17, clip_len=9, d=32, gn_iters=2, params=47636
Epoch 1: train_loss=0.000557, val_loss=0.000007, val_MPJPE=4.48mm (saved)
Epoch 2: train_loss=0.000021, val_loss=0.000003, val_MPJPE=2.59mm (saved)
Best val MPJPE: 2.59mm -> outputs\learned_tri_v1_smoke.pth
```

The model instantiates correctly, the differentiable Gauss-Newton triangulation
head back-propagates gradients, and the checkpoint is written.  The absolute
MPJPE is low because the synthetic data are well-conditioned and noise is small;
the purpose of this script is to verify training plumbing rather than report
real-world accuracy.

## Expected Impact

- Provides a fast, reproducible way to validate changes to the learned
triangulation head.
- Can be used as a pre-commit sanity check before launching full MPI-INF-3DHP
runs.
- Demonstrates that `RayAttentionFusionModelTemporalResidualLearnedTri` trains
end-to-end without NaNs or shape mismatches.

## Blockers / Notes

- No blockers. The script runs successfully on the `mf` conda environment.
- The current implementation uses a tiny synthetic dataset. For a meaningful
accuracy signal, the existing full training script should be used:
  `experiments/train_ray_attention_temporal_learned_tri_v1_mpiinf3dhp.py`.
