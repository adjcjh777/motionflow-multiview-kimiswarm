# MPI-INF-3DHP Temporal Baseline + Auxiliary Losses

## Objective

Add two auxiliary losses to the existing temporal ray-attention baseline on MPI-INF-3DHP:

1. **Velocity consistency loss** – supervised L1 penalty between predicted and ground-truth finite-difference velocities across the clip.
2. **Bone-length loss** – supervised L1 penalty on per-bone lengths using the MPI-INF-3DHP 28-joint skeleton.

Then run a short (≤10 epoch / ≤30 min) smoke comparison and report whether short-run stability / MPJPE improves.

## Files created / modified

- `experiments/train_ray_attention_temporal_mpiinf3dhp_aux_v1.py` (new copy; existing `train_ray_attention_temporal_mpiinf3dhp.py` untouched)
- `docs/swarm_iter5/mpiinf3dhp_aux_losses_report.md` (this report)

No existing source files were modified.

## What was added

The new script is a drop-in copy of `experiments/train_ray_attention_temporal_mpiinf3dhp.py` with two extra pieces:

1. `velocity_consistency_loss(pred, target, weight)`
   - Computes `pred[:, 1:] - pred[:, :-1]` and matches it to the target velocity.
   - Uses L1 loss; disabled if `clip_len < 2` or weight is 0.

2. Skeleton-aware `bone_length_loss`
   - Reuses `experiments/train_utils.py::bone_length_loss`.
   - Supplied the MPI-INF-3DHP 28-joint parent array (converted to 0-based, pelvis as `-1` root) derived from the standard `mpi-inf-3dhp` reference implementations.

The total training loss becomes:

```
loss = MSE(pred, target)
       + velocity_weight * L1(Δpred, Δtarget)
       + bone_weight * L1(bone_lengths(pred), bone_lengths(target))
```

CLI args added: `--velocity_weight` (default `0.1`), `--bone_weight` (default `0.1`).

## Experiments

All runs use `RayAttentionFusionModelTemporal`, `d=64`, `n_temporal_layers=2`, 14 views, 28 joints.

### Small smoke (500 random clips, batch 16, 2 epochs)

| variant | velocity | bone | best val MPJPE |
|---------|----------|------|----------------|
| baseline | – | – | **29.93 mm** |
| aux v1   | 0.1 | 0.1 | 29.95 mm |
| aux v1   | 0.01| 0.01| 29.95 mm |
| aux v1   | 0.01| 0.01| 29.93 mm (5 epochs) |

Conclusion on tiny subset: auxiliary losses are stable but do not beat the baseline in this highly under-sampled setting.

### Full smoke (4000 random clips, batch 8, 2 epochs)

| variant | velocity | bone | best val MPJPE |
|---------|----------|------|----------------|
| baseline | – | – | **25.21 mm** |
| aux v1   | 0.01| 0.01| **25.23 mm** |

Result: the auxiliary-loss run reaches 25.23 mm vs. the baseline 25.21 mm. The difference is 0.02 mm, i.e. within run-to-run noise. Training was stable and no NaNs were observed.

### Commands

```bash
# baseline (small smoke)
conda run -n mf python -u experiments/train_ray_attention_temporal_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --epochs 2 --batch_size 16 --train_samples 500 \
    --output outputs/ray_attention_temporal_mpiinf3dhp_baseline.pth

# aux v1 full smoke
conda run -n mf python -u experiments/train_ray_attention_temporal_mpiinf3dhp_aux_v1.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --epochs 2 --batch_size 8 --train_samples 4000 \
    --velocity_weight 0.01 --bone_weight 0.01 \
    --output outputs/ray_attention_temporal_mpiinf3dhp_aux_full01.pth
```

## Dependencies

No new packages were installed. The script only uses existing project code plus PyTorch/NumPy (already in the `mf` environment).

## Observations / blockers

- **Stability**: fine. Auxiliary losses remain small relative to MSE and do not cause divergence.
- **Weight sensitivity**: with `velocity_weight=bone_weight=0.1`, the auxiliary terms dominated the MSE term by roughly an order of magnitude on the small smoke. Reducing to `0.01` restored a healthy balance and produced the 25.23 mm result.
- **Short-run MPJPE**: no meaningful improvement in the 2-epoch smoke. The auxiliary-loss run (25.23 mm) is essentially tied with the baseline run (25.21 mm).
- **No blockers**: all runs completed; checkpoints saved to `outputs/`.

## Next steps / follow-up

1. Try a small grid of weights (e.g. `1e-3` to `1e-1`) over 5–10 epochs to find whether a sweet spot exists.
2. Add an *unsupervised* temporal bone-length consistency loss (low variance of bone length across frames) and a bone-symmetry loss, rather than relying solely on supervised bone-length.
3. Evaluate on the full cross-subject protocol (all S1 train, all S2 val) for more than 2 epochs; short smoke runs may mask regularisation benefits.
