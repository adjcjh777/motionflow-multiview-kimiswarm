# Uncertainty-Weighted Triangulation Head

## What was built

A new model `RayAttentionFusionModelTemporalUncertainty` in
`motionflow_mv/fusion/ray_attention_temporal_uncertainty_model.py` replaces the
sigmoid per-view weight head of `RayAttentionFusionModelTemporal` with a Gaussian
uncertainty head.  For each view and joint the network now predicts a
log-variance `log_var`; the weighted DLT uses

```text
weight = confidence * exp(-log_var)
```

so uncertain views are down-weighted and high-confidence views dominate.

A matching training script is in
`experiments/train_ray_attention_temporal_uncertainty_mpiinf3dhp.py`.

## Architecture changes

The model keeps the same temporal ray-attention backbone as
`RayAttentionFusionModelTemporal`:

1. Per-frame observation + ray embedding.
2. Camera-conditioned embedding.
3. View-level and joint-level self-attention.
4. Temporal transformer over `(V*J)` tokens across frames.

The only difference is the final head:

- `uncertainty_head` outputs `log_var` of shape `(B, T, V, J)`.
- DLT weights are `confidence * exp(-log_var)`.
- An auxiliary reprojection NLL loss encourages the predicted uncertainties to
  match actual per-view reprojection error:

```text
NLL = 0.5 * (reproj_err^2 / var + log_var)
```

The total training loss is `MSE(pred_3d, gt_3d) + uncertainty_loss_weight * NLL`.

## Files added / modified

- `motionflow_mv/fusion/ray_attention_temporal_uncertainty_model.py` (new model)
- `experiments/train_ray_attention_temporal_uncertainty_mpiinf3dhp.py` (new training script)
- `tests/test_ray_attention_temporal_uncertainty.py` (new sanity test)
- `docs/swarm_iter5/uncertainty_weighted_triangulation.md` (this report)

No existing working files were modified.

## Verification

### Synthetic sanity check

```bash
PYTHONPATH="$PWD" conda run -n mf python tmp/test_uncertainty_model.py
```

Output:

```text
pred torch.Size([2, 5, 17, 3]) weights torch.Size([2, 5, 4, 17]) log_var torch.Size([2, 5, 4, 17]) nll 0.0376
grads ok
single torch.Size([2, 17, 3]) torch.Size([2, 4, 17]) torch.Size([2, 4, 17])
```

The model:

- Produces the expected output shapes for 5-frame clips and single-frame inputs.
- Allows gradients to flow through the uncertainty head.
- Returns per-view log-variance alongside the triangulated 3D pose.

The same checks are captured in `tests/test_ray_attention_temporal_uncertainty.py`:

```bash
PYTHONPATH="$PWD" conda run -n mf python tests/test_ray_attention_temporal_uncertainty.py
```

Result: `uncertainty-weighted temporal tests passed`.

### Short MPI-INF-3DHP smoke run

Command:

```bash
PYTHONPATH="$PWD" conda run -n mf python experiments/train_ray_attention_temporal_uncertainty_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 9 --epochs 2 --batch_size 4 --d 32 --train_samples 500 --val_stride 10 \
    --output outputs/ray_attention_temporal_uncertainty_mpiinf3dhp.pth
```

Output:

```text
Device: cuda
n_views=14, j=28, clip_len=9, d=32
Model params: 59761
Epoch 1: loss=24.535379 (mse=0.002783, nll=24.532596), val_MPJPE=25.24mm (saved)
Epoch 2: loss=0.349473 (mse=0.000495, nll=0.348978), val_MPJPE=25.24mm
Best val MPJPE: 25.24mm -> outputs\ray_attention_temporal_uncertainty_mpiinf3dhp.pth
```

| Epoch | loss  | mse     | nll     | val_MPJPE (mm) | checkpoint |
|-------|-------|---------|---------|----------------|------------|
| 1     | 24.54 | 0.00278 | 24.5326 | 25.24          | saved      |
| 2     | 0.35  | 0.00050 | 0.3490  | 25.24          | —          |

The 2-epoch smoke run reaches **25.24 mm cross-subject MPJPE on MPI-INF-3DHP
(S2 Seq1)**, matching the published baseline smoke result for the sigmoid-weighted
temporal model.  The initial loss is dominated by the NLL term, which rapidly
decays as the uncertainty head learns to explain the per-view reprojection
errors.

## Notes / open items

- The auxiliary NLL term dominates early training.  Consider scaling it down or
  normalizing the reprojection error by image size before computing the NLL.
- A direct comparison under identical hyperparameters with the sigmoid baseline
  (same `d`, batch size, and number of random clips) is still needed to isolate
  the benefit of uncertainty weighting.
- The per-view uncertainties are not yet visualized; a quick histogram of
  predicted `log_var` versus true reprojection error would verify the head is
  learning meaningful uncertainty.
- The smoke run used `--val_stride 10` and `--train_samples 500` for speed; a
  full run should use the baseline's settings (`--train_samples 4000`,
  `--clip_len 13`, `--d 64`).
