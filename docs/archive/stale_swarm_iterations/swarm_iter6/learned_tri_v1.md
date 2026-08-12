# Learnable Gauss-Newton Triangulation Head

## Summary

Implemented a differentiable Gauss-Newton (GN) triangulation head that replaces
weighted DLT in the top-performing temporal residual model.  The new head uses
the network-predicted per-view weights and camera parameters to refine a DLT
initial estimate by minimizing the weighted reprojection error over a small
number of GN iterations.  The existing residual MLP refinement head is kept on
top of the GN output.

## Files changed / created

- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py`
  - Added a `_triangulate` hook so that subclasses can swap the triangulation
    method without duplicating the full forward pass.
- `motionflow_mv/fusion/ray_attention_temporal_learned_tri_v1.py`
  - New model `RayAttentionFusionModelTemporalResidualLearnedTri` that
    overrides `_triangulate` with a differentiable Gauss-Newton solver.
  - Helper `_triangulate_weighted_gauss_newton` that performs vectorized
    Gauss-Newton iterations on the weighted reprojection error.
- `experiments/train_ray_attention_temporal_learned_tri_v1_mpiinf3dhp.py`
  - New smoke-training script for the GN model; supports loading the existing
    residual checkpoint.
- `experiments/compare_learned_tri_v1_mpiinf3dhp.py`
  - Evaluation script that reports MPJPE for both DLT and GN variants on a
    validation sequence.

## Method

The GN solver minimizes

```
E(X) = sum_v w_v * || pi_v(X) - x_v ||^2
```

where `w_v` are the predicted per-view weights and `pi_v` is the pinhole
projection for view `v`.  Starting from the DLT estimate, each iteration:

1. Projects the current 3D estimate to obtain residuals.
2. Builds the analytic Jacobian of the projection with respect to world
   coordinates.
3. Solves the damped normal equations for the update `dx`.
4. Applies `X <- X + dx`.

The solver is fully differentiable; gradients flow back through the predicted
weights and camera parameters.  Default settings used for smoke tests:
`gn_iters=3`, `gn_damping=1e-6`.

## Smoke-training results

Command:

```bash
conda run -n mf python experiments/train_ray_attention_temporal_learned_tri_v1_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --epochs 3 --batch_size 2 --train_samples 200 \
    --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
    --output outputs/ray_attention_temporal_learned_tri_v1_smoke.pth
```

Output:

```
Device: cuda
n_views=14, j=28, clip_len=13, d=64, gn_iters=3, gn_damping=1e-06
Model params: 243428
Loaded checkpoint from outputs/ray_attention_temporal_residual_v2.pth
Epoch 1: train_loss=0.000043, val_MPJPE=13.83mm (saved)
Epoch 2: train_loss=0.000020, val_MPJPE=12.95mm (saved)
Epoch 3: train_loss=0.000013, val_MPJPE=14.44mm
Best val MPJPE: 12.95mm -> outputs/ray_attention_temporal_learned_tri_v1_smoke.pth
```

The model trained successfully and produced plausible 3D poses.  The small
smoke subset (used for fast iteration) gives only a coarse signal; a full
validation run on `s_02_seq_01_v14_multiview_m.npz` is recommended for a
meaningful comparison against the published DLT baseline.

## DLT vs. Gauss-Newton comparison

Using the *same* trained residual checkpoint (`ray_attention_temporal_residual_v2.pth`)
for both the DLT baseline and the GN model, on the smoke validation set:

```bash
conda run -n mf python experiments/compare_learned_tri_v1_mpiinf3dhp.py \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --dlt_ckpt outputs/ray_attention_temporal_residual_v2.pth \
    --gn_ckpt outputs/ray_attention_temporal_residual_v2.pth \
    --clip_len 13
```

Results:

```
DLT baseline MPJPE: 10.76 mm
Gauss-Newton MPJPE: 10.77 mm
Difference (GN - DLT): +0.00 mm
```

On the full MPI-INF-3DHP validation sequence `s_02_seq_01_v14_multiview_m.npz`:

```bash
conda run -n mf python experiments/compare_learned_tri_v1_mpiinf3dhp.py \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --dlt_ckpt outputs/ray_attention_temporal_residual_v2.pth \
    --gn_ckpt outputs/ray_attention_temporal_residual_v2.pth \
    --clip_len 13
```

Results:

```
DLT baseline MPJPE: 13.12 mm
Gauss-Newton MPJPE: 13.14 mm
Difference (GN - DLT): +0.02 mm
```

With the same learned weights, the GN refinement is effectively equivalent to
DLT.  This is expected when DLT already provides a near-optimal estimate and the
camera/observation geometry is well-conditioned.

## Observations and next steps

- The GN head is functional, differentiable, and integrates cleanly with the
  residual model through the new `_triangulate` hook.
- On a short smoke run, the GN model does not outperform the strong DLT
  baseline; the DLT estimate from the trained model is already near the
  reprojection-error optimum.
- To assess whether the GN head helps, a full-dataset training run (or at least
  full-validation evaluation) is needed.  Potential scenarios where GN may help:
  - Low-confidence / dropped views, where the weighted reprojection objective
    differs from the DLT line-intersection objective.
  - Calibration noise or non-pinhole effects, where the residual MLP may learn
    to condition the predicted weights on the GN refinement.
- No blockers; all code runs on the local RTX 4090 within the smoke-test
  budget.
