# Reprojection-Gated Residual Refinement Head

## 1. Technical report

The current best model, ``RayAttentionFusionModelTemporalResidual``
(`motionflow_mv/fusion/ray_attention_temporal_residual_model.py`), triangulates
a raw 3D pose with weighted DLT and then refines it with a small residual MLP.
By default the residual correction is added unconditionally, which can over-fit
to noisy views or amplify errors when the raw triangulation is already
geometrically consistent.  We add a **reprojection-error gate** that scales the
per-joint residual by a function of the current reprojection-error statistics.

### Key design decisions

1. **Per-joint reprojection summary.** After each residual iteration we project
the current 3D estimate back into every view and compute mean, standard
deviation, maximum, and inlier fraction of the reprojection errors.  This 4-D
summary captures both the magnitude and the consistency of the current estimate
across views.

2. **Gated residual correction.** The residual MLP still predicts a raw
`\Delta X`.  A second tiny MLP consumes the concatenation of the residual input
feature and the reprojection summary, and outputs a scalar in `[0, 1]` per
joint.  The final correction is `gate * \Delta X`, so joints with low
reprojection error receive little or no refinement, while joints with high error
are allowed a larger update.

3. **Plug-in to existing residual head.** The gate is an optional flag
(`use_reproj_gate`) inside the same `RayAttentionFusionModelTemporalResidual`
class.  No other model code changes, preserving the 11.17 mm baseline path when
`use_reproj_gate=False`.

4. **Auxiliary reprojection loss.** The full trainer also enables
`--reproj_weight 0.1` to explicitly supervise the 3D output to reproject well
into the input views, further aligning the residual head with the multi-view
geometry.

### Equations

Let `X \in \mathbb{R}^{J \times 3}` be the current 3D estimate and
`P_v = K_v [R_v | t_v]` the projection matrix for view `v`.  The per-view
reprojection error is

```
e_{v,j} = || \pi(P_v, X_j) - x_{v,j} ||_2
```

The 4-D summary per joint is

```
s_j = [ mean_v(e_{v,j}), std_v(e_{v,j}), max_v(e_{v,j}), frac_v(e_{v,j} < \tau) ]
```

with `\tau = 10` pixels.  The gated residual is

```
\Delta X_j = MLP_{gate}([f_j \| X_j \| s_j]) \cdot MLP_{res}([f_j \| X_j])
```

where `f_j` is the pooled temporal feature for joint `j` and `\|` denotes
concatenation.

### Expected impact

- **Robustness to noisy/occluded views:** the gate down-weights residual
  updates when all views already agree, preventing over-correction.
- **Better convergence:** gating supplies an explicit geometric feedback signal
  to the residual MLP, which should reduce the chance of the residual head
  drifting away from the observed rays.
- **Minimal overhead:** the gate adds only one small MLP (`d+3+4 -> 128 -> 1`)
  plus the cheap reprojection summary; training cost is essentially unchanged.

### Relevant files

- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py` —
  `use_reproj_gate` flag, `_reprojection_error_summary`, and the gated residual
  block.
- `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py` — base
  trainer that already exposes `--use_reproj_gate` and `--reproj_weight`.
- `experiments/train_ray_attention_temporal_residual_reprojgate_full_mpiinf3dhp.py`
  — new full 30-epoch wrapper with the cross-subject MPI-INF-3DHP split.

## 2. Implementation plan

1. Verify that `RayAttentionFusionModelTemporalResidual(...,
   use_reproj_gate=True)` instantiates the gate correctly and that gradients
   flow through `_reprojection_error_summary`.
2. Add the full training wrapper
   `experiments/train_ray_attention_temporal_residual_reprojgate_full_mpiinf3dhp.py`
   that calls the base trainer with the standard MPI-INF-3DHP S1/S3 train set
   and S2 validation set.
3. Run a CPU smoke test on a tiny synthetic dataset to confirm the script
   executes without errors and the gate is active.
4. (Future) Run the full 30-epoch training and compare MPJPE against the
   non-gated residual baseline (currently 10.46 mm on MPI-INF-3DHP).

## 3. Prototype / smoke test

The model class itself can be sanity-checked directly:

```python
python motionflow_mv/fusion/ray_attention_temporal_residual_model.py
```

A full training smoke test can be run by creating a small synthetic `.npz`
dataset and invoking the trainer with `--epochs 1`.  See the validation step in
the task output for the exact command and result.

## 4. Success metrics

- MPI-INF-3DHP S1→S2/Seq1 MPJPE ≤ 10.46 mm (no regression from the gated
  residual head).
- The gated model improves or matches the non-gated residual baseline on the
  robustness benchmark (occlusion / calibration noise).
- Training smoke test completes one epoch without errors on CPU.
