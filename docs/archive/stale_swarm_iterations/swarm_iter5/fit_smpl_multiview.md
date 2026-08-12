# Multi-View SMPL Fitting Stage

## What was implemented

`experiments/fit_smpl_multiview.py` is a minimal, self-contained script that takes
the fused world-coordinate 3D joints produced by the multi-view fusion pipeline
(and optional per-view 2D keypoints + calibrated cameras) and fits a coherent
SMPL body to them.

Key features:

- Pure PyTorch autograd optimization with Adam.
- Jointly optimizes `global_orient`, `body_pose`, `transl`, and a *shared*
  sequence-level `betas` vector.
- Loss: 3D joint MSE + optional multi-view reprojection error + shape/pose
  regularization + optional temporal smoothness.
- Procrustes initialization of global orientation and translation using
  `torch.linalg.svd` (avoids a platform-specific `np.linalg.svd` crash).
- Unit-aware input handling (`m`/`cm`/`mm`).
- Output is a standard `.npz` of SMPL parameters and reconstructed 3D joints,
  ready to be wrapped into a `HumanMotionIR`.

## Important design decisions

1. **Shared betas.** A single shape vector is shared across the whole clip by
   default. This matches the swarm recommendation to enforce sequence-level
   shape consistency and avoids per-frame shape drift.
2. **Joint correspondence.** The fitter assumes the input 3D joints are the
   first `J` SMPL body joints (the convention used by the synthetic generator and
   the ray-aware fusion demos). A COCO/H36M regressor is not included to keep the
   first version minimal.
3. **Read-only SMPL asset.** The script only reads `data/smpl/SMPL_NEUTRAL.pkl`;
   it does not redistribute or modify it.
4. **Reprojection is optional.** Cameras are passed through and used only if
   `reproj_weight > 0` and 2D observations are available. The default mode fits
   purely to 3D joints, which is the common post-fusion use case.

## Verification

Tested on `data/h36m_hf/s_01_act_02_multiview.npz` (Human3.6M, 4 views, 17
joints, millimeter units) on CPU:

```bash
/d/anaconda3/envs/jz_py310/python experiments/fit_smpl_multiview.py \
    --input data/h36m_hf/s_01_act_02_multiview.npz \
    --output outputs/fit_smpl_multiview_10f.npz \
    --max_frames 10 --n_iters 100 --input_unit mm --lr 0.01 --device cpu
```

Result:

```text
Fitting MPJPE: 0.321062 m
Final 3D MSE : 0.053551
```

A 3-frame run with `reproj_weight=0.01` also completed successfully,
confirming the reprojection code path.

## Known limitations and next steps

- **No COCO/H36M joint regressor.** If the input is not in the first-`J` SMPL
  joint order, the fitter will produce incorrect results. A regressor or a
  pre-alignment mapping is needed for arbitrary joint sets.
- **Optimization-based, not real-time.** It is intended for offline use after
  fusion, not for online ICRA-style pipelines.
- **No explicit temporal or physics priors beyond the optional smoothness
  term.** Adding a learned pose prior (e.g., from AMASS) is left for future work.
- **Shape/pose ambiguity is not fully resolved.** The current regularization is
  light; very short clips may produce unstable shapes.

## Files added/modified

- `experiments/fit_smpl_multiview.py` (new)
- `docs/swarm_iter5/fit_smpl_multiview.md` (new)
