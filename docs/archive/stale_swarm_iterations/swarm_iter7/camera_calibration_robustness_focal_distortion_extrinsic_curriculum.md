# Direction: Camera calibration robustness (focal / distortion / extrinsic curriculum)

## Problem statement

The current best cross-view principal-point (PP) model already corrects small
principal-point errors, but the overall pipeline remains vulnerable to the
other camera-calibration error modes: focal-length drift, radial distortion,
and extrinsic (rotation / translation) noise. The baseline plan shows the
largest robustness gaps at `rot_0.5° → 16.89 mm` and `focal_1% → 19.13 mm`.
To close these gaps we need to evaluate each failure mode in isolation, then
train the model under a stronger calibration-perturbation curriculum that
includes focal length, radial distortion, and larger extrinsic noise.

## Simplest concrete next step

1. Add a reusable radial-distortion augmentation helper in the calibration
   perturbation module (done below).
2. Run a CPU-only synthetic sanity check that measures how each perturbation
   type (rotation, translation, focal length, principal point, radial
   distortion) affects plain DLT triangulation error. This gives us a
   baseline matrix without touching any running GPU job.
3. When the GPU is free, launch a short smoke run of the cross-view PP model
   with the new, stronger curriculum (`--focal_max_scale 0.02`,
   `--focal_loss_weight 0.05`, larger extrinsics, and the new radial
   distortion augment `--cam_aug_k1_std 0.05`).

## Files to touch (rough diff / sketch)

- `motionflow_mv/calibration/perturb.py` — add `perturb_radial_distortion`.
  Already added; the function applies Brown–Conrady `k1` distortion in
  normalized image coordinates and is safe (returns input when `k1_std <= 0`).

- `experiments/calibration_robustness_cpu_diagnostic.py` — new CPU-only script.
  It builds a tiny 4-camera rig, projects a 17-joint skeleton, corrupts the
  calibration, and triangulates with the corrupted cameras to report MPJPE.

- `scripts/run_crossview_pp_focal_distortion_extrinsic_wsl.sh` — new GPU
  launcher skeleton. It reuses the existing principal-point training script
  with stronger focal / distortion / extrinsic arguments.

- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` —
  next patch (not applied now) to consume the new arguments. Rough sketch:

```python
from motionflow_mv.calibration.perturb import (
    perturb_cameras_with_delta,
    perturb_radial_distortion,
)

# Inside the training loop, after loading (K, R, t, x):
K, R, t, pp_delta, focal_scale = perturb_cameras_with_delta(
    K, R, t,
    rot_std=rot_std,
    trans_std=trans_std,
    focal_std=focal_std,
    pp_std=pp_std,
)
# x shape: (B, T, V, J, 3)
x = perturb_radial_distortion(x[..., :2], K, k1_std=k1_std)
# ...pack back into (B, T, V, J, 3)...
```

## CPU-only run (completed)

Command:

```bash
python experiments/calibration_robustness_cpu_diagnostic.py
```

Output on the local WSL workspace (CPU, no GPU, ~1 second):

```text
clean:               MPJPE = 0.00 mm
rot_0.5_deg:         MPJPE = 1.27 mm
rot_1.0_deg:         MPJPE = 6.37 mm
trans_5mm:           MPJPE = 0.67 mm
trans_10mm:          MPJPE = 0.63 mm
focal_1pct:          MPJPE = 1.66 mm
focal_2pct:          MPJPE = 2.21 mm
pp_3px:              MPJPE = 1.75 mm
pp_5px:              MPJPE = 1.18 mm
distortion_k1_0.10:  MPJPE = 1.14 mm
distortion_k1_0.30:  MPJPE = 3.43 mm
Saved results to outputs\calibration_robustness_cpu_diagnostic.json
```

The matrix is synthetic and idealized (no observation noise), so absolute
numbers are small. It confirms that the infrastructure can inject realistic
focal / extrinsic / distortion errors and that the existing perturbation
helpers are functional. The next step is to train under these perturbations
so the model learns to compensate.

## Expected success metric

After the GPU smoke and full runs:

- Clean MPI-INF-3DHP ≤ 9.6 mm (maintain current baseline).
- `rot_0.5°` MPJPE < 12 mm and `focal_1%` MPJPE < 14 mm (down from the
  baseline ~16–19 mm).
- Distortion-aware model shows measurable gain over the no-distortion baseline
  on a synthetic `k1` robustness sweep.

## Resource requirement

- CPU-only analysis: completed above.
- GPU training: queued / skeleton only. No GPU training was started per the
  constraint that the RTX 4090 is busy.

## Commit note

Run after creating the report:

```bash
git add docs/swarm_iter7/camera_calibration_robustness_focal_distortion_extrinsic_curriculum.md \
        experiments/calibration_robustness_cpu_diagnostic.py \
        motionflow_mv/calibration/perturb.py \
        scripts/run_crossview_pp_focal_distortion_extrinsic_wsl.sh
git commit -m "swarm_iter7: camera calibration robustness skeleton and CPU diagnostic"
```
