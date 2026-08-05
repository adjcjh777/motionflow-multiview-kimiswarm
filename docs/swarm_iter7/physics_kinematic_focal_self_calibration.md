# Direction 11: Physics / Kinematic Consistency — Focal Self-Calibration

**Date:** 2026-08-05  
**Agent:** Direction 11 swarm agent  
**Status:** CPU smoke verified; GPU launcher prepared (queued — RTX 4090 busy)

---

## Problem Statement

Current best checkpoint (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`) already contains a learned principal-point correction layer, but focal-length self-calibration is only tested with a large correction head (`focal_max_scale=0.1`, loss weight matched to PP loss). The robustness matrix shows focal errors are still the dominant failure mode (focal 1% → ~19 mm). To make the model physically consistent we need a *tighter* focal self-calibration: allow only a small predicted focal correction (`focal_max_scale=0.02`) and supervise it with a small, dedicated loss weight (`focal_loss_weight=0.05`). This tests whether the network can learn a subtle, kinematically plausible intrinsic correction rather than relying on a large corrective re-scaling.

---

## Minimal Next Experiment

Train the existing PP model on MPI-INF-3DHP with focal self-calibration enabled:

```bash
python experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --principal_point_hidden 64 \
  --epochs 10 --num_workers 0 --train_samples 500 --val_stride 50 --batch_size 8 \
  --pp_loss_weight 0.1 --focal_max_scale 0.02 --focal_loss_weight 0.05 \
  --cam_aug_rot 0.5 --cam_aug_trans 0.005 --cam_aug_focal 0.01 --cam_aug_pp 5.0 \
  --output outputs/principal_point_focal_selfcalib_0p02.pth
```

This is a small 10-epoch smoke run (mirrors `scripts/run_focal_small_wsl.sh`) so it can be used as the first checkpoint before committing GPU time to a full run.

---

## Files to Touch / Add

### New launcher script

- `scripts/run_focal_selfcalib_0p02_wsl.sh` (created below)

### Existing files (read-only review confirms they already support this)

- `motionflow_mv/fusion/principal_point_correction.py` — already predicts a bounded focal scale via `focal_mlp` when `max_focal_scale > 0`.
- `motionflow_mv/fusion/ray_attention_temporal_residual_principal_point_model.py` — forwards `focal_max_scale` to the correction layer and returns predicted `focal_scale`.
- `experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py` — already accepts `--focal_max_scale` and `--focal_loss_weight` and supervises the predicted focal scale against the inverse of the applied perturbation.
- `motionflow_mv/losses/focal_calibration_loss.py` — CPU-only supervised MSE loss used in the smoke test below.

No existing experiment runners are modified.

---

## Rough Diff / Sketch

### `scripts/run_focal_selfcalib_0p02_wsl.sh`

```bash
#!/usr/bin/env bash
# Focal self-calibration small run (0.02 scale, 0.05 loss weight).
# GPU required. Queue after the current cross-view PP curriculum finishes.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --principal_point_hidden 64 \
  --epochs 10 --num_workers 0 --train_samples 500 --val_stride 50 --batch_size 8 \
  --pp_loss_weight 0.1 --focal_max_scale 0.02 --focal_loss_weight 0.05 \
  --cam_aug_rot 0.5 --cam_aug_trans 0.005 --cam_aug_focal 0.01 --cam_aug_pp 5.0 \
  --output outputs/principal_point_focal_selfcalib_0p02.pth \
  "$@"
```

### Training code already does the following

```python
if args.focal_max_scale > 0.0:
    pred_focal_scale = outputs[3]  # (B*T, V)
    true_focal_scale = true_focal_scale.to(device).squeeze(-1).unsqueeze(1).expand(B, T, -1)
    target_focal_scale = 1.0 / true_focal_scale.reshape(B * T, -1)
    focal_loss_weight = args.focal_loss_weight if args.focal_loss_weight is not None else args.pp_loss_weight
    loss = loss + focal_loss_weight * criterion(pred_focal_scale, target_focal_scale)
```

No new model code is needed.

---

## CPU-Only Verification (Run Now)

The existing `experiments/train_focal_calibration_smoke.py` exercises the `focal_calibration_loss` on synthetic perturbed focal-length pairs. It was run as a sanity check:

```bash
python experiments/train_focal_calibration_smoke.py --config configs/train_focal_calibration_smoke.yaml
```

Result:

```
Device: cpu
Seed:   42
Initial focal_loss=0.184038, final focal_loss=0.013997
Checkpoint saved to outputs\focal_calibration_smoke.pth
focal calibration loss CPU smoke test passed
```

The supervised focal loss is numerically stable and trains a tiny network to recover the inverse focal perturbation.

---

## Expected Success Metric

- **Primary:** validation MPJPE on MPI-INF-3DHP clean after 10 epochs comparable to the existing PP small run (target ≤ 12 mm for this smoke).
- **Robustness:** after the 10-epoch smoke, run `experiments/eval_curriculum_robustness.py` with `--focal_max_scale 0.02` and compare `focal_1%` MPJPE to the baseline (target < 14 mm vs current ~19 mm).
- **Calibration error:** predicted focal scale stays near identity on clean data and recovers the inverse of the injected focal perturbation under augmentation.

---

## Resource Requirement

- **GPU required** for the MPI-INF-3DHP training in `scripts/run_focal_selfcalib_0p02_wsl.sh`. Do **not** start it while the RTX 4090 is training the cross-view PP curriculum.
- The CPU smoke test above already passed and does not require GPU.

---

## Commit

- Local commit created with the report and the new launcher script.
- Commit hash: TBD after this report is committed.
- Push was not attempted because the GPU training is queued locally; the commit remains on branch `multiview-residual-exploration`.
