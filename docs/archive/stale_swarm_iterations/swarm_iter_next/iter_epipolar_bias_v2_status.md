# Epipolar Bias v2 Wiring & Smoke

## Summary

Wired the existing `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2` model into the training and evaluation harness under the key `epipolar_bias_v2_pp`.

## Changes

- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
  - Added import for the v2 model.
  - Added `epipolar_bias_v2_pp` to `--model_type` choices.
  - Added model instantiation branch for `epipolar_bias_v2_pp`.

- `experiments/eval_full_metrics.py`
  - Added import for the v2 model.
  - Registered `epipolar_bias_v2_pp` in `MODEL_CLASSES`.

- `scripts/run_epipolar_bias_v2_smoke_wsl.sh`
  - New CPU smoke script (2 epochs, batch size 2, `d=32`, `residual_hidden=64`, `CUDA_VISIBLE_DEVICES=-1`).

## Smoke Result

```bash
bash scripts/run_epipolar_bias_v2_smoke_wsl.sh
```

- Device: CPU
- Dataset: `tmp/mpi_s01_seq01_smoke.npz` (train), `tmp/mpi_s02_seq01_smoke.npz` (val)
- Epoch 1 train_loss=40.09, val_MPJPE=48.66 mm
- Epoch 2 train_loss=40.55, val_MPJPE=27.69 mm
- **Best val MPJPE: 27.69 mm**

## Commit

- Hash: `0bbb6f7`
- Branch: `multiview-residual-exploration`
- Pushed to `origin`.

## Blockers

None. The v2 model ran cleanly through both epochs without modification to its implementation.
