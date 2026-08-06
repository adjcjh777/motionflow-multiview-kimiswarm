# Camera-Conditioned Model Wiring Status

## Summary

Wired `RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned` into the training and evaluation harness under the model key `camera_conditioned_pp`.

## Changes

- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
  - Imported the camera-conditioned model.
  - Added `camera_conditioned_pp` to `--model_type` choices.
  - Added construction branch with the same kwargs as the other PP models.
- `experiments/eval_full_metrics.py`
  - Imported the camera-conditioned model.
  - Registered `camera_conditioned_pp` in `MODEL_CLASSES`.
  - Added it to the `n_st_layers` + `residual_hidden` kwargs branch in `build_model`.
- `scripts/run_camera_conditioned_smoke_wsl.sh`
  - CPU-only smoke script (2 epochs, batch size 2, d=32, residual_hidden=64).

## Smoke Test Result

```text
Device: cpu
n_views=14, j=28, ...
Model params: 87542
Epoch 1: train_loss=42.942889, val_MPJPE=183.08mm (saved)
Epoch 2: train_loss=40.800921, val_MPJPE=69.19mm (saved)
Best val MPJPE: 69.19mm
```

The smoke run completed both epochs on CPU and reported a validation MPJPE of **69.19 mm**.

## Commit / Push

- Commit: `f317998`
- Branch: `multiview-residual-exploration`
- Pushed to `origin multiview-residual-exploration` (initial push failed with a transient TLS EOF error; retry succeeded).

## Next Steps / Blockers

No blockers. The model is ready for full GPU runs when the RTX 4090 is free.
