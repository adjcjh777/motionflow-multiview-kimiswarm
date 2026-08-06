# Hierarchical View → Temporal → Skeleton-Joint Attention Status

**Branch:** `multiview-residual-exploration`  
**Commit:** `9f57410`  
**Date:** 2026-08-06

## Summary

Wired `RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint` into the training and evaluation harness under the key `hierarchical_view_temporal_joint_pp`.

## Changes

- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
  - Added model import.
  - Added `hierarchical_view_temporal_joint_pp` to `model_type` choices.
  - Added CLI args `--n_view_groups` and `--n_joint_graph_layers`.
  - Added model instantiation branch passing `n_view_groups`, `n_view_layers`, `n_temporal_layers`, `n_joint_graph_layers`, and `use_skeleton_graph=True`.

- `experiments/eval_full_metrics.py`
  - Added model import.
  - Registered `hierarchical_view_temporal_joint_pp` in `MODEL_CLASSES`.
  - Added CLI args `--n_view_groups`, `--n_joint_graph_layers`, and `--no_skeleton_graph`.
  - Updated `build_model()` to populate the hierarchical kwargs.

- `motionflow_mv/fusion/graph_joint_relation.py`
  - Fixed `RuntimeError` by replacing `x.view(...)` with `x.reshape(...)` so non-contiguous tensors from the hierarchical pipeline work correctly.

- `scripts/run_hierarchical_attention_smoke_wsl.sh`
  - New CPU-only smoke script (2 epochs, batch size 2, d=32, residual_hidden=64).

## Smoke Test

Run:

```bash
./scripts/run_hierarchical_attention_smoke_wsl.sh
```

Log: `outputs/hierarchical_attention_smoke.log`

Result:

```text
Device: cpu
n_views=14, j=28, clip_len=13, d=32, model_type=hierarchical_view_temporal_joint_pp, ...
Model params: 104535
Epoch 1: train_loss=40.898004, val_MPJPE=20.16mm (saved)
Epoch 2: train_loss=40.822014, val_MPJPE=31.39mm
Best val MPJPE: 20.16mm -> outputs/hierarchical_attention_smoke.pth
```

Both epochs completed and produced a validation MPJPE.

## Blockers

None. The smoke script completed successfully on CPU without disturbing the GPU run.

## Next Steps

- Tune hierarchical hyperparameters (view groups, view/temporal/graph layer counts).
- Run a full training/evaluation sweep on MPI-INF-3DHP/H36M.
