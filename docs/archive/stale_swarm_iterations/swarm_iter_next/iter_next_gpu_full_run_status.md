# Next GPU Full-Run Status

**Branch:** `multiview-residual-exploration`  
**Date:** 2026-08-06  
**Current GPU job:** Bayesian Triangulation full run (`scripts/run_bayesian_tri_pp_full_wsl.sh`) on the RTX 4090.

## Objective

Prepare two candidate GPU full-run experiments to be launched as soon as the RTX 4090 is free:

1. **Primary candidate:** `epipolar_bias_v2_pp` — geometry-aware ST-attention bias.
2. **Fallback candidate:** `hierarchical_view_temporal_joint_pp` — three-stage hierarchical attention (view groups → temporal → skeleton graph).

No training has been started; only launch scripts and evaluation wrappers are committed.

## Scripts Added

| Script | Purpose |
|--------|---------|
| `scripts/run_epipolar_bias_v2_pp_full_wsl.sh` | 20-epoch GPU full run for `epipolar_bias_v2_pp` |
| `scripts/eval_epipolar_bias_v2_pp_full_wsl.sh` | Clean full-metrics evaluation of the v2 checkpoint |
| `scripts/run_hierarchical_attention_pp_full_wsl.sh` | 20-epoch GPU full run for `hierarchical_view_temporal_joint_pp` |
| `scripts/eval_hierarchical_attention_pp_full_wsl.sh` | Clean full-metrics evaluation of the hierarchical checkpoint |

## Hyperparameters

Both runs mirror the **Bayesian Tri full-run** configuration for a fair comparison:

- `clip_len = 13`
- `d = 64`
- `residual_hidden = 128`
- `batch_size = 8`
- `train_samples = 1000` per train sequence
- `epochs = 20`
- `val_stride = 50`
- `pp_loss_weight = 0.2`
- `cam_aug_pp = 5.0`
- `cam_aug_focal = 0.01`
- `cam_aug_schedule = intrinsics_curriculum`
- `cam_aug_intrinsics_ramp_epochs = 5`
- `pp_pretrain_epochs = 3`

Model-specific settings:

- **Epipolar Bias v2:** `n_st_layers = 2`
- **Hierarchical Attention:** `n_view_groups = 2`, `n_view_layers = 2`, `n_temporal_layers = 2`, `n_joint_graph_layers = 1`

## Data Split

Training sequences (WebBridge MPI-INF-3DHP v14):

- `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz`
- `data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz`
- `data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz`
- `data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz`

Validation sequence:

- `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`

## Outputs

| Candidate | Log | Checkpoint | Eval JSON |
|-------------|-----|------------|-----------|
| Epipolar Bias v2 | `outputs/epipolar_bias_v2_pp_full_mpiinf3dhp.log` | `outputs/epipolar_bias_v2_pp_full_mpiinf3dhp.pth` | `outputs/epipolar_bias_v2_pp_full_mpiinf3dhp_eval.json` |
| Hierarchical Attention | `outputs/hierarchical_attention_pp_full_mpiinf3dhp.log` | `outputs/hierarchical_attention_pp_full_mpiinf3dhp.pth` | `outputs/hierarchical_attention_pp_full_mpiinf3dhp_eval.json` |

## Trigger Condition

Start these runs only after:

1. The current **Bayesian Triangulation full run** has finished (or been stopped), freeing the RTX 4090.
2. The `outputs/bayesian_tri_pp_full_mpiinf3dhp.pth` checkpoint and its `outputs/bayesian_tri_pp_full_mpiinf3dhp_eval.json` have been inspected.

Recommended execution order once the GPU is free:

1. Run `scripts/run_epipolar_bias_v2_pp_full_wsl.sh` (primary candidate).
2. After it completes, run `scripts/eval_epipolar_bias_v2_pp_full_wsl.sh`.
3. If the Epipolar Bias v2 result is not conclusive or the run fails, launch `scripts/run_hierarchical_attention_pp_full_wsl.sh` and its eval.

## Commit Reference

Scripts committed in: `ae27069` on branch `multiview-residual-exploration`.

## Blockers

- The RTX 4090 is currently occupied by the Bayesian Triangulation full run.
- No GPU training for these candidates should begin until that run finishes and the GPU is free.
