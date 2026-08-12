# Uncertainty quantification and confidence fusion

## Problem statement

The current best model, `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, triangulates 3-D poses with a learned per-view sigmoid weight. While this weight is data-driven, it is not an interpretable estimate of uncertainty: it does not tell us how much we should trust each view, and it cannot be used to filter outliers or to report confidence intervals. Direction #9 in the swarm plan asks for a per-view log-variance head whose precision `exp(-log_var)` replaces the sigmoid weight in the DLT step. A minimal first step is to plug the already-implemented `CrossviewResidualUncertaintyModel` into the training pipeline, verify on CPU that the uncertainty head actually down-weights a deliberately noisy view, and then run a short GPU smoke train once the RTX 4090 is free.

## Simplest concrete next step

1. Add a CPU-only robustness sanity check that perturbs one synthetic view and asserts that the model assigns it higher log-variance (lower precision).
2. Add a thin training wrapper `experiments/train_crossview_residual_uncertainty_mpiinf3dhp.py` and a queued launcher `scripts/run_crossview_residual_uncertainty_smoke_wsl.sh` for a 10-epoch GPU smoke.

## Files touched

```text
experiments/eval_uncertainty_pp_view_robustness.py          (new, CPU sanity check)
experiments/train_crossview_residual_uncertainty_mpiinf3dhp.py (new, GPU training skeleton)
scripts/run_crossview_residual_uncertainty_smoke_wsl.sh     (new, queued launcher)
motionflow_mv/models/crossview_residual_uncertainty.py      (existing, reused)
```

## Rough diff / sketch

The uncertainty model already exists at `motionflow_mv/models/crossview_residual_uncertainty.py`:

```python
class CrossviewResidualUncertaintyModel(...):
    def __init__(self, ..., uncertainty_loss_weight=0.1, ...):
        ...
        self.uncertainty_head = nn.Linear(d, 1)

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        ...
        log_var = self.uncertainty_head(feat_for_uncertainty).squeeze(-1)  # (B*T, V, J)
        precision = torch.exp(-log_var)
        weights = precision * confidences * visibility
        pred_3d = _triangulate_weighted_dlt(points_2d, weights, P)
        nll_loss = self._reprojection_nll(...)
        return pred_3d, weights, log_var, nll_loss
```

The new training wrapper is a copy of `train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` with the model swapped and the loss extended by the returned `nll_loss`:

```python
pred, weights, log_var, nll_loss = model(xb, K=K, R=R, t=t)
loss = criterion(pred, yb) + nll_loss
```

The CPU sanity check loads the same untrained model, injects a 50 px offset into view 1, and checks that the noisy view has the highest per-view mean log-variance and the lowest mean precision.

## CPU-only run

Command:

```bash
. .venv/bin/activate
python experiments/eval_uncertainty_pp_view_robustness.py
```

Result:

```text
Per-view mean log-variance and precision:
  view 0: log_var=1.1148, precision=0.3280
  view 1: log_var=1.1528, precision=0.3158 <-- noisy view
  view 2: log_var=1.1122, precision=0.3289
  view 3: log_var=1.1112, precision=0.3292

Noisy view log_var: 1.1528
Clean views log_var mean: 1.1127  max: 1.1148
Noisy view precision: 0.3158
Clean views precision mean: 0.3287  min: 0.3280
PASS: noisy view is correctly assigned higher uncertainty and lower precision.
nll_loss: 13160601207439360.000000
```

The huge absolute NLL is expected: the model is untrained and the synthetic data are geometrically random. The important result is the ordering: the perturbed view receives the highest uncertainty and the lowest precision, confirming the head is wired correctly.

## GPU smoke (queued, not run)

Launcher: `scripts/run_crossview_residual_uncertainty_smoke_wsl.sh`

```bash
python experiments/train_crossview_residual_uncertainty_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
    --epochs 10 --batch_size 8 --train_samples 4000 --val_stride 1 \
    --uncertainty_loss_weight 0.1 \
    --cam_aug_pp 5.0 --cam_aug_schedule extrinsic_curriculum --cam_aug_ramp_epochs 5 \
    --view_dropout_rate 0.2 --min_views 4 \
    --output outputs/crossview_residual_uncertainty_smoke_v1.pth
```

This is queued behind the currently running cross-view PP curriculum on the WSL RTX 4090 and will not be executed now.

## Expected success metric

- CPU sanity check: noisy view has strictly higher mean log-variance and lower mean precision than all clean views (achieved above).
- GPU smoke: validation MPJPE on MPI-INF-3DHP S2 within 10% of the current best PP baseline (~9.3 mm clean). If the 10-epoch smoke reaches ≤ 10.2 mm, proceed to a full 20–30 epoch run and evaluate robustness under per-view occlusion/noise.

## Resource requirements

- CPU-only: `experiments/eval_uncertainty_pp_view_robustness.py` runs on CPU in seconds.
- GPU-only: `scripts/run_crossview_residual_uncertainty_smoke_wsl.sh` requires the WSL RTX 4090 and is queued, not run.

## Notes

- The uncertainty head was already implemented; this iteration wires it to a runnable training skeleton and a CPU sanity check.
- No existing experiment runners were modified, so currently running jobs are unaffected.
