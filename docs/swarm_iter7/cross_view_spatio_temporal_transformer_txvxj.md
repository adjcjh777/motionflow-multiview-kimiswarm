# Direction 3: Cross-view spatio-temporal Transformer (T×V×J)

## Problem statement

The current best model, `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, fuses multi-view evidence with two separate axes: joint-level attention inside each frame, then a combined (time × view) attention block. This factorisation prevents the model from directly learning relationships where, for example, an elbow joint in one view at time `t` should attend to a wrist in another view at time `t+1`. A true (T×V×J) transformer tokenises every `(time, view, joint)` tuple and lets all three axes interact in the same attention field, which is the natural next architectural step toward pushing MPI-INF-3DHP clean MPJPE below 9 mm. The repo already contains two prototype implementations (`SpatiotemporalPrincipalPointModel` and `RayAttentionFusionModelSpatiotemporal`), but they have not been rigorously compared to the best baseline or queued for full training.

## Simplest concrete next step

Run a CPU-only shape/gradient and timing smoke test of the existing factorised `(T×V×J)` model against the best PP baseline, then prepare the existing GPU launcher for the pending queue. No training is started now because the RTX 4090 is occupied.

## Files to touch / sketch

1. `scripts/smoke_spatiotemporal_txvxj_cpu.py` (new) — instantiates both models on CPU, checks forward/backward shapes and gradients, and reports parameter counts and relative CPU latency.
2. `scripts/run_spatiotemporal_principal_point_wsl.sh` (existing, queued) — will run the full GPU training once the queue frees.
3. `experiments/train_spatiotemporal_principal_point_mpiinf3dhp.py` (existing) — no changes; used as-is for the queued GPU run.
4. `motionflow_mv/models/spatiotemporal_principal_point_model.py` (existing) — no changes required; the model already passes its unit tests.

### Rough diff of the smoke script

```python
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.models.spatiotemporal_principal_point_model import (
    SpatiotemporalPrincipalPointModel,
    _make_cameras,
)

device = torch.device("cpu")
B, T, V, J = 2, 13, 4, 17
cameras = _make_cameras(V)
x = torch.rand(B, T, V, J, 3)

baseline = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
    j=J, d=64, n_views=V, return_pp_delta=True
).to(device)
st_model = SpatiotemporalPrincipalPointModel(
    j=J, d=64, n_views=V, return_pp_delta=True
).to(device)

pred_b = baseline(x, cameras=cameras)[0]
pred_s = st_model(x, cameras=cameras)[0]
assert pred_b.shape == (B, T, J, 3)
assert pred_s.shape == (B, T, J, 3)
```

## CPU-only run

Command:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/smoke_spatiotemporal_txvxj_cpu.py
```

Result:

```text
=== (T x V x J) spatio-temporal smoke test (CPU) ===
batch=2, clip_len=13, views=4, joints=17

Baseline params: 211,750
Factorised T x V x J params: 225,702

Forward shapes OK.
Baseline forward+backward: 0.133s/iter
Factorised T x V x J forward+backward: 0.155s/iter

Relative CPU time (T x V x J / baseline): 1.17x
Factorised model runs within ~3x of the baseline on CPU.
```

The factorised `(T×V×J)` model is only ~7 % larger and ~17 % slower per CPU iteration than the best PP baseline, so the architecture is not obviously too expensive to pursue.

## GPU training queued next

When the RTX 4090 is free, run the existing launcher (already in the GPU queue):

```bash
bash scripts/run_spatiotemporal_principal_point_wsl.sh
```

This trains the factorised `SpatiotemporalPrincipalPointModel` on MPI-INF-3DHP using the canonical multiview clips.

## Expected success metric

- CPU smoke: forward/backward shapes and gradients pass; relative CPU time ≤ 3× the best PP baseline.
- GPU smoke (queued): one full epoch on the small `_smoke` clips completes without NaNs/infinites.
- Full GPU run: clean MPJPE ≤ 9.2 mm on the standard MPI-INF-3DHP test set, with a plausible path to < 9.0 mm after longer training or a slight increase in `d`/`n_st_layers`.

## Resource classification

- This CPU-only smoke test: **CPU-only, safe to run now**.
- The full experiment: **GPU required**; launcher already exists and is queued, so no GPU job is started now.
