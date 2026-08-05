# Direction 8: Variable-View Inference and View Dropout

## Problem Statement

The current best model `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` is trained with a fixed number of cameras (`n_views`). At deployment, camera rigs may have anywhere from 2 to 14 active views, and views can drop out because of occlusion or calibration loss. We need to (a) quantify how accuracy degrades as the active view count decreases and (b) harden training against such dropouts. The zero-confidence masking wrapper in `motionflow_mv/fusion/variable_view_inference.py` already lets a fixed-view model run with fewer views; the next step is to run the MPJPE@k benchmark on the best checkpoint and pair it with the existing `view_dropout_rate` training augmentation.

## Simplest Concrete Next Step

Run the existing `experiments/eval_variable_views.py` on the best cross-view PP checkpoint and the MPI-INF-3DHP validation set to produce an MPJPE@k curve for `k = 2..14`. In parallel, prepare a CPU-only synthetic smoke launcher so the variable-view pipeline can be verified without touching GPU/data. Once the GPU queue frees up, launch a short training run with `view_dropout_rate=0.2` and re-evaluate the curve.

## Files to Touch / Code Sketch

### 1. CPU smoke launcher (new)
`scripts/eval_variable_views_smoke_cpu.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tmp/variable_view_smoke
KMP_DUPLICATE_LIB_OK=TRUE \
    python experiments/eval_variable_views.py \
        --n_views 6 --j 17 --clip_len 9 \
        --num_subsets_per_k 10 --seed 42 \
        --output_json tmp/variable_view_smoke/results.json \
        --output_csv  tmp/variable_view_smoke/results.csv
```

### 2. Real GPU launcher (skeleton only — do not run)
`scripts/eval_variable_views_crossview_pp_wsl.sh` already exists. The next run should be:
```bash
python experiments/eval_variable_views.py \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_full_ppw005_20ep.pth \
    --model_class crossview_residual_pp \
    --d 64 --residual_hidden 128 --n_temporal_layers 2 --clip_len 13 \
    --min_views 2 --max_views 14 --num_subsets_per_k 50 \
    --output_json tmp/variable_view_crossview_pp/results.json \
    --output_csv  tmp/variable_view_crossview_pp/results.csv
```

### 3. Training with view dropout (use existing trainer)
`experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` already supports `view_dropout_rate`. Example command skeleton:
```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --view_dropout_rate 0.2 --min_views 2 \
    --epochs 20 --clip_len 13 --d 64 --residual_hidden 128
```

### 4. Minimal Python sketch: variable-view masking helper
Already implemented in `motionflow_mv/fusion/variable_view_inference.py`:
```python
from motionflow_mv.fusion.variable_view_inference import (
    prepare_variable_view_input,
    VariableViewInferenceWrapper,
)

wrapper = VariableViewInferenceWrapper(model)
x, K, R, t, active = prepare_variable_view_input(
    x, K, R, t, active_views=[0, 2], n_views_max=n_views_max
)
pred, weights = wrapper(x, K=K, R=R, t=t, active_views=[0, 2])
```

## Expected Success Metric

- CPU smoke: `eval_variable_views.py` completes for `k = 2..6` without error and writes `tmp/variable_view_smoke/results.json`.
- Real benchmark (GPU): `MPJPE@14` matches the best fixed-view baseline (~9.3 mm clean); `MPJPE@k` degrades gracefully, e.g. ≤ 20% relative increase at `k = 6` and ≤ 40% at `k = 2`.
- Training: a model trained with `view_dropout_rate=0.2` matches the no-dropout baseline at `k = n_views` while improving MPJPE@2..6 by ≥ 10%.

## CPU vs. GPU

- Smoke validation is **CPU-only** and has been run now.
- Real MPJPE@k benchmark and view-dropout training require **GPU**; only skeletons/launchers are produced here. The WSL RTX 4090 is currently running the cross-view PP curriculum, so no GPU work is started.

## CPU Smoke Run — Command and Result

Command executed:
```bash
bash scripts/eval_variable_views_smoke_cpu.sh
```

Result (`tmp/variable_view_smoke/results.json`):
```
 k=2: mean=2020.17 mm, std=437.13, subsets=10
 k=3: mean=1813.82 mm, std=158.17, subsets=10
 k=4: mean=1557.07 mm, std=183.22, subsets=10
 k=5: mean=1484.35 mm, std= 27.88, subsets= 6
 k=6: mean=1405.26 mm, std=  0.00, subsets= 1
```
The synthetic 2D/3D data is geometrically meaningless, so the absolute MPJPE values are large; the important check is that the fixed-slot model accepts every `k ∈ [2, 6]`, runs on CPU, and produces a per-k summary. This confirms the variable-view inference wrapper is functional and the benchmark is ready for the real checkpoint.

## Notes

- No existing experiment runner was modified; only a new smoke launcher and a new report were added.
- The `.venv` python symlink is broken in this session, so the smoke script falls back to the Anaconda CPU python with `KMP_DUPLICATE_LIB_OK=TRUE`.
- If the push network fails, the local commit will remain; see git log for the exact hash.
