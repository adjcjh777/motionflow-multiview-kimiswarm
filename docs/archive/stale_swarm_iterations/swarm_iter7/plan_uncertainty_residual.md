# Plan: Full-data validation of the uncertainty-aware residual model

**Direction:** `uncertainty residual`  
**Goal:** Beat the current best MPI-INF-3DHP cross-subject MPJPE of **11.17 mm** by validating the uncertainty-aware residual refinement head (`RayAttentionFusionModelTemporalResidualV3`) on the full benchmark.

---

## What to change

1. **Resolve the class-name collision.**  
   `motionflow_mv/fusion/ray_attention_temporal_residual_v3_model.py` and `motionflow_mv/fusion/ray_attention_temporal_residual_model_v3.py` both define `RayAttentionFusionModelTemporalResidualV3`. Rename the uncertainty-aware class to, e.g., `RayAttentionFusionModelTemporalResidualUncertainty`.

2. **Update the trainer to use the uncertainty-aware model.**  
   In `experiments/train_ray_attention_temporal_residual_v3_mpiinf3dhp.py`, change the import from the base residual model to:
   ```python
   from motionflow_mv.fusion.ray_attention_temporal_residual_v3_model import (
       RayAttentionFusionModelTemporalResidualV3,  # or the renamed class
   )
   ```
   The rest of the training loop stays the same because the model still returns `(pred_3d, weights)`.

3. **(Optional but recommended follow-up)** Add an explicit reprojection-NLL auxiliary loss to supervise the uncertainty branch. Use the `_reprojection_nll` implementation in `motionflow_mv/fusion/ray_attention_temporal_uncertainty_model.py` as a template and add `loss = mse + 0.1 * nll_loss` in the trainer. This can be skipped for the first full run so the architecture itself is isolated.

4. **Update the evaluator** `experiments/eval_ray_attention_temporal_residual_v3.py` to load the same uncertainty-aware class.

---

## Why it should help

- The smoke run of the uncertainty-aware residual model already reported **9.72 mm MPJPE** on a small smoke subset.
- The weight/uncertainty head gives the residual MLP a per-joint confidence summary, allowing it to down-weight noisy views/joints and focus corrections where the raw DLT triangulation is weakest.
- Full-data validation is the only way to tell whether the smoke gain transfers to the real benchmark and whether it can push the current best **11.17 mm** lower.

---

## Exact commands

### Smoke test (quick sanity)

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_v3_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --d 64 --residual_hidden 128 --n_temporal_layers 2 \
    --epochs 2 --batch_size 8 --train_samples 4000 \
    --output outputs/ray_attention_temporal_residual_v3_mpiinf3dhp_smoke.pth
```

### Full MPI-INF-3DHP run

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_v3_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --residual_hidden 128 --n_temporal_layers 2 \
    --epochs 5 --batch_size 8 --train_samples 4000 \
    --output outputs/ray_attention_temporal_residual_v3_full5.pth
```

### Evaluation

```bash
conda run -n mf python experiments/eval_ray_attention_temporal_residual_v3.py \
    --checkpoint outputs/ray_attention_temporal_residual_v3_full5.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --residual_hidden 128 --batch_size 8 \
    --out outputs/eval_residual_v3_full5.json
```

---

## Expected metrics

| Metric | Target |
|---|---|
| MPJPE (mm) | **< 11.17**, ideally 9.5–10.5 |
| PA-MPJPE (mm) | 7.5–9.0 |
| PCK@50/100/150 mm | ~0.999+ at 100/150 mm |
| PCK AUC (0–150 mm) | ~0.93–0.94 |

The smoke subset result (9.72 mm) suggests the full-data result has a good chance of beating the 11.17 mm baseline.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Class-name collision** between `ray_attention_temporal_residual_v3_model.py` and `ray_attention_temporal_residual_model_v3.py` could mask the wrong model. | Rename the uncertainty-aware class before running. |
| **Trainer still imports the base residual model**, so the run would not test uncertainty at all. | Verify the import line before launching. |
| The uncertainty branch is currently learned **implicitly**; it may not generalize to full data. | If no improvement, add the explicit reprojection-NLL auxiliary loss. |
| Overfitting after 5 epochs on the full set. | Monitor validation MPJPE and reduce/extend epochs accordingly. |
| Memory/latency: d=64, h=128, clip_len=13, 14 views, 28 joints fits on an RTX 4090 with batch_size=8, but larger `d` or `clip_len` may OOM. | Keep these hyperparameters fixed for the first full run. |

---

## Next step after this plan

If the full run beats 11.17 mm, integrate the explicit reprojection-NLL uncertainty loss and run a second full experiment. If it does not, debug whether the uncertainty branch is being loaded and consider a supervised uncertainty objective or uncertainty gating inside the residual MLP.
