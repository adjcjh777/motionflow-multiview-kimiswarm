# MotionFlow-MultiView: Iteration 5-6 Results Summary

## Best confirmed results

### MPI-INF-3DHP cross-subject (train S1, val S2/Seq1)

| Metric | Value |
|---|---|
| MPJPE | **13.12 mm** |
| PA-MPJPE | **10.86 mm** |
| PCK@50 mm | 0.9999 |
| PCK@100 mm | 1.0000 |
| PCK@150 mm | 1.0000 |
| PCK AUC (0-150 mm) | 0.9125 |

### Human3.6M cross-subject (train S1, val S5)

| Metric | Value |
|---|---|
| MPJPE | **5.74 mm** |
| PA-MPJPE | **3.99 mm** |
| PCK@50 mm | 0.9980 |
| PCK@100 mm | 0.9995 |
| PCK@150 mm | 0.9998 |
| PCK AUC (0-150 mm) | 0.9618 |

Checkpoint: `outputs/ray_attention_temporal_residual_v2.pth`

Full eval command:
```bash
conda run -n mf python experiments/eval_ray_attention_temporal_residual_v3.py \
    --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --batch_size 8 --out outputs/eval_residual_v2_full.json
```

## Key model variants (full-data, where available)

| Model | Params | MPJPE (mm) | Notes |
|---|---|---|---|
| Baseline temporal | 217,825 | 25.21 | Weighted DLT + temporal attention |
| **Residual v2** | 243,428 | **13.12** | Current best on MPI-INF-3DHP |
| Residual (small, d=32, h=64) | 66,420 | 13.19 | Lightweight variant, ~3× fewer params |
| **Residual on H36M** | 185,572 | **5.71** | Cross-dataset validation |
| V4 single-frame + residual | — | 12.13 | Smoke subset |
| Temporal + cross-view + residual | — | 11.56 | Smoke subset; full training too slow |
| Uncertainty-aware residual | — | 9.72 | Smoke subset |
| Deeper residual (3 blocks) | — | 13.24 | Smoke subset |

## Robustness (residual v2 on S2/Seq1)

| Perturbation | Level | MPJPE (mm) |
|---|---|---|
| Gaussian 2D noise | 5 px | 14.77 |
| Gaussian 2D noise | 20 px | 28.86 |
| Random joint occlusion | 50% | 13.13 |
| Random 2D outliers | 20% | 17.86 |

The model is extremely robust to occlusion (14-view redundancy) and degrades gracefully with noise/outliers.

## Worst joints (temporal baseline 25.26 mm)

| Joint | MPJPE (mm) |
|---|---|
| r_eye | 42.71 |
| l_thumb | 42.45 |
| l_ear | 42.39 |
| l_hand_tip | 42.14 |
| spine | 38.86 |

Residual head substantially reduces these, but distal/face joints remain the hardest.

## Ablations

### Residual head capacity (H36M S1→S5, 3 epochs)

| residual_hidden | Params | Best val MPJPE (mm) |
|---|---:|---:|
| 64 | 185,572 | **5.71** |
| 128 | 202,468 | 5.74 |
| 256 | 260,836 | 6.43 |

128 remains the practical default; 256 starts to overfit on this 3-epoch schedule.

## Artifacts

- Model: `motionflow_mv/fusion/ray_attention_temporal_residual_model.py`
- Trainer: `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py`
- Evaluator: `experiments/eval_ray_attention_temporal_residual_v3.py`
- FusionModule plugin: `motionflow_mv/fusion/ray_attention_temporal_residual_module.py`
- End-to-end demo: `experiments/demo_ray_attention_temporal_residual.py`
- Paper story: `docs/swarm_iter6/paper_story_residual.md`
- Robustness report: `docs/swarm_iter6/robustness_residual_v1.md`
- Failure analysis: `docs/swarm_iter5/failure_analysis_temporal_mpiinf3dhp.md`

## In-progress

- Full 10-epoch residual v2 re-training: `outputs/ray_attention_temporal_residual_full10.pth`
- Goal: confirm whether 13.12 mm can be improved further with longer training.
