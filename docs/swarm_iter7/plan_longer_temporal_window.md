# Swarm Iter 7 Plan: Longer Temporal Window

**Date:** 2026-08-04  
**Direction:** Longer temporal window for `RayAttentionFusionModelTemporalResidual`  
**Goal:** Push MPI-INF-3DHP cross-subject MPJPE below the current best 11.17 mm toward ICRA/CVPR 2027.

## Current Best (to beat)

| Setting | Value |
|---|---|
| Model | `RayAttentionFusionModelTemporalResidual` |
| d / residual_hidden | 64 / 128 |
| Params | ~243 k |
| clip_len | 13 |
| MPI-INF-3DHP val (S2/Seq1) MPJPE | **11.17 mm** |
| Train files | `s_01_seq_01_v14_multiview_m.npz`, `s_01_seq_02_v14_multiview_m.npz` |
| Val file | `s_02_seq_01_v14_multiview_m.npz` |

## What to Change

Double the temporal context from **13 frames** to **27 frames** while keeping the model capacity identical:

- `clip_len`: 13 → **27**
- `d`: 64 (unchanged)
- `residual_hidden`: 128 (unchanged)
- `n_temporal_layers`: 2 (unchanged)
- `batch_size`: 8 → **4** (memory trade-off)
- `epochs`: 5 (matches the current 11.17 mm run)
- `train_samples`: 4000 per train sequence (unchanged)

No model code changes are required: `max_temporal_len=256` in the temporal transformer already supports `clip_len=27`.

## Why It Should Help

1. **More motion context.** 27 frames at 30 fps covers ~0.9 s of motion vs. ~0.43 s for 13 frames. The temporal attention can resolve short-term ambiguities (occlusion, motion blur, self-occlusion) by observing the full motion arc.
2. **Better temporal consistency.** A longer window gives the residual head stronger evidence for smooth, physically plausible trajectories, which should reduce frame-to-frame jitter and improve PCK tails / AUC.
3. **Low-friction experiment.** Only the training/evaluation `clip_len` and batch size change. The architecture and checkpoint format stay the same, so any improvement can be cleanly attributed to temporal context.

## Smoke Test (≤ 1 epoch, fast validation)

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 27 --d 64 --residual_hidden 128 --n_temporal_layers 2 \
    --epochs 1 --batch_size 2 --train_samples 500 \
    --output outputs/ray_attention_temporal_residual_clip27_smoke.pth
```

## Full Training Command

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 27 --d 64 --residual_hidden 128 --n_temporal_layers 2 \
    --epochs 5 --batch_size 4 --train_samples 4000 \
    --output outputs/ray_attention_temporal_residual_clip27.pth
```

## Evaluation Command

```bash
conda run -n mf python experiments/eval_ray_attention_temporal_residual_v1.py \
    --checkpoint outputs/ray_attention_temporal_residual_clip27.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 27 --d 64 --residual_hidden 128 --batch_size 4 \
    --out outputs/eval_residual_clip27.json
```

## Expected Metrics

| Metric | Current best (clip_len=13) | Target (clip_len=27) |
|---|---:|---:|
| MPJPE | 11.17 mm | **9.5–10.5 mm** |
| PA-MPJPE | 8.24 mm | **7.0–8.0 mm** |
| PCK@150 mm | 1.0000 | 1.0000 |
| PCK AUC (0–150 mm) | 0.9256 | **0.935–0.945** |

Success is defined as **MPJPE < 11.17 mm** on the same S2/Seq1 validation split.

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Memory blow-up** from O(T²) attention (T=27 is ~4.3× memory vs. T=13) | Medium | Use `batch_size=4` (or drop to 2 if OOM); gradient accumulation can be added if needed. |
| **Diminishing returns** if 2D detections are already near-optimal with 13 frames | Medium | Treat as a quick ablation; if gain < 0.5 mm, pivot to cross-view attention or uncertainty head. |
| **Overfitting** to longer temporal patterns on small MPI-INF-3DHP train set | Low–Medium | Keep 5-epoch schedule; monitor val error and stop early if it diverges. |
| **Fewer effective clips** per epoch due to longer windows | Low | `RandomClipDataset` still samples 4000 clips per sequence; longer clips increase context without reducing diversity. |
| **Evaluation mismatch** with current best (clip_len differs) | Low | Use the same `eval_ray_attention_temporal_residual_v1.py` protocol; report both clip-wise and per-frame aggregated metrics if needed. |

## Follow-up if Successful

- Extend to **clip_len=51** (~1.7 s) to find the point of diminishing returns.
- Combine with the uncertainty-aware residual head (`RayAttentionFusionModelTemporalResidualV3`) or the self-critic refiner (`RayAttentionFusionModelTemporalResidualCritic`) for further gains.
- Run the same `clip_len=27` setting on Human3.6M (17 joints, 4 views) to confirm cross-dataset benefit.
