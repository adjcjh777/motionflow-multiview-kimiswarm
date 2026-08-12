# Longer Temporal Window (>13 frames)

**Date:** 2026-08-04  
**Topic:** Extend temporal context for `RayAttentionFusionModelTemporalResidual`  
**Goal:** Beat the current MPI-INF-3DHP best of **11.17 mm** MPJPE by training with clips longer than the default 13 frames.

## 1. Current state

- **Best model:** `RayAttentionFusionModelTemporalResidual` in `motionflow_mv/fusion/ray_attention_temporal_residual_model.py:38`.
- **Current settings:** `clip_len=13`, `d=64`, `residual_hidden=128`, `n_temporal_layers=2` (~243 k params).
- **Verified results:** MPI-INF-3DHP S1→S2/Seq1 MPJPE **11.17 mm**, PA-MPJPE **8.24 mm**, AUC **0.9256** (`docs/swarm_iter7/verified_results.json`).
- **Temporal support:** The learned positional embedding is sized for `max_temporal_len=256` (`motionflow_mv/fusion/ray_attention_temporal_model.py:110`), so no code change is needed for `clip_len=27` or even `clip_len=51`.
- **Existing plan:** `docs/swarm_iter7/plan_longer_temporal_window.md` already proposed `clip_len=27`, `batch_size=4`, 5 epochs.
- **Prior run:** A 1-epoch smoke run with `clip_len=27` produced **27.15 mm** and left checkpoint `outputs/ray_attention_temporal_residual_clip27_smoke2.pth`; it was undertrained and not representative of full convergence.

## 2. Gap / opportunity

No full 5-epoch training/evaluation has been completed for any `clip_len > 13`. The smoke run does not rule out gains: longer windows (27 frames  0.9 s at 30 fps) provide more motion context, which can help the temporal transformer resolve short-term ambiguities such as occlusion, motion blur, and self-occlusion, while also reducing frame-to-frame jitter.

## 3. Concrete next step

Run the full `clip_len=27` experiment using the existing scripts and model (no core code changes required):

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 27 --d 64 --residual_hidden 128 --n_temporal_layers 2 \
    --epochs 5 --batch_size 4 --train_samples 4000 --lr 1e-3 \
    --output outputs/ray_attention_temporal_residual_clip27_full5.pth
```

Then evaluate with the full metrics script:

```bash
conda run -n mf python experiments/eval_ray_attention_temporal_residual_v1.py \
    --checkpoint outputs/ray_attention_temporal_residual_clip27_full5.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 27 --d 64 --residual_hidden 128 --batch_size 4 \
    --out docs/swarm_iter7/eval_residual_clip27_full5.json
```

If the full run improves over the baseline, follow up with `clip_len=51` and/or validate on H36M to confirm cross-dataset benefit.

## 4. Expected success metric

| Metric | Current best (clip_len=13) | Target (clip_len=27) |
|---|---:|---:|
| MPJPE (mm) | 11.17 | **< 11.17**, ideally 9.5–10.5 |
| PA-MPJPE (mm) | 8.24 | **< 8.24** |
| PCK@150 mm | 1.0000 | ≥ 1.0000 |
| PCK AUC (0–150 mm) | 0.9256 | **> 0.9256** |

## 5. Risks / blockers

| Risk | Mitigation |
|---|---|
| **Memory blow-up:** temporal self-attention scales as O(T²); T=27 uses ~4.3× memory vs. T=13. | Use `batch_size=4` (or drop to 2 with gradient accumulation if OOM). |
| **Diminishing returns:** 2D detections may already be near-optimal with 13 frames. | Treat as a focused ablation; if gain < 0.5 mm, pivot to cross-view attention or uncertainty head. |
| **Overfitting:** small MPI-INF-3DHP train set. | Keep 5-epoch schedule; monitor val MPJPE and stop early if it diverges. |
| **Read-only A800-D / Docker:** do not modify any Docker container or vendor data. | Run locally on the WSL RTX 4090 using already-downloaded WebBridge `.npz` files. |
| **Do not commit large files:** WebBridge data is in `data/webbridge/`. | Keep outputs in `outputs/` or `docs/swarm_iter7/` and do not commit raw data or checkpoints. |

---

*Next review: compare `docs/swarm_iter7/eval_residual_clip27_full5.json` against `docs/swarm_iter7/verified_results.json`.*
