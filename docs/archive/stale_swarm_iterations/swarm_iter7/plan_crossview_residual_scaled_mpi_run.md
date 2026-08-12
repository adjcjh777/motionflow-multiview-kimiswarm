# Plan: Cross-View + Residual Scaled Full Run on MPI-INF-3DHP

**Direction:** cross-view attention  
**Date:** 2026-08-04  
**Proposed by:** MotionFlow-MultiView planning swarm

## 1. Goal
Push the MPI-INF-3DHP cross-subject MPJPE below the current best of **11.17 mm** by scaling up the existing `RayAttentionFusionModelTemporalCrossviewResidual`, which jointly attends over time **and** camera views for each joint token.

## 2. What to change
Run a **full-data, scaled-capacity training** of `RayAttentionFusionModelTemporalCrossviewResidual`:

| Hyperparameter | Baseline residual best | This run | Rationale |
|---|---:|---:|---|
| `d` | 64 | **128** | Wider per-token features for richer cross-view/temporal fusion |
| `residual_hidden` | 128 | **256** | Larger residual MLP to model fine 3D corrections |
| `n_st_layers` | 2 | **3** | Deeper spatio-temporal transformer for higher-level reasoning |
| `clip_len` | 13 | **13** | Keep memory tractable with T·V=182 tokens per joint (MPI has 14 views) |
| `batch_size` | 8 | **4** | Compensate for doubled feature dimension |
| `train_samples` | 4000 | **8000** | More random clips to match larger capacity |
| `epochs` | 5 | **30** | Allow full convergence; early-stop on val MPJPE |

**Files / code touched:**
- `experiments/train_ray_attention_temporal_crossview_residual_mpiinf3dhp.py` — existing trainer already supports `--n_st_layers`, `--d`, `--residual_hidden`.
- Create `experiments/eval_ray_attention_temporal_crossview_residual_mpiinf3dhp.py` by adapting `eval_ray_attention_temporal_residual_v3.py` to load `RayAttentionFusionModelTemporalCrossviewResidual`.
- Output checkpoint: `outputs/crossview_residual_d128_h256_nst3_full.pth`

## 3. Why it should help
- **Cross-view attention** lets each joint token attend directly across all 14 views at every time step. Smoke tests showed a **5.45 mm gap** over the temporal-only residual model on the same smoke split.
- **Scaling capacity** (`d=128`, `h=256`, `n_st_layers=3`) gives the head enough expressiveness to exploit the richer spatio-temporal features, while the residual formulation keeps the problem low-dimensional and stable.
- The architecture is still lightweight: parameter count is estimated at **~350–400 k**, still well within the “small model” regime the paper highlights.

## 4. Commands

### Smoke / sanity run (1–2 epochs)
```bash
conda run -n mf python experiments/train_ray_attention_temporal_crossview_residual_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --d 128 --n_st_layers 3 --residual_hidden 256 \
    --batch_size 2 --train_samples 200 --epochs 2 \
    --output outputs/crossview_residual_d128_h256_nst3_smoke.pth
```

### Full training run
```bash
conda run -n mf python experiments/train_ray_attention_temporal_crossview_residual_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 128 --n_st_layers 3 --residual_hidden 256 \
    --batch_size 4 --train_samples 8000 --epochs 30 \
    --output outputs/crossview_residual_d128_h256_nst3_full.pth
```

### Evaluation
```bash
conda run -n mf python experiments/eval_ray_attention_temporal_residual_v3.py \
    --checkpoint outputs/crossview_residual_d128_h256_nst3_full.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 128 --residual_hidden 256 --batch_size 4 \
    --out outputs/crossview_residual_d128_h256_nst3_eval.json
```

**Note:** `eval_ray_attention_temporal_residual_v3.py` currently instantiates `RayAttentionFusionModelTemporalResidual`. Adapt the import/constructor to `RayAttentionFusionModelTemporalCrossviewResidual` (same args: `j`, `d`, `n_views`, `n_st_layers`, `residual_hidden`).

## 5. Expected metrics
| Metric | Current best (residual d64 h128) | Target for this run |
|---|---:|---:|
| MPJPE (mm) | **11.17** | **< 10.0** |
| PA-MPJPE (mm) | 8.24 | **< 7.5** |
| PCK@100mm | 1.000 | ≥ 1.000 |
| AUC (0–150 mm) | 0.926 | **> 0.935** |

A gain of ~1–2 mm is realistic given the smoke-test improvement and the capacity increase.

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **OOM on RTX 4090** with `d=128`, `T=13`, `V=14` | Medium | Reduce `batch_size` to 2 or `d` to 96 if OOM occurs. |
| **Overfitting / slower convergence** with larger model | Medium | Early-stop on val MPJPE; add cosine LR schedule (copy from v3 trainer) if validation plateaus. |
| **Smoke gain does not transfer** to full data | Medium | Run baseline temporal residual at same capacity as a fair comparison. |
| **No eval script for cross-view model** | Low | Create/adapt `eval_ray_attention_temporal_crossview_residual_mpiinf3dhp.py` before running. |
| **Long wall-clock time** due to T·V attention | Medium | Budget ~10–14 h for 30 epochs; run overnight. |

## 7. Follow-up if successful
If this run beats 11.17 mm, the next cross-view experiment is to merge it with the uncertainty-aware pooling from `RayAttentionFusionModelTemporalResidualV3`, which reached **9.72 mm** in smoke tests.
