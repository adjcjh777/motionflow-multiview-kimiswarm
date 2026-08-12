# v51 Long-Clip Efficient Temporal Memory (LET-Mem)

## Focus
Extend the v46/v47 chain to longer training/inference clips without quadratic `(time, joint)` memory growth, while preserving the self-evolution feedback loop and sparse-view gains.

## Architecture

**LET-Mem** replaces v47's full self-attention over `(T, J)` tokens with a two-level hierarchy:

1. **Local temporal perceiver (LTP):** each contiguous window of `v51_ltm_window_size` frames is compressed into `v51_ltm_latent_tokens` latent tokens via cross-attention. The query latents are learned; keys/values are the v46 reliability-weighted pose features flattened over the window. This reduces attention from `O((T*J)^2)` to `O((T/w)*L^2)` where `w` is the window size and `L` the number of latent tokens.

2. **Causal long-range memory bank (MB):** a single momentum-updated memory token per window stores slow context. The update rule is `m_t = α * m_{t-1} + (1-α) * Pool(LTP_t)`, with `α` from `v51_ltm_memory_update_rate`. The memory token is cross-attended by the current window’s latents to propagate long-range motion constraints without materializing the full long-range attention matrix.

3. **Reliability-conditioned gate:** v46 per-view reliability scores are pooled across the window and fed into a small 2-layer MLP to scale the memory update. This keeps the v50 self-evolution loop active over longer temporal horizons: views that the model currently distrusts contribute less to the memory bank.

4. **Decoder:** the latent tokens are cross-attended back to the original `(window, joint)` grid, producing per-frame 3-D pose estimates compatible with the existing `experiments/eval_variable_views.py` path.

All components are identity-at-init: the perceiver decoder is initialized so that, at startup, the module passes the v47 pose estimate through unchanged.

## New config flags and defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_long_clip_temporal_memory` | bool | `False` |
| `v51_ltm_window_size` | int | `5` |
| `v51_ltm_latent_tokens` | int | `8` |
| `v51_ltm_num_layers` | int | `2` |
| `v51_ltm_d_model` | int | `64` |
| `v51_ltm_num_heads` | int | `4` |
| `v51_ltm_memory_update_rate` | float | `0.5` |
| `v51_ltm_use_reliability_gate` | bool | `True` |
| `v51_ltm_local_attention_only` | bool | `False` |
| `loss.v51_ltm_temporal_consistency_weight` | float | `0.01` |

## Loss term

A lightweight temporal consistency loss enforces agreement between the current window prediction and the memory bank’s prediction for the same central frame:

```
L_ltm = loss.v51_ltm_temporal_consistency_weight * Huber(P_current, P_memory)
```

This is added to the existing v50 total loss. No change to the supervised MPJPE term is required.

## Evaluation metrics

- `val_MPJPE@full` on `clip_len = 25` and `51`
- `MPJPE@2`, `MPJPE@3`, `MPJPE@4` under v46 view-dropout
- Peak GPU memory (GB) and frames per second during validation
- Temporal jitter: per-joint acceleration smoothness metric

## Expected MPJPE impact

- `MPJPE@full` for `clip_len=25`: **-1.5 to -3.0 mm** vs. v47 baseline
- `MPJPE@2/3`: **-2.0 to -3.0 mm** thanks to longer temporal context and stable memory
- Short-clip (`clip_len=9`) performance: within **±0.5 mm** of v47
- Memory: **<50%** of v47 for `clip_len=25`; throughput: **+20–30%**

## Main risk

The memory bank can over-smooth fast motions or drift under domain shift, and causal momentum may introduce a small temporal lag for rapid accelerations. Mitigation: keep local window attention intact, warm-start from the best v47 checkpoint, and freeze the v46/v48 base for the first epoch while the memory bank initializes.