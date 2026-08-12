# v50 Efficient Temporal Attention for Long Clips (`EfficientLongClipAttentionV50`)

## 1. Architecture

`EfficientLongClipAttentionV50` replaces the v47 temporal transformer with a **memory-efficient, hierarchical temporal attention block** that scales to longer clips without quadratic memory growth. The input is the same (time, joint) token sequence produced by v46 sparse-view generalization. Internally, the module first pools tokens along the temporal axis with a lightweight strided Conv1D (stride 2), then runs a stack of local-window self-attention layers with a small set of learned **global anchor tokens**. Local attention is restricted to a symmetric window around each time step; the global anchors attend to all pooled positions and provide cross-window context. Multi-head attention uses grouped-query attention and optional FlashAttention-style memory-efficient kernels. A final linear upsample (via nearest-neighbor interpolation) maps back to the original temporal resolution. The module is identity-at-init: the strided path plus residual connection can fall back to the original v47 behavior when weights are near zero.

## 2. Config Flags

| Flag | Default | Description |
|---|---|---|
| `use_v50_efficient_temporal_attention` | `False` | Master switch for the module. |
| `v50_eta_window_size` | `7` | Temporal window for local self-attention (odd). |
| `v50_eta_num_global_tokens` | `4` | Number of learned global anchor tokens. |
| `v50_eta_use_flash_attention` | `True` | Use memory-efficient attention kernel when available. |
| `v50_eta_d_model` | `64` | Hidden dimension (matches v47 default). |
| `v50_eta_n_heads` | `4` | Number of attention heads. |
| `v50_eta_num_layers` | `2` | Number of local-window attention layers. |
| `v50_eta_pool_stride` | `2` | Temporal stride before global anchors. |
| `v50_eta_dropout` | `0.1` | Dropout rate inside attention/MLP blocks. |

## 3. Loss Term

A lightweight **temporal consistency loss** on the final 3-D pose sequence:

```
L_cons = (1/T) Σ_t || (p_{t+1} - 2p_t + p_{t-1}) ||²
L_total = L_base + v50_eta_temporal_consistency_weight * L_cons
```

- `v50_eta_temporal_consistency_weight`: default `0.01`. This penalizes large joint accelerations over the longer temporal window and is small enough to avoid smoothing away fast motion.

## 4. Evaluation Metric

Report the standard v46/v47 metrics: `MPJPE@k` for `k = 2, 3, 4` and full views. In addition, report:

- `MPJPE@2_long`: `MPJPE@2` evaluated on clips of length `clip_len >= 25` to validate long-clip gains.
- `peak_mem_GB`: peak GPU memory during a forward/backward pass on the longest tested clip.
- `latency_ms_per_frame`: wall-clock latency on a single RTX 4090 batch.

## 5. Expected MPJPE Impact

- `MPJPE@full`: within ±0.5 mm of the v47 baseline (no regression).
- `MPJPE@2` on `clip_len=25`: **−2 to −3 mm** versus v47 with the same clip length, because the local window captures cleaner temporal dynamics and global anchors summarize motion context.
- `MPJPE@2` on `clip_len=13`: **−1 to −2 mm** versus v47; smaller but positive.
- Memory: for `clip_len=25`, peak memory should be **< 60 %** of the v47 full-attention baseline, enabling full runs on the existing A800-D configuration.

## 6. Main Risk / Mitigations

| Risk | Mitigation |
|---|---|
| **OOM on very long clips** despite efficient attention. | Keep `v50_eta_window_size` small (7); use gradient checkpointing; cap `clip_len` at 49 for first smoke. |
| **Local window misses long-range motion context**, hurting fast actions. | Global anchor tokens (`v50_eta_num_global_tokens=4`) and hierarchical pooling aggregate cross-window information. |
| **FlashAttention kernel unavailable** in current environment. | Fallback to standard PyTorch attention when `v50_eta_use_flash_attention=False`; smoke both paths. |
| **Temporal consistency loss over-smooths** rapid motions. | Keep weight at `0.01`; ablate to `0.0` and `0.05` in smoke. |
| **Gradient instability from strided/interpolation path** at init. | Initialize strided convolutions to near-identity and add a residual bypass so the module is warm-start friendly from a v47 checkpoint. |
