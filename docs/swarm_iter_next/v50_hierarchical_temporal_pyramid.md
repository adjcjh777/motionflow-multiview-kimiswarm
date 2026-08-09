# Hierarchical Temporal Pyramid v50

## Module Name

`HierarchicalTemporalPyramidV50`

## Architecture Description

Replace v47's single-scale temporal transformer with a **multi-resolution temporal pyramid** that processes pose tokens at three temporal scales: fine (full clip resolution), medium (stride-2 downsampled), and coarse (stride-4 downsampled). At each level, a lightweight transformer encoder block with the same `d_model` operates on its downsampled sequence; coarse levels therefore see larger effective receptive fields without increasing sequence length. Cross-scale fusion is performed by nearest-neighbor upsampling followed by element-wise addition and a final residual MLP, so the module is identity-at-init when warm-started from a v47 checkpoint. Per-level layer normalization and causal padding preserve streaming compatibility, and the whole block is inserted after v46 sparse-view generalization in the same slot currently occupied by v47 temporal aggregation.

## New Config Flags

```yaml
use_v50_hierarchical_temporal_pyramid: false
v50_htp_levels: 3                         # fine / medium / coarse
v50_htp_d_model: 64                       # per-level hidden dim
v50_htp_num_layers_per_level: 2           # transformer layers in each pyramid level
v50_htp_temporal_kernel: 3                # causal conv kernel for downsampling/upsampling
v50_htp_use_pyramid_loss: true
v50_htp_pyramid_loss_weight: 0.1          # total auxiliary weight across all levels
```

## Loss Term

`L_htp = v50_htp_pyramid_loss_weight * Σ_l w_l * MPJPE(pred_l, y_gt)`

The coarse level carries weight `0.05`, medium `0.03`, and fine `0.02`. This forces intermediate pyramid representations to be meaningful pose estimates rather than relying solely on the final fused output. The main pose loss is unchanged.

## Evaluation Metric

Report `val_MPJPE@full`, `MPJPE@2`, `MPJPE@3`, and `MPJPE@4` using the canonical `MPJPE@k` protocol, plus a new per-sequence **temporal consistency score** measuring the average 3-D joint acceleration error relative to ground truth (`mean_jerk_error` in mm/frame²). The primary success gate is `val_MPJPE@full` within 1 mm of the v47 baseline and improved `MPJPE@2/3/4` on dynamic action subsequences.

## Expected MPJPE Impact

Based on v46-SVG local smoke at 32.97 mm and the v47 temporal aggregation queued as the immediate baseline, HTP v50 should improve motion-heavy clips where single-scale attention under-smooths or misses periodic structure. Expected deltas on the local smoke scale: `MPJPE@full` -1.2 to -2.0 mm, `MPJPE@2` -2.5 to -3.5 mm, and `MPJPE@3/4` -1.5 to -2.5 mm. Full-scale A800 gains may be smaller (≈0.5–1.0 mm) but should hold because the pyramid explicitly regularizes temporal consistency.

## Main Risk / Mitigations

* **Risk: Multi-scale structure increases memory and wall-clock latency.** Mitigation: keep each level shallow (2 layers), downsample temporally rather than duplicating full compute, and strided attention at coarse levels. Profile on RTX 4090 smoke before scaling. If latency exceeds 50 ms/frame, drop the coarse level and fuse only fine+medium.
* **Risk: Extra loss weights destabilize v47 warm-start or cause gradient interference.** Mitigation: initialize the module as identity (zero out cross-scale residual branches), freeze the pyramid for the first 500 steps, and linearly ramp `v50_htp_pyramid_loss_weight` from 0 to 0.1 over the first epoch. Monitor per-level MPJPE to detect collapse to the coarse level only.
* **Risk: Pyramid overfits on slow/static clips and regresses v47's full-view accuracy.** Mitigation: clamp the module output with a residual gate (`v50_htp_output_alpha` initialized to 0), so the base v47 output is preserved at the start of training, and enable the gate only after the smoke shows no regression.

## Smoke Plan

1. Run on the local RTX 4090 with `d=64`, `clip_len=9`, `train_samples=500`, 2–5 epochs warm-started from the best v47 checkpoint.
2. Success gates: `val_MPJPE@full` within 1 mm of v47; `MPJPE@2` improves by ≥2 mm; no NaN/OOM.
3. If smoke passes, add an A800 queue entry with `d=128`, `clip_len=13`, full manifest, and label the issue `experiment` / `P1-next`.
