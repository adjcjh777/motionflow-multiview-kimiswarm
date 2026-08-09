# v51 Lightweight Temporal Transformer (LTT)

**Focus area:** `lightweight_temporal_transformer`  
**Module:** `LightweightTemporalTransformerV51` → `motionflow_mv/fusion/lightweight_temporal_transformer_v51.py`

## 1. Architecture

The v51 Lightweight Temporal Transformer replaces the heavier v47 temporal aggregation block with a causal, memory-efficient transformer that attends over time **per joint**, then mixes information **across joints** through a tiny shared MLP. This keeps the sparse-view generalization of v46 intact while adding stable temporal smoothing at a fraction of v47's cost.

- **Input:** a clip of per-frame 3-D pose estimates `P ∈ R^(T×J×3)` and optional per-joint uncertainty `σ ∈ R^(T×J)` from v37/v39/v50.
- **Temporal branch:** causal sliding-window self-attention (`window=W`) applied independently to each joint's temporal sequence, with shared weights across joints. Complexity is `O(T·W·J·d)` rather than v47's `O((T·J)^2·d)`.
- **Cross-joint branch:** a two-layer depth-separable MLP (or 1-D grouped conv over joints) fuses the temporally refined joint tokens.
- **Output:** a residual correction `ΔP` added to the input pose. The final projection layer is initialized to zero so the module is **identity-at-init** and cannot regress the v46/v48 baseline.

If v50 Self-Evolution Feedback Head is enabled, the temporal attention scale can be modulated by a frame-level residual score; the coupling is optional and read-only.

## 2. New config flags

| Flag | Type | Default |
|---|---|---|
| `use_v51_lightweight_temporal_transformer` | bool | `False` |
| `v51_ltt_d_model` | int | `64` |
| `v51_ltt_num_layers` | int | `2` |
| `v51_ltt_n_heads` | int | `4` |
| `v51_ltt_window` | int | `7` |
| `v51_ltt_dropout` | float | `0.1` |
| `v51_ltt_loss_weight` | float | `0.01` |
| `v51_ltt_smooth_weight` | float | `0.1` |
| `v51_ltt_use_residual_feedback` | bool | `False` |
| `v51_ltt_max_clip_len` | int | `25` |

## 3. Loss term

```
L_ltt = v51_ltt_loss_weight * [ L_mse(P_refined, P_gt) + v51_ltt_smooth_weight * L_smooth(P_refined) ]
```

`L_smooth` is the per-joint acceleration penalty `(P_{t+1} - 2P_t + P_{t-1})`, encouraging jitter-free motion without imposing a strong motion prior.

## 4. Evaluation metric

- `val_MPJPE@full` (clip-level)
- `MPJPE@k` for `k = 2,3,4` and full views
- Per-joint temporal jerk (mm/frame²)
- Peak GPU memory per training step and inference latency per clip

## 5. Expected MPJPE impact

- `val_MPJPE@full`: −0.8 to −1.5 mm versus v46 baseline
- `MPJPE@2/3/4`: −1.5 to −2.5 mm, because temporal smoothing recovers the noisiest sparse-view estimates
- Long-clip (`clip_len=25`): −2 to −3 mm with <50 % memory of v47's full joint×time attention

## 6. Main risk

**Over-smoothing at motion boundaries.** A small causal window can blur rapid motion (e.g., hand contacts, foot strikes). Mitigation: keep the residual gate zero-initialized, cap the window at `max(5, clip_len//4)`, and ablate `v51_ltt_window ∈ {5,7,9}` on the local smoke before committing to a full A800 run.

## 7. Integration sketch

In `motionflow_mv/fusion/omniview_fusion_v5.py`:

```python
if cfg.use_v51_lightweight_temporal_transformer:
    pose = ltt_v51(pose, uncertainty=sefh_uncertainty)
```

No existing flags are removed. The module is warm-startable from any v46/v48 checkpoint and can be smoke-tested on the local RTX 4090 before the v47/v48 A800 runs finish.

## 8. Smoke test plan

- **Config:** `configs/benchmark_v51_lightweight_temporal_transformer_smoke.yaml`
- **Command:** `bash scripts/run_v51_ltt_smoke_local_4090.sh`
- **Acceptance:** `val_MPJPE@full` within 1 mm of v46 smoke; `MPJPE@2` improves ≥ 1 mm; no NaN/OOM; per-step memory < 60 % of v47 smoke.
