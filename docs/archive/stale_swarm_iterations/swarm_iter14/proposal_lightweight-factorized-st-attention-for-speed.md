# Lightweight Factorized ST Architecture: Factor View and Temporal Attention for Speed

## 1. Problem

The existing factorized ST+PP model already factors view-level and temporal-level attention, but its first smoke test (d=32, residual_hidden=64, 500 training samples, 5 epochs) produced **57.68 mm MPJPE** on MPI-INF-3DHP — far from the 9.32 mm anchor — suggesting the current factorized blocks are too weak while still being slower than a simpler baseline because they use full `TransformerEncoderLayer` blocks at every stage.

## 2. Hypothesis

Replacing the full Transformer blocks in the factorized path with lightweight single-head attention + narrow FFN (or no FFN), adding a temporal-pooling downsampling stage, and keeping the residual correction and PP-correction heads will recover most of the anchor accuracy on a short smoke while cutting per-clip latency by ≥30% on an RTX 4090.

## 3. Method

### 3.1 Architecture changes

Create a new module:

- `motionflow_mv/fusion/ray_attention_temporal_crossview_factorized_lightweight_model.py`
  - Inherits from `RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint` (already has PP correction and residual head).
  - Replace the two `TransformerEncoderLayer` lists (`view_layers`, `temporal_layers`) with a single lightweight factorized block repeated `n_blocks` times (default 2):
    - **View attention**: single-head `MultiheadAttention(d, num_heads=1, batch_first=True)` + LayerNorm + 1-layer FFN (`d → d`) with residual.
    - **Temporal attention**: same as view but applied over time.
    - **Temporal pooling**: after the first block, optionally downsample temporal dimension by 2 with a strided D conv (`Conv1d(d, d, kernel_size=3, stride=2, padding=1)`) to cut temporal cost by half.
  - Keep the per-frame ray encoder and the residual/PP heads unchanged so the only new variables are the factorized layers.
  - Parameter target: ≤60 k params at d=32 (vs. 90.6 k in the failed smoke).

### 3.2 Loss / data changes

No new loss terms; the model reuses the existing MSE pose loss and optional reprojection loss already used by `train_factorized_pp_smoke_mpiinf3dhp.py`.

### 3.3 Training scripts

- New: `experiments/train_factorized_lightweight_pp_smoke_mpiinf3dhp.py`
  - Copy of `experiments/train_ray_attention_temporal_crossview_factorized_residual_mpiinf3dhp.py` but instantiates the lightweight model and adds `--temporal_pool` and `--n_blocks` flags.
- Modify: `experiments/benchmark_runtime.py`
  - Add a new model entry `RayAttentionFusionModelTemporalCrossviewFactorizedLightweightResidualPrincipalPoint` to the `_build_model` mapping so the speedup is measured with the same harness.

### 3.4 Exact code edits (to be applied after smoke approval)

1. Create `motionflow_mv/fusion/ray_attention_temporal_crossview_factorized_lightweight_model.py` with the `RayAttentionFusionModelTemporalCrossviewFactorizedLightweightResidualPrincipalPoint` class.
2. Copy `experiments/train_ray_attention_temporal_crossview_factorized_residual_mpiinf3dhp.py` → `experiments/train_factorized_lightweight_pp_smoke_mpiinf3dhp.py` and swap the model import/instantiation.
3. In `experiments/benchmark_runtime.py`, add the new model to `_build_model` and to the default `models` list.

## 4. Smoke-test plan

Run a 5-epoch smoke on MPI-INF-3DHP with the same small setting used for the failed factorized smoke:

```bash
python experiments/train_factorized_lightweight_pp_smoke_mpiinf3dhp.py \
    --train data/webbridge/mpi_train.npz \
    --val data/webbridge/mpi_val.npz \
    --clip_len 13 \
    --d 32 \
    --residual_hidden 64 \
    --n_blocks 2 \
    --temporal_pool True \
    --batch_size 8 \
    --train_samples 500 \
    --epochs 5 \
    --output outputs/factorized_lightweight_pp_smoke.pth
```

**Pass / fail criteria:**

- **Pass**: val MPJPE ≤ 15 mm after 5 epochs and no NaNs/crashes.
- **Pass**: end-to-end single-clip latency on RTX 4090 is ≥30% lower than the 90.6 k-param factorized baseline at the same `d`/`clip_len`.
- **Fail**: val MPJPE > 20 mm or latency reduction < 20%.

## 5. Evaluation plan

After the smoke passes, run the standard eval harness:

- Metrics: MPJPE, PA-MPJPE, PCK@50/100/150, AUC on MPI-INF-3DHP clean val.
- Scripts:
  - `python experiments/eval_full_metrics.py --checkpoint outputs/factorized_lightweight_pp_smoke.pth --model factorized_lightweight_pp`
  - `python experiments/benchmark_runtime.py --models RayAttentionFusionModelTemporalCrossviewFactorizedLightweightResidualPrincipalPoint --device cuda`
- Compare against the anchor (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, 9.32 mm) and the previous failed factorized smoke (57.68 mm).
- Acceptable target for full run: MPJPE ≤ 9.8 mm (≤0.5 mm above anchor) with ≥30% latency reduction.

## 6. Estimated GPU/CPU cost on RTX 4090

- Smoke (5 epochs, 500 samples, B=8): ~3–5 minutes on RTX 4090, <2 GB VRAM.
- Full run (30 epochs, 4000 samples): ~30–45 minutes, ~4 GB VRAM.
- Runtime benchmark: <2 minutes, single GPU inference only.
- CPU cost: data prep and report generation negligible (<1 minute).

## 7. Risks & fallback

- **Risk: accuracy still lags.** If the lightweight blocks are too shallow, the model may not converge in the smoke.
  - *Fallback*: Re-introduce the full `TransformerEncoderLayer` but reduce `n_blocks` to 1 and increase `d` to 48, or use a distilled MSE loss from the anchor teacher.
- **Risk: temporal pooling loses fine temporal detail.**
  - *Fallback*: Make pooling optional and default to off; instead use local-window temporal attention with window size 5 to keep complexity low.
- **Risk: PP correction destabilizes the lighter model.**
  - *Fallback*: Freeze PP correction for the first 3 epochs or disable focal correction (`focal_max_scale=0.0`) during the smoke.
- **Risk: speedup below 30%.**
  - *Fallback*: Profile with `torch.profile` and replace the remaining heavy attention with depthwise-separable temporal convolutions.
