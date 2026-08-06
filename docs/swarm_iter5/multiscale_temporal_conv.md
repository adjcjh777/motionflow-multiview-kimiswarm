# Multi-Scale Temporal Convolution for Ray-Aware Multi-View Pose

**Swarm iteration:** 5  
**Date:** 2026-08-04  
**Goal:** Replace the temporal transformer in `RayAttentionFusionModelTemporal` with a multi-scale 1-D temporal convolution head and compare short-training performance on MPI-INF-3DHP.

## What was implemented

1. `motionflow_mv/fusion/multiscale_temporal_conv_model.py`  
   - New `MultiScaleTemporalConvModel` keeps the same per-frame ray/view/joint encoder as the baseline, but the temporal mixing is done with a stack of `MultiScaleTemporalBlock`s.
   - Each block applies several parallel `Conv1d` branches with different kernel sizes / dilations, concatenates their outputs, and projects back to the model dimension with a residual connection and layer norm.
   - Default temporal stack: 2 blocks × kernel 3 with dilations `[1, 2, 4]`.

2. `experiments/train_multiscale_temporal_mpiinf3dhp.py`  
   - Training script that reuses the data pipeline from `train_ray_attention_temporal_mpiinf3dhp.py` and swaps in the new model.

3. `tests/test_multiscale_temporal.py`  
   - Forward/backward sanity checks and a custom kernel/dilation configuration test.

## Training setup

Because the GPU is shared with many other swarm agents, the smoke run uses a **250-frame subset** of each canonical MPI-INF-3DHP meters file to keep training short and avoid queueing behind large data loads:

- Train: `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz` + `s_01_seq_02_v14_multiview_m_smoke.npz`
- Val:   `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz`
- `clip_len=13`, `d=64`, `n_temporal_layers=2`, `lr=1e-3`
- Smoke run: **2 epochs** on the 250-frame subsets (reduced to avoid GPU contention with other swarm agents)

Commands:

```bash
# Baseline (temporal transformer)
conda run -n mf python experiments/train_ray_attention_temporal_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --epochs 2 --batch_size 4 --train_samples 200 --output outputs/baseline_temporal_smoke.pth

# Multi-scale temporal convolution
conda run -n mf python experiments/train_multiscale_temporal_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --epochs 2 --batch_size 2 --train_samples 100 --output outputs/multiscale_temporal_conv_smoke.pth
```

## Results

| Model                     | Epochs | Train config (subset)               | Val MPJPE (mm) | Params  | Checkpoint |
|---------------------------|--------|--------------------------------------|----------------|---------|------------|
| Transformer baseline      | 2      | `train_samples=200`, `batch_size=4`  | **29.94**      | 217,825 | `outputs/baseline_temporal_smoke.pth` |
| Multi-scale temporal conv | 2      | `train_samples=100`, `batch_size=2`  | **29.96**      | 249,569 | `outputs/multiscale_temporal_conv_smoke.pth` |

On the same 250-frame MPI-INF-3DHP smoke subsets, the multi-scale temporal convolution model matches the transformer baseline within 0.02 mm after two short epochs. The transformer was trained with double the random clips and twice the batch size, so the comparison is rough, but it shows the conv head is at least competitive and does not collapse.

Parameter counts for the MPI-INF-3DHP configuration (J=28, d=64, V=14, 2 temporal layers):

- Transformer baseline: **217,825** params
- Multi-scale temporal conv: **249,569** params (slightly larger because of the per-dilation branches)

The prior reported baseline smoke result on the *full* sequences was **25.25 mm** after 2 epochs; the higher numbers here are expected because the smoke subset is smaller and noisier.

## Files changed / added

- `motionflow_mv/fusion/multiscale_temporal_conv_model.py`
- `experiments/train_multiscale_temporal_mpiinf3dhp.py`
- `tests/test_multiscale_temporal.py`
- `docs/swarm_iter5/multiscale_temporal_conv.md`

No existing working files were modified.

## Notes / follow-ups

- The per-frame encoder is duplicated from the baseline to avoid touching the working v3/temporal model. A future refactor could factor out the shared per-frame encoder.
- The convolution head has no attention across the `(V, J)` grid; it only mixes temporal context per view-joint token. This is a deliberate first-step comparison to isolate the effect of the temporal operator.
- If the conv head underperforms, next steps could include:
  - Adding a channel-wise / view-joint mixing step after temporal convs.
  - Using larger receptive fields (e.g., kernel 5 or 7) or gated temporal convolutions.
  - Investigating causal vs symmetric padding for online inference.

## Dependencies

No new dependencies beyond the existing `mf` conda environment (PyTorch, NumPy).
