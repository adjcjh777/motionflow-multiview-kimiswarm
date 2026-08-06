# Two-stage training for residual temporal ray-attention model

## Goal
Implement the two-stage training recipe that was previously failing:

1. **Pre-train** `RayAttentionFusionModelV4` on single-frame MPI-INF-3DHP.
2. **Initialize** the per-frame encoder of the residual temporal model from that checkpoint and fine-tune on video clips.

## What changed

### New model
- `motionflow_mv/fusion/ray_attention_temporal_residual_v2_model.py`
  - `RayAttentionFusionModelTemporalResidualV2` subclasses `RayAttentionFusionModelTemporalResidual`.
  - Replaces the raw 21-D camera embedding with the same **V4 normalised camera embedding** (13-D), so the per-frame encoder is bit-for-bit compatible with a `RayAttentionFusionModelV4` checkpoint.

### New training script
- `experiments/train_ray_attention_temporal_residual_v2_mpiinf3dhp.py`
  - **Stage 1**: trains `RayAttentionFusionModelV4` on random single frames.
  - Saves the best V4 checkpoint (default: `outputs/ray_attention_v4_mpiinf3dhp_pretrain.pth`).
  - **Stage 2**: builds `RayAttentionFusionModelTemporalResidualV2`, loads the V4 checkpoint into the per-frame encoder with `strict=False`, and fine-tunes on clips.
  - Supports `--stage1_checkpoint` to skip stage 1 and reuse an existing V4 checkpoint.

## Smoke test run
Executed on the small `_smoke.npz` files to verify the pipeline end-to-end while staying within the ≤10-epoch/≤30-minute GPU budget.

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_v2_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --stage1_epochs 2 --stage2_epochs 2 \
    --stage1_train_samples 200 --stage2_train_samples 200 \
    --batch_size 4 --d 32 \
    --stage1_output outputs/ray_attention_v4_mpiinf3dhp_pretrain_smoke.pth \
    --output outputs/ray_attention_temporal_residual_v2_mpiinf3dhp_smoke.pth
```

### Results
- **Stage 1**: V4 single-frame best val MPJPE = **29.96 mm** after 2 epochs.
- **Stage 2**: residual temporal model best val MPJPE = **16.33 mm** after 2 epochs.
- V4 checkpoint loaded successfully:
  - `Loaded V4 weights: missing=31, unexpected=0`
  - Missing keys are exactly the temporal/residual layers (`temporal_pos_embed`, `temporal_attn.*`, `residual_mlp.*`), which is expected.

No errors; the two-stage pipeline is functional.

## Files touched
- `motionflow_mv/fusion/ray_attention_temporal_residual_v2_model.py` (new)
- `experiments/train_ray_attention_temporal_residual_v2_mpiinf3dhp.py` (new)
- `docs/swarm_iter6/report_two_stage_residual_v2.md` (this file)

## Next steps / blockers
- None for the implementation; the pipeline runs.
- For a meaningful result on MPI-INF-3DHP (S2 Seq1), run the script on full sequences with the usual `--d 64` and 20–30 epochs per stage. The smoke run was kept short per the GPU resource constraints.
