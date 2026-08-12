# Cross-View Transformer Variant for MPI-INF-3DHP

**Date:** 2026-08-04
**Author:** Kimi Code sub-agent
**Goal:** Implement an explicit pairwise cross-view attention block in the per-frame encoder and smoke-train on MPI-INF-3DHP.

## What was implemented

- `motionflow_mv/fusion/ray_attention_crossview_model.py`
  - New `RayAttentionFusionModelCrossView` model.
  - Adds `CrossViewPairAttention` after the existing view-level self-attention.
  - For every joint, the block builds V x V pairwise view tokens, adds a
    learnable pair positional embedding, runs a transformer encoder over the
    V^2 tokens, and max-pools over the source-view dimension to refine the
    target-view features.
  - Uses a bottleneck dimension (`d_cross=32`) for the cross-view block to keep
    memory and computation modest on the RTX 4090, then projects back to the main
    dimension `d`.
  - Keeps the temporal transformer and weighted DLT triangulation from the
    temporal baseline.

- `experiments/train_ray_attention_crossview_mpiinf3dhp.py`
  - Mirrors the temporal training script but loads `RayAttentionFusionModelCrossView`.
  - Adds `--d_cross` (default 32) and `--n_crossview_layers` (default 1) flags.
  - Adds `--val_max_clips` for fast smoke tests that limit validation clips.

- `tmp/test_crossview_model.py`, `tmp/debug_crossview_cuda.py`
  - Sanity checks for output shapes and CUDA backward pass.

## Smoke-test result

Run command (2 epochs, 200 random train clips, 100 val clips, batch size 4):

```bash
conda run -n mf python experiments/train_ray_attention_crossview_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --epochs 2 --batch_size 4 --train_samples 200 --val_max_clips 100 \
    --output outputs/ray_attention_crossview_mpiinf3dhp.pth
```

Output:

```
Device: cuda
n_views=14, j=28, clip_len=13, d=64
Model params: 238913
Epoch 1: train_loss=0.027760, val_MPJPE=29.68mm (saved)
Epoch 2: train_loss=0.027676, val_MPJPE=29.68mm
Best val MPJPE: 29.68mm -> outputs\ray_attention_crossview_mpiinf3dhp.pth
```

For reference, the temporal baseline run under the same limited setting
(`--train_samples 200 --val_max_clips 100`) was not executed; however, the prior
full-val 1-epoch temporal baseline run reported **25.26 mm** on S2 Seq1. The
cross-view variant is therefore slightly behind in this very short smoke test,
likely because the pairwise V x V transformer adds capacity that needs more
epochs to converge.

## Implementation notes and blockers

- **OOM on first attempt:** The naive implementation with `d_cross=d=64` and all
  pairs flattened to `(N*J, V*V, d)` caused an out-of-memory error during the
  optimizer step on the RTX 4090. Reducing the cross-view bottleneck to
  `d_cross=32` resolved the OOM.
- **Speed:** The V x V pairwise transformer (sequence length 196) is
  significantly slower than the view-level self-attention. This is expected;
  full pairwise attention trades compute for explicit cross-view reasoning.
- **Validation limit:** `--val_max_clips 100` was added to keep the smoke test
  under the 10-minute tool timeout. A full validation run would be needed for
  a proper comparison.
- **No new dependencies:** The implementation only uses existing project
  dependencies (`torch`, `numpy`).

## Follow-up suggestions

- Train for more epochs (e.g., 10-30) with the full validation set to see if the
  cross-view block converges below the temporal baseline.
- Try cross-view attention variants that are cheaper than the full V x V
  transformer, such as pairwise MLP aggregation or graph attention across
  neighboring views.
- Experiment with `d_cross` (16, 32, 64) and the number of cross-view layers to
  balance capacity and speed.
