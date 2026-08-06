# Swarm Iter 6 — Residual Temporal Model on CMU Panoptic Sample

## Task

Run the top-performing residual temporal ray-attention model
(`RayAttentionFusionModelTemporalResidual`) on the converted CMU Panoptic sample
and report MPJPE. Document blockers and limitations.

## Files touched / created

| Path | Purpose |
|------|---------|
| `experiments/eval_ray_attention_temporal_residual_panoptic.py` | **New** eval script for the residual model on a Panoptic canonical `.npz`. |
| `experiments/train_ray_attention_temporal_residual_panoptic_smoke.py` | **New** smoke-train script that fits the residual model to the Panoptic sample from scratch. |
| `outputs/ray_attention_temporal_residual_panoptic_smoke.pth` | Best checkpoint from the 10-epoch Panoptic smoke run. |
| `outputs/eval_residual_panoptic_smoke.json` | JSON metric summary produced by the eval run. |
| `docs/swarm_iter6/panoptic_residual_eval.md` | This report. |

## What we did

1. **Attempted direct evaluation of the MPI-INF-3DHP checkpoint.**
   The current best checkpoint `outputs/ray_attention_temporal_residual_v2.pth`
   was trained on MPI-INF-3DHP with **14 views** and **28 joints**.
   The converted Panoptic sample has **4 views** and **19 joints** (COCO19).
   Loading the checkpoint into a 4-view / 19-joint model fails with a size
   mismatch in `fusion_mlp.0.weight` (`torch.Size([64, 896])` vs the required
   `torch.Size([64, 256])`), because `fusion_mlp` input width is `d * n_views`.

2. **Smoke-trained a matching residual model on the Panoptic sample.**
   Because the released checkpoint is not directly compatible, we trained a
   `RayAttentionFusionModelTemporalResidual` from scratch on the Panoptic sample
   for 10 epochs as a smoke test (≈ 2 min on the local RTX 4090).
   - 80 frames used for training (augmented random clips).
   - 21 frames used for validation.
   - Model: `j=19`, `n_views=4`, `d=64`, `n_temporal_layers=2`, `residual_hidden=128`.
   - Best validation MPJPE during training: **0.38 mm**.

3. **Evaluated the smoke-trained checkpoint on the full Panoptic sample.**
   The trained model was run on all 89 non-overlapping clips of length 13.

## How to reproduce

### Direct checkpoint evaluation (blocked)

```bash
conda run -n mf python experiments/eval_ray_attention_temporal_residual_panoptic.py \
    --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
    --panoptic data/webbridge/panoptic/171204_pose1_sample/171204_pose1_sample_canonical.npz \
    --clip_len 13 --batch_size 4
```

This fails because the checkpoint was trained with 14 views and 28 joints, while
the Panoptic sample has 4 views and 19 joints.

### Smoke-train on Panoptic and evaluate

```bash
# Train (≤ 10 epochs,  2 min on RTX 4090)
conda run -n mf python experiments/train_ray_attention_temporal_residual_panoptic_smoke.py \
    --epochs 10 --batch_size 4 --train_samples 300

# Evaluate on the full Panoptic sample
conda run -n mf python experiments/eval_ray_attention_temporal_residual_panoptic.py \
    --checkpoint outputs/ray_attention_temporal_residual_panoptic_smoke.pth \
    --panoptic data/webbridge/panoptic/171204_pose1_sample/171204_pose1_sample_canonical.npz \
    --clip_len 13 --batch_size 4 \
    --out outputs/eval_residual_panoptic_smoke.json
```

## Results on CMU Panoptic (`171204_pose1_sample`)

- **Checkpoint:** `outputs/ray_attention_temporal_residual_panoptic_smoke.pth`
- **Dataset:** `data/webbridge/panoptic/171204_pose1_sample/171204_pose1_sample_canonical.npz`
- **Model:** `RayAttentionFusionModelTemporalResidual`, 202,468 parameters  
  `j=19`, `n_views=4`, `d=64`, `n_temporal_layers=2`, `residual_hidden=128`
- **Clips evaluated:** 89 non-overlapping clips of length 13

### Metric summary

| Metric | Value |
|----------|-------|
| MPJPE | **0.3819 mm** |
| PA-MPJPE | **0.3227 mm** |
| PCK@50 mm | **1.0000** |
| PCK@100 mm | **1.0000** |
| PCK@150 mm | **1.0000** |
| PCK-AUC (0–150 mm) | **0.9949** |

### Per-joint MPJPE (mm), COCO19 order

```
0.562, 0.212, 0.480, 0.526, 0.380, 0.461, 0.449, 0.233,
0.480, 0.397, 0.479, 0.568, 0.511, 0.285, 0.285, 0.240,
0.262, 0.212, 0.232
```

## Analysis

1. **The MPI-INF-3DHP checkpoint cannot be run on the Panoptic sample directly.**
   The checkpoint embeds the camera rig geometry (`fusion_mlp.0.weight` is sized
   for 14 views), so a 4-view Panoptic input is fundamentally incompatible.

2. **Smoke-training from scratch is possible but not meaningful as a
   generalization test.** The Panoptic sample is tiny (101 frames) and the 2D
   keypoints are noise-free projections of the 3D ground truth. The model quickly
   memorizes the sequence, producing an MPJPE below 0.4 mm.

3. **The near-perfect metrics confirm the data pipeline works end-to-end:**
   canonical `.npz` loading, temporal clip sampling, ray-aware attention,
   residual refinement, and DLT triangulation all function correctly on the
   Panoptic skeleton and 4-view rig.

## Limitations and caveats

- **Checkpoint/domain mismatch.** The existing top-performing checkpoint was
  trained on MPI-INF-3DHP (14 views, 28 joints) and is not transferable to the
  4-view, 19-joint Panoptic sample without architecture changes or joint/view
  mapping.
- **Extremely small dataset.** The sample has only 101 frames, so training from
  scratch is a smoke test at best.
- **Synthetic 2D detections.** The Panoptic `.npz` was created by projecting 3D
  ground truth into the HD cameras. There is no real detector noise, occlusion,
  or outlier corruption, so the error floor is artificially low.
- **No cross-skeleton mapping.** Even if the view count matched, the model was
  trained on a 28-joint MPI-INF-3DHP skeleton while Panoptic uses COCO19. A
  fair cross-dataset evaluation would require a joint-name mapping and ideally
  fine-tuning on a larger Panoptic split.
- **Flash-attention warning.** PyTorch emits a flash-attention warning during
  forward passes. It is harmless for these tiny clips and the model falls back
  to the default SDPA implementation.

## Blockers

The main blocker is that the MPI-INF-3DHP checkpoint (`v2`) is incompatible
with the 4-view Panoptic sample due to the `fusion_mlp` input size mismatch.
We worked around it by smoke-training a fresh model on the Panoptic sample.

## Next-step suggestions

- Train a residual model on MPI-INF-3DHP with the same 4-view subset and
  a 19-joint skeleton (or a mapped subset), then evaluate on Panoptic.
- Acquire/fold a larger Panoptic split (e.g., the full `171204_pose1` sequence)
  for a realistic fine-tuning/generalization experiment.
- Add a joint-name mapping between MPI-INF-3DHP and COCO19 so that a model
  trained on one skeleton can be evaluated fairly on the other.
