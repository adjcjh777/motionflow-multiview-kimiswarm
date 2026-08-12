# Combined Temporal + Cross-View Ray-Attention Fusion (Swarm Iter 5)

## Goal
Build a single model where temporal tokens can attend across views, unifying the existing temporal ray-attention fusion model with cross-view attention.

## What was added
- **Model**: `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py`
  - Keeps the per-frame v3 encoder (observation + ray embeddings, camera-conditioned embedding, view-level self-attention, joint-level self-attention).
  - Replaces the per-(view,joint) temporal transformer with a **joint-wise spatio-temporal transformer** that attends jointly over the `(time, view)` grid for every joint.
  - Learns separate time and view positional embeddings.
  - Adds a small floor (`clamp(min=1e-4)`) on predicted weights before DLT to avoid all-zero-weight singular systems.
  - Output head and weighted DLT triangulation are unchanged from the temporal baseline.
- **Training script**: `experiments/train_ray_attention_temporal_crossview_mpiinf3dhp_v1.py`
  - Reuses the same dataset helpers and collate logic as the temporal baseline.
  - Defaults tuned for a short smoke run (`--epochs 3`, `--train_samples 1000`, `--batch_size 4`).
  - Augmentation is opt-in via `--augment` (see Notes).
- **Test**: `tests/test_ray_attention_temporal_crossview.py`
  - Forward/backward shape and gradient checks for 5D clips and 4D single-frame inputs.

## Architecture details
Input: `(B, T, V, J, 3)` containing `(x_pixel, y_pixel, confidence)`.

1. Per-frame v3 encoder → `(B, T, V, J, d)`.
2. Add learned time and view positional embeddings.
3. For each joint, reshape `(B, T, V, d)` → `(B, T·V, d)` and run `n_st_layers` of `TransformerEncoderLayer`. This is a single self-attention operation over **both** time and views, so temporal tokens can attend across views.
4. Reshape back to `(B·T, V, J, d)`, predict per-view weights, and triangulate.

## Smoke-run results

Command:
```bash
conda run -n mf python experiments/train_ray_attention_temporal_crossview_mpiinf3dhp_v1.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
           data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --epochs 3 --batch_size 8 --train_samples 500 --d 64
```

Results:

```
Device: cuda
n_views=14, j=28, clip_len=13, d=64
Model params: 218721
Epoch 1: train_loss=0.000431, val_MPJPE=25.09mm (saved)
Epoch 2: train_loss=0.000434, val_MPJPE=25.00mm (saved)
Epoch 3: train_loss=0.000432, val_MPJPE=25.00mm
Best val MPJPE: 25.00mm -> outputs\ray_attention_temporal_crossview_mpiinf3dhp.pth
```

The combined temporal+cross-view model reaches a very similar cross-subject MPJPE to the temporal-only baseline (~25 mm) after only 3 epochs.

## Notes / blockers

- **Augmentation**: Enabling the existing `augment_clip` (noise + confidence dropout + 100 px outliers) triggered sporadic `CUDA error: an illegal memory access was encountered` inside `torch.linalg.lstsq` / `torch.linalg.pinv` during the first validation pass. The same triangulation code works reliably on the baseline, so the instability appears specific to the spatio-temporal attention + heavy outliers. For the smoke run, augmentation is disabled by default; use `--augment` only if you want to debug / harden against this.
- **Training time**: ~5 minutes for 3 epochs on the RTX 4090 with the settings above.
- **Next steps**: compare a longer run with light augmentation (noise only), evaluate whether the cross-view attention yields larger gains on harder sequences, and consider a batched/vectorized DLT to remove the per-joint Python loop.
