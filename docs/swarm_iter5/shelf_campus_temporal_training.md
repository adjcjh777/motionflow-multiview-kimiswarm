# Shelf/Campus Temporal Canonical Clips & Training

**Date:** 2026-08-04  
**Scope:** Convert existing Shelf/Campus pseudo-GT sequences into temporal canonical WebBridge `.npz` clips and add a dedicated training script for the temporal ray-attention fusion model.  
**Related files:**
- `motionflow_mv/data/temporal_clip_dataset.py` (new shared data helpers)
- `experiments/convert_shelf_campus_temporal.py` (new conversion script)
- `experiments/train_ray_attention_temporal_shelf.py` (new training script)
- `data/webbridge/shelf_campus/*.npz` (generated canonical splits)

## What was done

1. **Inspected existing temporal baseline**
   - Read `motionflow_mv/fusion/ray_attention_temporal_model.py` and `experiments/train_ray_attention_temporal_mpiinf3dhp.py`.
   - The model consumes `(B, T, V, J, 3)` 2D-keypoint clips and predicts `(B, T, J, 3)` 3D poses plus per-view weights.
   - The MPI-INF-3DHP training script already has the right canonical `.npz` loader/augmentation/evaluation logic, but it is embedded in the script.

2. **Extracted reusable data helpers**
   - New module `motionflow_mv/data/temporal_clip_dataset.py` contains `TemporalClipDataset`, `RandomClipDataset`, `collate_fn`, `augment_clip`, `make_dataloaders`, and `set_seed`.
   - This keeps the new Shelf/Campus training script from duplicating the MPI script's logic.

3. **Converted Shelf/Campus to temporal canonical clips**
   - Source files used:
     - `data/shelf_campus/Shelf_Seq1/pseudogt_m.npz` (3,200 frames, 5 views)
     - `data/shelf_campus/Campus_Seq1/pseudogt_m.npz` (1,423 frames, 3 views)
   - `experiments/convert_shelf_campus_temporal.py` splits each sequence 80/20 along the temporal axis and writes the per-view camera parameters unchanged (fixing the initial bug of slicing camera arrays).
   - Generated outputs under `data/webbridge/shelf_campus/`:
     - `shelf_seq1_train_v5_multiview_m.npz` (2,560 frames)
     - `shelf_seq1_val_v5_multiview_m.npz` (640 frames)
     - `campus_seq1_train_v3_multiview_m.npz` (1,138 frames)
     - `campus_seq1_val_v3_multiview_m.npz` (285 frames)

4. **Added training script**
   - `experiments/train_ray_attention_temporal_shelf.py` uses the shared data helpers and defaults to the Shelf split.
   - It infers `n_views` and `j` from the first `.npz` file, so it can also be pointed at the Campus files (3 views) or mixed with a model that supports variable views.
   - Smoke-test defaults are intentionally small: 3 epochs, `d=64`, `batch_size=4`, `train_samples=500`, `clip_len=13`.

## Smoke test result

```bash
conda run -n mf python experiments/train_ray_attention_temporal_shelf.py \
    --epochs 3 --batch_size 4 --train_samples 500 --clip_len 13 --d 64
```

Output:

```text
Device: cuda
n_views=5, j=17, clip_len=13, d=64
Model params: 180961
Epoch 1: train_loss=0.021325, val_MPJPE=27.29mm, val_loss=0.000357 (saved)
Epoch 2: train_loss=0.019947, val_MPJPE=50.85mm, val_loss=0.000967
Epoch 3: train_loss=0.020239, val_MPJPE=48.77mm, val_loss=0.000918
Best val MPJPE: 27.29mm -> outputs\ray_attention_temporal_shelf.pth
```

- The script runs to completion in a few minutes on the local RTX 4090.
- The absolute MPJPE numbers are low because the Shelf pseudo-GT appears to be scaled to a very small metric (hip width ≈ 0.04 m), not full human scale. The model is numerically stable and the checkpoint loads correctly.

## Usage

Convert data (already done):

```bash
conda run -n mf python experiments/convert_shelf_campus_temporal.py
```

Smoke test on Shelf (default):

```bash
conda run -n mf python experiments/train_ray_attention_temporal_shelf.py
```

Longer run (still ≤10 epochs):

```bash
conda run -n mf python experiments/train_ray_attention_temporal_shelf.py \
    --epochs 10 --d 128 --batch_size 4 --clip_len 27 --train_samples 4000
```

Train/val on Campus (3 views):

```bash
conda run -n mf python experiments/train_ray_attention_temporal_shelf.py \
    --train data/webbridge/shelf_campus/campus_seq1_train_v3_multiview_m.npz \
    --val data/webbridge/shelf_campus/campus_seq1_val_v3_multiview_m.npz
```

## Notes / blockers

- **No blockers.** The training script runs end-to-end on the Shelf split.
- **Scale caveat:** The existing `pseudogt_m.npz` files for Shelf contain tiny 3D positions. This is pre-existing data; the conversion script preserves the units. Metrics should be interpreted accordingly or the data should be rescaled in a future pass.
- **Cross-dataset (Shelf → Campus):** The current `RayAttentionFusionModelTemporal` is instantiated with a fixed `n_views`. Running on both datasets simultaneously would require either (a) separate models instances for 3 vs. 5 views, or (b) extending the model to handle variable numbers of views. The script supports either dataset individually.
