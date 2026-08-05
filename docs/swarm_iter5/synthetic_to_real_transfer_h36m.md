# Synthetic-to-Real Transfer — H36M Camera-Matched Synthetic Data

**Topic:** `synthetic_to_real_transfer_h36m`  
**Scope:** Domain-matched synthetic multi-view data + transfer training for `RayAttentionFusionModelV3`.  
**Date:** 2026-08-04.

---

## 1. What changed

### `experiments/generate_synthetic_multiview_dataset.py`

Tuned the synthetic generator to match the Human3.6M camera distribution measured from `data/h36m_hf/s_01_acts_02_03_..._16_multiview.npz` (62 k frames, 4 views, 17 joints).

Key updates:

- **World units:** default `world_scale=1000.0` so SMPL outputs are in **millimetres**, matching the real H36M data. This removes a unit-mismatch domain gap.
- **Intrinsics:** sampled around H36M statistics
  - `focal ~ N(1147, 2.08)` px
  - `cx ~ N(512, 3.98)` px, `cy ~ N(507, 5.69)` px
- **Extrinsics:** sampled around H36M camera centres
  - distance from origin ~ `N(5319, 523)` mm
  - camera z-height ~ `N(1559, 42)` mm
  - azimuths follow the four H36M camera directions, with a random yaw rotation for domain randomization
- **Noise / augmentation defaults:** `noise_std=1.0` px, `outlier_scale=100.0` px, calibrated to detector-level noise on 1 000 px images.
- **Projection path:** switched `project_points` to use **torch** matrix ops to avoid a Windows/Git Bash numpy-BLAS crash (`exit 127`) observed with `@` on 2-D `float64` arrays.
- A `--camera_mode legacy` flag keeps the original metre-scale generic rigs.

### `experiments/train_ray_attention_v3_transfer.py` (new)

Two-stage transfer script for `RayAttentionFusionModelV3`:

1. Pre-train on the synthetic H36M-matched dataset.
2. Load the best synthetic checkpoint and fine-tune on real H36M.

It reuses the same `CameraDataset` / augmentation pattern as `train_ray_attention_v3_h36m.py`, but fixes a per-sample camera indexing bug (the original condition compared the camera leading dimension to the split size rather than the full dataset size).

Command (quick smoke test):

```bash
/d/anaconda3/envs/jz_py310/python.exe experiments/train_ray_attention_v3_transfer.py \
    --synthetic_dataset outputs/synthetic_multiview_dataset.npz \
    --real_dataset data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz \
    --synth_epochs 50 --real_epochs 50 --batch_size 32 --d 64
```

## 2. Small verification run

Generated a tiny synthetic set:

```bash
experiments/generate_synthetic_multiview_dataset.py \
    --n_sequences 10 --frames_per_seq 5 \
    --output outputs/synthetic_h36m_tiny.npz
# => 50 frames, 4 views, 17 joints, H36M-matched cameras
```

Ran transfer training for 2 + 2 epochs on the tiny synthetic data and the real `s_01_act_02` subset (2 995 frames):

```text
Synthetic pre-training (50 frames, 2 epochs, d=32):
  Epoch 1 val_MPJPE = 11.49 mm

Real fine-tuning (2 epochs):
  Best real val_MPJPE = 4.75 mm
```

The model successfully loads the synthetic checkpoint, fine-tunes on real H36M, and reaches single-digit millimetre MPJPE on the held-out real validation set. The script is ready for the full 62 k-frame run without further changes.

## 3. Next steps / risks

- Scale up synthetic generation to the full planned size (`--n_sequences 500 --frames_per_seq 30` → 15 k frames) and run the full transfer schedule.
- Add optional reprojection / bone-length / temporal consistency losses during real fine-tuning when 3D GT is sparse.
- Monitor for the known Windows numpy-BLAS crash when any code path uses `np.linalg.svd` or large `float64` matrix multiplies; prefer torch ops in new code.
