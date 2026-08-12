# Robustness Training Curriculum for Ray-Attention v4 (H36M)

## Task

Design a noise/outlier schedule that increases corruption over epochs and
implement it in `experiments/train_ray_attention_v4_h36m.py`.

## Design

The v4 trainer keeps the v3 `RayAttentionFusionModelV3` architecture and adds a
`RobustnessCurriculum` that progressively increases three corruption modes during
training:

1. **2D Gaussian noise** on keypoint coordinates (`noise_std`, px).
2. **View dropout** that zeros confidence for randomly selected views (`dropout_rate`).
3. **Sparse 2D outliers** that replace a fraction of observations with large
   random offsets (`outlier_rate` + `outlier_scale`, px).

The schedule holds corruption at a minimum for `warmup_epochs`, then linearly
ramps each parameter to its maximum over the remaining epochs.  This lets the
network first learn the clean geometric triangulation problem before being
exposed to heavier real-world noise.

## CLI knobs

| Arg | Default | Description |
|-----|---------|-------------|
| `--warmup_epochs` | 5 | Clean-training epochs before ramping. |
| `--noise_std_min`/`_max` | 0.0 / 5.0 | Gaussian noise std in px. |
| `--dropout_rate_min`/`_max` | 0.0 / 0.3 | Fraction of views dropped. |
| `--outlier_rate_min`/`_max` | 0.0 / 0.05 | Fraction of observations corrupted. |
| `--outlier_scale_min`/`_max` | 50.0 / 100.0 | Outlier magnitude in px. |
| `--val_*` | see script | Fixed high corruption for robustness validation. |

## Validation protocol

Two validation passes are run each epoch:

* **Clean validation**: no corruption; reports reconstruction quality.
* **Corrupted validation**: fixed high corruption (`val_noise_std=5`,
  `val_dropout_rate=0.2`, `val_outlier_rate=0.05`); reports robustness.

The checkpoint is saved based on the best *clean* validation MPJPE, while the
corrupted validation metric is logged for tracking.

## Data

Default dataset:
`data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz`
(~62 k frames, 4 views, 17 joints).

## Usage

```bash
/d/anaconda3/envs/jz_py310/python.exe experiments/train_ray_attention_v4_h36m.py \
    --dataset data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz \
    --epochs 50 --lr 1e-3 --d 64 --batch_size 32
```

## Important findings (preliminary)

* Curriculum training should be evaluated against the v3 constant-augmentation
  baseline on the same H36M split to isolate the benefit of progressive
  corruption.
* The clean validation MPJPE is used for checkpoint selection to avoid
  overfitting to synthetic corruptions.
* No long training was launched; only the script and a short verification run
  were produced.
