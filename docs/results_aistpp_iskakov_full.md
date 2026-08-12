# Iskakov et al. (ICCV 2019) Learnable Triangulation – Full AIST++ Val Set

> **Date:** 2026-08-11
> **Status:** Complete
> **Reference:** Iskakov, K., Burkov, E., Lempitsky, V., Malkov, Y.,
> 'Learnable Triangulation of Human Pose', ICCV 2019, arXiv:1905.05754.

## Protocol

- **Data:** canonical AIST++ multi-view `.npz` files under `data/webbridge/aistpp_canonical/`.
- **Split:** `configs/splits/webbridge_aistpp_train_val.yaml` — 1280 train clips, 128 val clips (≈90/10).
- **Format:** 9 views, 17 joints (H36M skeleton), 720 frames per clip, metres.
- **Views kept:** all 9 views; no sparse-view subsampling.
- **Model:** `motionflow_mv/fusion/iskakov_learnable_triangulation.py` — re-implementation of the paper's weight-prediction branch, 1,569 parameters (cross-view, hidden_dim=32).
- **Training:** `experiments/train_iskakov_aistpp_full.py`.
  - AdamW, lr=1e-3, weight decay=1e-4, cosine annealing, grad clip=1.0.
  - Batch size 32, 4,096 training samples per epoch (~0.4% of the 1,016,604 train frames).
  - Early-stop patience 3 on combined direct val MPJPE, max 10 epochs.
  - Seed 20260811.
- **Hardware:** CPU (Intel oneMKL) on the local RTX 4090 workstation.  The GPU was busy with other training runs at start time, so the run was executed on CPU with double-precision DLT for numerical stability.
- **Pre-processing:** Canonical AIST++ uses `NaN` for missing 2D detections and missing 3D ground-truth.  Missing 2D points are zeroed and their confidences set to 0; frames with any NaN GT are excluded from training and metrics.

## Run command

```bash
python -u experiments/train_iskakov_aistpp_full.py \
    --device cpu \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-3 \
    --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --patience 3 \
    --seed 20260811 \
    --log_path outputs/iskakov_learnable_tri_aistpp_full.log \
    --ckpt_path outputs/iskakov_learnable_tri_aistpp_full.pth
```

## Results

| Method | Val direct MPJPE (mm) | Val root MPJPE (mm) | Notes |
|---|---:|---:|---|
| Unweighted DLT (frozen) | 53.72 | 46.63 | Baseline triangulation, no learning |
| Confidence-weighted DLT (frozen) | 34.64 | 32.79 | Weights = detection confidences |
| **Iskakov learned weights** | **29.27** | **26.03** | Best epoch 10/10 |

### Gains vs frozen baselines

- Learned vs unweighted DLT (direct): **+24.45 mm** improvement.
- Learned vs confidence-weighted DLT (direct): **+5.37 mm** improvement.

### Training trajectory

| Epoch | Train direct (mm) | Train root (mm) | Val direct (mm) | Val root (mm) |
|---:|---:|---:|---:|---:|
| 1 | 20.62 | 17.35 | 45.50 | 38.92 |
| 2 | 13.87 | 13.05 | 38.39 | 34.19 |
| 3 | 9.03 | 8.63 | 34.09 | 30.75 |
| 4 | 7.27 | 7.39 | 31.66 | 28.18 |
| 5 | 5.95 | 5.96 | 30.53 | 27.19 |
| 6 | 9.33 | 9.32 | 29.88 | 26.50 |
| 7 | 5.56 | 5.75 | 29.50 | 26.28 |
| 8 | 6.10 | 6.26 | 29.37 | 26.05 |
| 9 | 11.07 | 14.79 | 29.28 | 26.05 |
| 10 | 5.43 | 5.58 | 29.27 | 26.03 |

The model continued to improve monotonically in val MPJPE through epoch 10; no early stop was triggered.

## Interpretation

1. **The learned weights meaningfully outperform both frozen DLT baselines** on the full 128-clip AIST++ val set, beating the strong confidence-weighted DLT by 5.37 mm direct MPJPE and the unweighted DLT by 24.45 mm.
2. **Absolute error is low** (29.27 mm direct, 26.03 mm root-aligned) for a pure 2D-to-3D triangulation baseline with only 1,569 parameters, confirming that AIST++'s canonical multi-view detections are clean.
3. **Training was data-limited:** the model improved steadily across all 10 epochs on only 4,096 sampled frames per epoch (≈4k out of ~1M).  Longer training or more aggressive data sampling may still yield small gains.
4. **CPU execution worked but required double precision** because oneMKL's float32 `torch.linalg.lstsq` produced NaNs on the AIST++ NaN-masked inputs.  On a free GPU the same script runs in float32 without this workaround.

## Evidence

| Artifact | Path |
|---|---|
| Log | `outputs/iskakov_learnable_tri_aistpp_full.log` |
| Checkpoint | `outputs/iskakov_learnable_tri_aistpp_full.pth` |
| Config + history JSON | `outputs/iskakov_learnable_tri_aistpp_full.config.json` |
| Trainer | `experiments/train_iskakov_aistpp_full.py` |
| Runner script (local) | `scripts/run_iskakov_aistpp_full_local_4090.sh` |
| Runner script (A800 GPU 6) | `scripts/run_iskakov_aistpp_full_a800_gpu6.sh` |

> **Note:** The A800 GPU 6 script is prepared but not yet launched. As of 2026-08-12 the A800 GPU 6 run has no output artifacts.
