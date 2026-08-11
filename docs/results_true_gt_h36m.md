# H36M True-GT Standard Protocol Leaderboard

> Standard protocol: **S1, S5, S6, S7, S8 train → S9, S11 test**  
> Labels: `data/h36m_true_gt/*_multiview_m.npz` (true mocap world coordinates, non-circular).  
> Manifest: `configs/splits/h36m_true_gt_standard.yaml`.  
> Last updated: **2026-08-11**.

## Current results

| Method | S9 direct (mm) | S11 direct (mm) | Combined direct (mm) | Combined PA-MPJPE (mm) | Notes |
|---|---:|---:|---:|---:|---|
| DLT (unweighted) | 33.61 | 24.77 | 29.19 | 29.31 | frozen reference |
| DLT (confidence-weighted) | 29.82 | 21.91 | 25.87 | 25.55 | frozen reference |
| **Iskakov ICCV 2019** | **27.10** | **19.60** | **23.35** | **23.10** | best run, epoch 4 |
| v80 (reg v3) | — | — | **42.60** | — | best epoch 2 |
| v80 (smoke) | — | — | **98.12** | — | 2-epoch smoke only |
| **v80 (medium)** | — | — | **39.98** | — | best epoch 4; diverged afterward |
| **v25** | **67.92** | **77.68** | **72.80** | — | best val epoch 2; diverged to 207.62 mm by epoch 8 |

- Iskakov outperforms both DLT variants by a clear margin on the true-GT protocol.
- v80 has been swept on A800 with several recipes; the best converged result is **39.70 mm** (v2, epoch 2, checkpoint on A800), while the local copy gives **42.60 mm** (v3). A new local medium run reached **39.98 mm** (epoch 4). All v80 recipes overfit after the best epoch.
- v25 medium run finished at **72.80 mm** (best epoch 2/8) but diverged afterward to **207.62 mm** by epoch 8.

## AIST++ smoke (cross-dataset sanity)

AIST++ uses the same 17-joint skeleton as H36M and 9 calibrated views. The smoke split below trains on `gBR_sBM_cAll_d04_mBR0_ch01/ch02` and validates on `gBR_sBM_cAll_d04_mBR0_ch03` (see `configs/splits/aist_only_smoke.yaml`).

| Method | val MPJPE (mm) | Notes |
|---|---:|---|
| DLT (unweighted) | **12.66** | frozen reference |
| DLT (confidence-weighted) | **6.52** | frozen reference |
| Iskakov ICCV 2019 | **9.31** | best epoch 6, CPU smoke |
| v25 | **71.79** | 3-epoch smoke |
| v80 | **76.34** | 3-epoch smoke |

- The geometric baselines are very strong on AIST++: confidence-weighted DLT is already below 7 mm, and Iskakov reaches ~9 mm.
- v25/v80 smoke results are far behind the geometric baselines, suggesting the learned models have not yet adapted to AIST++'s camera rig / motion style. These are 3-epoch smoke runs only; full medium runs are needed before drawing firm conclusions.
- Numbers are comparable to the H36M true-GT scale, confirming AIST++ is a viable, non-circular cross-domain dataset.
- Source logs: `outputs/iskakov_aist_smoke.log`, `outputs/omniview_fusion_v25_aist_only_smoke.log`, `outputs/omniview_fusion_v80_aist_only_smoke.log`.

## Per-method details

### DLT baselines

Computed by the Iskakov baseline script with deterministic stride sampling (`ref_max_frames=2000`).

| Reference | S9 direct | S11 direct | Combined direct |
|---|---:|---:|---:|
| Unweighted DLT | 33.61 mm | 24.77 mm | 29.19 mm |
| Confidence-weighted DLT | 29.82 mm | 21.91 mm | 25.87 mm |

### Iskakov learnable triangulation

Best run:

```bash
python experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m --epochs 10 --batch_size 8 --lr 1e-3 --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --log_path outputs/iskakov_h36m_true_gt.log \
    --ckpt_path outputs/iskakov_h36m_true_gt.pth
```

| Epoch | Combined direct | S9 direct | S11 direct |
|---|---:|---:|---:|
| 1 | 23.40 mm | 27.12 mm | 19.68 mm |
| 2 | 23.37 mm | 27.11 mm | 19.63 mm |
| 3 | 23.37 mm | 27.12 mm | 19.62 mm |
| 4 (best) | **23.35** | **27.10** | **19.60** |
| ... | stable | stable | stable |
| 10 | 23.37 mm | 27.13 mm | 19.62 mm |

- Early-stopped by patience, best epoch = 4.
- Gain over confidence-weighted DLT: **+2.52 mm** combined direct.
- Gain over unweighted DLT: **+5.85 mm** combined direct.
- A confirmation run with larger batches (batch 64, 8,192 samples/epoch) produced **23.38 mm** (best epoch 7); see `docs/results_iskakov_h36m_true_gt.md`.

### v80 (view-reliability weighting)

Detailed sweep in `docs/results_v80_h36m_true_gt.md`. Local evidence:

| Recipe | Best val MPJPE (mm) | Best epoch | Log / checkpoint |
|---|---:|---:|---|
| v1 (long, no reg) | 65.28 | 2 | `outputs/a800_h36m_reg/omniview_fusion_v80_h36m_true_gt_long.log` |
| v2 | **39.70** | 2 | A800 only (`..._reg_epoch2best.pth`) |
| v3 (reg) | **42.60** | 2 | `outputs/a800_h36m_reg/omniview_fusion_v80_h36m_true_gt_reg.{log,pth}` |
| v4 (reg) | 45.31 | 2 | `outputs/a800_h36m_reg/v4.{log,pth}` |
| medium | **39.98** | 4 | `outputs/omniview_fusion_v80_h36m_true_gt_medium.{log,pth}` |
| smoke | 98.12 | 2 | `outputs/omniview_fusion_v80_h36m_true_gt_smoke.{log,pth}` |

- The medium recipe improves the local best to **39.98 mm** at epoch 4, but still overfits afterward (epoch 8: 133.71 mm).
- Best local result: **39.98 mm** (medium, epoch 4). Best known result: **39.70 mm** (v2, A800 checkpoint).
- v80 still lags Iskakov (~23.35 mm) and even confidence-weighted DLT (~25.87 mm).

### v25 (multiview geometry fusion)

```bash
bash scripts/run_v25_h36m_true_gt_medium_local_4090.sh
```

- Log: `outputs/omniview_fusion_v25_h36m_true_gt_medium.log`
- Checkpoint: `outputs/omniview_fusion_v25_h36m_true_gt_medium.pth`

| Epoch | val MPJPE (mm) |
|---:|---:|
| 1 | 83.19 |
| 2 (best) | **72.80** |
| 3 | 80.14 |
| 4 | 94.27 |
| 5 | 113.48 |
| 6 | 139.21 |
| 7 | 174.90 |
| 8 (final) | 207.62 |

- Training completed 8 epochs before early-stopping patience was exhausted; the run began to diverge after epoch 2.
- Best checkpoint: epoch 2, **combined direct MPJPE = 72.80 mm**.
- S9 direct: **67.92 mm**; S11 direct: **77.68 mm**.
- Gap to baselines on combined direct:
  - **Iskakov**: +49.45 mm (72.80 vs. 23.35 mm).
  - **DLT (unweighted)**: +43.61 mm (72.80 vs. 29.19 mm).
- v25 does not currently beat the geometric or learnable-triangulation baselines on this true-GT protocol.

## How to reproduce

```bash
# Iskakov baseline
python experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m --epochs 10 --batch_size 8 --lr 1e-3 --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --log_path outputs/iskakov_h36m_true_gt.log \
    --ckpt_path outputs/iskakov_h36m_true_gt.pth

# v25 medium
bash scripts/run_v25_h36m_true_gt_medium_local_4090.sh

# v80 medium
bash scripts/run_v80_h36m_true_gt_medium.sh

# v80 smoke
bash scripts/run_v80_h36m_true_gt_smoke_local_4090.sh
```

## Related docs

- `docs/results_iskakov_h36m_true_gt.md` — full Iskakov baseline report including MPJPE@k curves.
- `docs/results_v80_h36m_true_gt.md` — v80 recipe sweep and interpretation.

## Takeaways

1. **True-GT protocol is now reliable**: numbers are in the expected 15–30 mm range, unlike the old circular-label 0.62 mm.
2. **Iskakov is a strong baseline**: it beats both DLT variants and is the current leader on this protocol.
3. **MotionFlow variants need re-tuning**: v80 reaches 39.98 mm at epoch 4 but then overfits (133.71 mm by epoch 8); v25 reaches 72.80 mm at best and then diverges. Neither yet beats the geometric / learnable-triangulation baselines.
4. **The project now has a real leaderboard**: DLT / Iskakov / v25 / v80 on a non-circular H36M standard protocol.
