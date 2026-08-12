# MotionFlow Multi-View

> CVPR 2027 direction: **reliable multi-view human pose under the true-GT H36M protocol**, with emphasis on sparse-view and cross-domain robustness.

## Quick orientation

This repository trains and evaluates multi-view 3D human pose estimators.
The previous leaderboards were contaminated by **circular H36M labels** (the stored 3D target was a DLT triangulation of the input 2D).
We are rebuilding the evaluation foundation on non-circular, true mocap-ground-truth data.

- **Standard protocol:** H36M train on S1, S5, S6, S7, S8; test on S9 and S11.
- **True-GT manifest:** `configs/splits/h36m_true_gt_standard.yaml`
- **Trusted labels:** `data/h36m_true_gt/` (H36M), `data/webbridge/shelf_campus_detected/` (Shelf/Campus)
- **Deprecated (circular):** any config or script still pointing to `data/h36m_hf/` or `data/webbridge/h36m*.npz` will fail loudly.

## Current H36M true-GT leaderboard (S9 / S11)

| Method | Combined direct (mm) | Combined PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| **Iskakov ICCV 2019** | **23.40** | **23.15** | learnable triangulation baseline |
| DLT (confidence-weighted) | 25.67 | 28.05 | frozen geometric baseline |
| MVPose (geometry-only) | 26.06 | 28.32 | COCO17 native skeleton |
| RANSAC/conf-DLT | 26.47 | 28.98 | reproducible 3-view random subset |
| **v25 stability** | **30.83** | **33.59** | best learned result so far |
| v25 mixed H36M+AIST++ | 33.42 | 34.60 | early-stopped Epoch 1 |
| v81 temporal-pose-attention | 37.83 | 37.75 | — |
| v82 multi-scale temporal-pose-attention | 39.46 | 39.94 | — |
| v80 regularization | 53.98 | 32.47 | — |
| v52 UWT | 54.01 | 42.22 | — |
| v57 | 57.10 | 37.30 | re-run with MPJPE checkpoint monitor |

- Full table: `docs/results_true_gt_h36m.md`
- Source JSONs: `outputs/eval_*_true_gt_h36m_test*.json`

## Sparse-view (variable-view) robustness

Learned models trained on the full 4-view rig currently fail catastrophically when fewer than 4 views are active.
Direct confidence-weighted DLT on the same active views gives reasonable numbers, confirming the 2D observations themselves are sound.

| Subject | k=2 (mm) | k=3 (mm) | k=4 (mm) |
|---|---:|---:|---:|
| S9 | 58.18 | 33.32 | 116.98 |
| S11 | 49.35 | 25.28 | 110.58 |

- Source: `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json`
- v85 random-view dropout is in progress to train a model that natively handles k=2/3/4.

## Cross-dataset status

| Dataset | Status | Key number |
|---|---|---|
| AIST++ | Non-circular `.npz` ready; full DLT baseline **15.93 mm** / PA-MPJPE **21.12 mm** | AIST++-only → H36M cross-eval **93.94 mm** |
| MPI-INF-3DHP | RTMPose detected-2D regenerated (16/16 `.npz`); DLT baseline **115.09 mm** / PA-MPJPE **132.68 mm** | `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json` |
| Shelf/Campus detected | Non-circular `.npz` ready | Iskakov **128.73 mm** val direct |

## Repository structure

```
motionflow-multiview-kimiswarm/
├── configs/          # experiment configs and dataset splits
├── docs/             # status, results, paper draft
├── experiments/      # training / evaluation scripts
├── motionflow_mv/    # core package (data, fusion, models, eval, losses, training)
├── outputs/          # checkpoints, logs, evaluation JSONs
├── scripts/          # shell runners and utilities
├── tests/            # unit tests
└── requirements.txt
```

## Quick start (WSL + RTX 4090)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -v
```

Run a true-GT baseline:

```bash
# Confidence-weighted DLT baseline
python scripts/run_h36m_true_gt_dlt_baseline.py

# Iskakov learnable-triangulation baseline
python experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m --epochs 10 --batch_size 8 --lr 1e-3 --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --log_path outputs/iskakov_h36m_true_gt.log \
    --ckpt_path outputs/iskakov_h36m_true_gt.pth
```

## Key documents

- `docs/cvpr2027_status.md` — latest overall status, blockers, and next steps.
- `docs/results_true_gt_h36m.md` — full H36M true-GT leaderboard and per-method details.
- `docs/results_true_gt_shelf_campus.md` — Shelf/Campus detected leaderboard.
- `docs/cvpr2027_pivot_for_new_collaborators.md` — onboarding for new collaborators.
- `AGENTS.md` — active run status, GPU policy, and handoff notes.

## Hardware and GPU rules

- **Local WSL:** NVIDIA RTX 4090 (one training task at a time).
- **A800:** MotionFlow-MultiView uses **only GPU 6 and GPU 7**. Never use GPUs 0–5.
- `/mnt/nvme0n1p1/zhangzy/projects` and the A800 Docker `motionflow` service are **read-only**.
- Do not stop, kill, or interfere with running A800 jobs.

## License

To be determined based on the baseline MotionFlow license.
