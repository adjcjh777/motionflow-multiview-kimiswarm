# H36M True-GT Standard Training Protocol

> **TL;DR:** Train on subjects **S1, S5, S6, S7, S8** (all actions 02–16), evaluate on **S9** and **S11**. Use only the metre-convention true-GT files in `data/h36m_true_gt/`. The old `data/h36m_hf/*.npz` and `data/webbridge/h36m*.npz` labels are circular and must not be used for model selection.

## 1. The exact standard protocol

### Subject split

| Role | Subjects | Actions | Source `.npz` |
|---|---|---|---|
| Train | S1, S5, S6, S7, S8 | 02–16 (all 15 actions) | `data/h36m_true_gt/s_{01,05,06,07,08}_acts_02_03_..._16_multiview_m.npz` |
| Test (val) | S9, S11 | 02–16 (all 15 actions) | `data/h36m_true_gt/s_{09,11}_acts_02_03_..._16_multiview_m.npz` |

* 4 calibrated studio views per subject/action.
* 17 joints (standard H36M-17 subset: `[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]`).
* All 3D coordinates and camera translations are in **metres** (`*_m.npz`); `camera_K` is in pixels.
* Canonical manifest: `configs/splits/h36m_true_gt_standard.yaml`.

### Manifest

```yaml
name: H36M True GT Standard (S1,5,6,7,8 -> S9/S11)
train_paths:
  - data/h36m_true_gt/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
  - data/h36m_true_gt/s_05_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
  - data/h36m_true_gt/s_06_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
  - data/h36m_true_gt/s_07_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
  - data/h36m_true_gt/s_08_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
val_paths:
  - data/h36m_true_gt/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
  - data/h36m_true_gt/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
```

### Why this split

This is the canonical Human3.6M cross-subject protocol used in the multi-view pose literature. It keeps training and test actors disjoint, so numbers measure **generalisation to unseen performers**, not just unseen actions or camera framings.

## 2. Why the old S1-only protocol was invalid

The project previously reported numbers from a **train-S1 → validate-S5/Act2** (or similar S1-only) setup. That setup is invalid for two independent reasons:

### 2.1 Too little training data

* S1 contains **~62k frames**, while S1+S5+S6+S7+S8 contain **~390k frames**.
* A model selected on a S1-only split is tuned on a small corpus and is not comparable to literature baselines trained on the full five-subject training set.

### 2.2 Labels were circular

The previous `.npz` files in `data/h36m_hf/` and `data/webbridge/h36m*.npz` stored 3D labels that were the **unweighted DLT triangulation of the input 2D keypoints**:

```text
joints_3d == triangulate_dlt(points_2d, cameras)
```

This is visible in `motionflow_mv/data/webbridge_loader.py:182` (the fallback branch triangulates `points_2d` when no true GT is supplied) and confirmed by the diagnostic:

```bash
python scripts/diagnose_circular_labels.py data/h36m_hf/s_01_act_02_multiview.npz
```

Output on the old files:

```text
frames=2995, views=4, joints=17
  direct MJE (no root align): 0.0000 mm
  root-aligned MPJPE:       0.0000 mm
```

When the label is a deterministic function of the 2D input, the model is scored on how well it reproduces the DLT layer, not on real 3D pose accuracy. This made the S1-only leaderboard numbers **scientifically meaningless**.

### 2.3 Current true-GT acceptance gates

The repaired `data/h36m_true_gt/*.npz` files were accepted only after passing both gates:

| Gate | Old circular files | True-GT files | Tool |
|---|---|---|---|
| Circularity (direct MJE) | ~0 mm | **27,762–37,320 mm** ≫ 0 | `scripts/diagnose_circular_labels.py` |
| Reprojection RMSE | ~0 px | **3.13–7.15 px** ≤ 15 px | `scripts/check_true_gt_reprojection.py` |

These numbers prove the labels are independent mocap world coordinates, not a function of the 2D inputs.

## 3. How to reproduce the true-GT data

### 3.1 Obtain the source files

You need the official Human3.6M release with true mocap world coordinates. The pipeline accepts either:

1. Per-subject `.cdf`/`.mat` mocap files (`PosesD3_Positions`).
2. The VideoPose3D-format `data_3d_h36m.npz` (Google Drive id `1mAHq0YhO75frDkgUgebFQYnnPQOjUcr4`, ~174 MB).

### 3.2 Generate the canonical true-GT `.npz`

Using the VideoPose3D source (recommended):

```bash
python experiments/prepare_h36m_true_gt.py \
    --data3d-npz data/h36m_true_gt/data_3d_h36m.npz \
    --subject 1 --actions 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --split train \
    --out data/h36m_true_gt/s_01_acts_02_03_..._16_true_gt.npz
```

Repeat for subjects 5, 6, 7, 8 (train) and 9, 11 (test).

Using raw `.mat`/`.cdf` per action:

```bash
python experiments/prepare_h36m_true_gt.py \
    --mocap_dir data/h36m_true_gt --subject 9 --actions 2 14 --split test \
    --out data/h36m_true_gt/s_09_acts_02_14_true_gt.npz
```

### 3.3 Convert to metre convention

Training manifests use metres, so convert the millimetre canonical files:

```bash
python experiments/make_h36m_true_gt_metres.py \
    --glob "data/h36m_true_gt/s_*_multiview.npz"
```

This writes `<stem>_m.npz` files with `joints_3d` and `camera_t` in metres.

### 3.4 Validate the generated files

Run the non-circularity and reprojection audits:

```bash
# Non-circularity: direct MJE must be ≫ 0 mm
python scripts/diagnose_circular_labels.py "data/h36m_true_gt/s_*_multiview_m.npz"

# Reprojection: RMSE must be ≤ 15 px
python scripts/check_true_gt_reprojection.py "data/h36m_true_gt/s_*_multiview_m.npz"
```

Expected results (true-GT metres):

```text
# diagnose_circular_labels.py
S1 direct MJE  15.94 mm
S5 direct MJE ≈ 16.12 mm
...
S9 direct MJE ≈ 33.83 mm
S11 direct MJE ≈ 24.75 mm

# check_true_gt_reprojection.py
worst RMSE ≈ 7.15 px within threshold 15 px
```

### 3.5 Unit tests

The full pipeline is covered by:

```bash
python -m pytest tests/test_h36m_true_gt_pipeline.py -q
```

## 4. How to reproduce training runs

All commands use `configs/splits/h36m_true_gt_standard.yaml` as the data manifest and output checkpoints/logs under `outputs/`.

### DLT / geometric baselines

```bash
# DLT baseline (unweighted)
python experiments/baselines.py \
    --datasets "data/h36m_true_gt/s_*_multiview_m.npz" \
    --max_frames 2000

# Iskakov learnable triangulation
python experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m --epochs 10 --batch_size 8 --lr 1e-3 --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --log_path outputs/iskakov_h36m_true_gt.log \
    --ckpt_path outputs/iskakov_h36m_true_gt.pth
```

### v25 / v57 / v80 learned models

```bash
# v25
bash scripts/run_v25_h36m_true_gt_medium_local_4090.sh

# v57
bash scripts/run_v57_h36m_true_gt_medium.sh

# v80
bash scripts/run_v80_h36m_true_gt_medium.sh
```

Each script points to the same true-GT manifest and records per-epoch combined direct MPJPE on S9+S11. Results are collected in `docs/results_true_gt_h36m.md`.

## 5. What numbers to expect

Reported on the true-GT standard protocol (`docs/results_true_gt_h36m.md`):

| Method | Combined direct MPJPE | Notes |
|---|---|---|
| DLT (unweighted) | 29.19 mm | Geometric baseline |
| DLT (confidence-weighted) | 25.87 mm | Geometric baseline |
| Iskakov ICCV 2019 | **23.35 mm** | Current leader |
| v80 | 39.98 mm | Best epoch 4, then overfits |
| v25 | 72.80 mm | Best epoch 2, then diverges |
| v57 | 75.16 mm (obs.) / 81.47 mm (ckpt) | Best epoch 3, early stopped at epoch 5; saved ckpt is epoch 2 |

These are **hundreds of times larger** than the old circular-label 0.62 mm results, which is the expected consequence of evaluating on real mocap coordinates rather than DLT reproduction.

## 6. Checklist before claiming an H36M number

- [ ] Training uses only `data/h36m_true_gt/*_multiview_m.npz` (metre convention).
- [ ] Manifest is `configs/splits/h36m_true_gt_standard.yaml` (S1,S5,S6,S7,S8 → S9,S11).
- [ ] Validation subjects are disjoint from training subjects.
- [ ] `scripts/diagnose_circular_labels.py` reports direct MJE ≫ 0 mm on all used `.npz` files.
- [ ] `scripts/check_true_gt_reprojection.py` reports RMSE ≤ 15 px on all used `.npz` files.
- [ ] Reported metric is **combined direct MPJPE** on S9+S11, optionally with per-subject splits.

## 7. Related files

| File | Purpose |
|---|---|
| `configs/splits/h36m_true_gt_standard.yaml` | Training/val manifest |
| `experiments/prepare_h36m_true_gt.py` | Build true-GT `.npz` from mocap |
| `experiments/make_h36m_true_gt_metres.py` | Convert mm → metres |
| `scripts/diagnose_circular_labels.py` | Non-circularity audit |
| `scripts/check_true_gt_reprojection.py` | Reprojection audit |
| `tests/test_h36m_true_gt_pipeline.py` | Unit tests for the pipeline |
| `docs/results_true_gt_h36m.md` | Current leaderboard |

## 8. Common mistakes to avoid

1. **Using `data/h36m_hf/` or `data/webbridge/h36m*.npz`** — these contain circular DLT labels (`direct MJE = 0 mm`).
2. **Training on S1 only** — not comparable to standard literature or to the current true-GT leaderboard.
3. **Using the wrong joint subset** — the pkl order differs from the first 17 mocap joints; the correct 17-joint list is `VP3D_JOINT_INDICES` in `experiments/prepare_h36m_true_gt.py`.
4. **Forgetting the metre conversion** — the true-GT `.npz` are in mm by default; training manifests use the `_m.npz` metre versions.
