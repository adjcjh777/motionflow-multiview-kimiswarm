# H36M True-GT v2 Leaderboard Re-run Plan

> **Goal:** Re-train and re-evaluate the key MotionFlow-MultiView variants on the corrected, non-circular H36M true-GT **v2** labels, producing an internally consistent CVPR 2027 leaderboard.
> **Models covered:** v25, v46, v52, v57, v80, v81, v82, v85.
> **Data split:** `configs/splits/h36m_true_gt_v2_standard.yaml` (S1,5,6,7,8 → S9/S11).
> **Last updated:** 2026-08-12

## 1. Why a v2 re-run is needed

The existing `data/h36m_true_gt/` files were generated before the camera-alignment fix in `scripts/convert_h36m_true_gt_v2.py`.  Their stored 3D mocap coordinates are **physically inconsistent** with the stored 2D keypoints and camera parameters (direct MJE ≈ 16.7 m).  The v2 labels (`data/h36m_true_gt_v2/`) re-project the official mocap 3D GT into a consistent 2D/camera frame, giving direct MJE in the tens of millimetres.

Consequences:
- All numbers in `docs/results_true_gt_h36m.md` labelled *historical true-GT v1* are on the misaligned data set and must be replaced.
- The geometric baselines (DLT, RANSAC) have already been re-run on v2 and are stable (see Section 3).
- Every learned model from v25 through v85 must be re-trained **from scratch** on the v2 split using the exact same hyperparameters, then re-evaluated on the v2 S9/S11 test files.

## 2. Hard constraints

- **A800 GPUs:** only GPU 6 and GPU 7 may be used.  `CUDA_VISIBLE_DEVICES` must be `6` or `7`.
- **Do not stop or interfere** with currently running jobs:
  - GPU 7: `v85_random_view_dropout_medium_a800` (training)
  - GPU 6: `v85` split-k no-fallback variable-view evaluation
- `/mnt/nvme0n1p1/zhangzy/projects` and the A800 Docker `motionflow` service are **read-only**.
- A800 disk is ~99 % full; run the safe cleanup dry-run before any large write.
- WSL/local RTX 4090 is for smoke tests only; full medium/long runs stay on A800.

## 3. Pre-requisites

### 3.1 Generate and verify the v2 labels

```bash
# 1. Local WSL: regenerate all v2 .npz files
bash scripts/convert_all_h36m_true_gt_v2.sh

# 2. Audit a few files (direct MJE should be tens of mm, not 0 or thousands)
python scripts/diagnose_circular_labels.py data/h36m_true_gt_v2/s_01_acts_*.npz
python scripts/diagnose_circular_labels.py data/h36m_true_gt_v2/s_09_acts_*.npz

# 3. Sync to A800 (run from WSL)
bash scripts/sync_h36m_true_gt_v2_to_a800.sh
```

- [ ] `data/h36m_true_gt_v2/` contains 7 train `.npz` files (S1,5,6,7,8) and 2 test `.npz` files (S9, S11).
- [ ] `configs/splits/h36m_true_gt_v2_standard.yaml` resolves every path.
- [ ] Direct MJE audit is in the tens of millimetres on the test files.

### 3.2 Re-run v2 geometric baselines

These are fast and provide the reference anchors for the leaderboard.

```bash
bash scripts/run_h36m_true_gt_v2_baselines.sh
```

Expected outputs:
- `outputs/h36m_true_gt_v2_baselines/dlt_baseline_h36m_true_gt_v2.json`
- `outputs/h36m_true_gt_v2_baselines/ransac_baseline_h36m_true_gt_v2.json`

- [ ] Confidence-weighted DLT v2 result recorded.
- [ ] RANSAC/conf-DLT v2 result recorded.

### 3.3 Re-run Iskakov learnable triangulation on v2

The Iskakov script does not use the WebBridge manifest; it uses the `data_3d_h36m.npz` release.  Verify it still points at the same true mocap source and re-run on v2-equivalent data.

```bash
python experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m --epochs 10 --batch_size 8 --lr 1e-3 --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --log_path outputs/iskakov_h36m_true_gt_v2.log \
    --ckpt_path outputs/iskakov_h36m_true_gt_v2.pth
```

- [ ] Iskakov v2 result recorded.

### 3.4 Free disk and queue resources

- [ ] Run `scripts/cleanup_a800_safe.sh` dry-run and inspect output.
- [ ] Confirm GPU 6 and GPU 7 are free (do not disturb the running v85 jobs).
- [ ] Decide serial vs. queue order: v85 finishes first, then schedule the other models.

## 4. Common adaptation for every learned model

Every training script currently points at the v1 manifest:

```bash
--mixed_manifest configs/splits/h36m_true_gt_standard.yaml
```

For the v2 re-run, change it to:

```bash
--mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml
```

And update output paths so v1 and v2 results do not collide, e.g.:

```bash
--output outputs/ablations/v25_true_gt_v2_stability_a800.pth
> outputs/ablations/v25_true_gt_v2_stability_a800.log
```

Recommended convention: append `_v2` to the existing output name.

Evaluation scripts also need the v2 paths.  Each `eval_v*.py` defaults to `data/h36m_true_gt/`; pass:

```bash
--s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
--s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz
```

## 5. Per-model re-run checklist

### 5.1 v25 stability (baseline)

**Source script:** `scripts/run_v25_ablation_true_gt_stability_a800.sh`

**Changes for v2:**
- `--mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml`
- `--output outputs/ablations/v25_true_gt_v2_stability_a800.pth`
- Log: `outputs/ablations/v25_true_gt_v2_stability_a800.log`
- GPU: use `CUDA_VISIBLE_DEVICES=6` or `7` when free.

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v25_ablation_true_gt_stability_a800.sh
```

**Test eval:**
```bash
python scripts/eval_v25_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v25_true_gt_v2_stability_a800.pth \
    --config_json outputs/ablations/v25_true_gt_v2_stability_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 1 \
    --out_json outputs/eval_v25_true_gt_v2_stability_h36m_test.json
```

**Sparse-view eval:**
```bash
bash scripts/run_eval_variable_views_v25_true_gt_stability_a800.sh  # adapt to v2 paths
# OR use the variable-view wrapper with --var_view_dlt_fallback on v2 S9/S11
```

- [ ] v25 v2 training finished.
- [ ] v25 v2 S9/S11 test MPJPE/PA-MPJPE recorded.
- [ ] v25 v2 variable-view k=2/3/4 MPJPE recorded (with and without DLT fallback).

### 5.2 v46 sparse-view generalization (SVG)

**Source script:** `scripts/run_v46_true_gt_h36m_a800.sh`

**Changes for v2:**
- `--mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml`
- `--output outputs/ablations/v46_true_gt_v2_h36m_a800.pth`
- Log: `outputs/ablations/v46_true_gt_v2_h36m_a800.log`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v46_true_gt_h36m_a800.sh
```

**Test eval:**
```bash
python scripts/eval_v46_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v46_true_gt_v2_h36m_a800.pth \
    --config_json outputs/ablations/v46_true_gt_v2_h36m_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v46_true_gt_v2_h36m_test_a800.json
```

- [ ] v46 v2 training finished.
- [ ] v46 v2 S9/S11 test MPJPE/PA-MPJPE recorded.

### 5.3 v52 uncertainty-weighted triangulation (UWT)

**Source script:** `scripts/run_v57_h36m_true_gt_medium.sh` (v52 is the full stack in that script minus DC-PSC; v52 also has `scripts/run_eval_v52_true_gt_h36m_test_a800.sh` for evaluation only)

**Training changes for v2:**
- `--mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml`
- `--output outputs/ablations/v52_true_gt_v2_h36m_a800.pth`
- Log: `outputs/ablations/v52_true_gt_v2_h36m_a800.log`

**Test eval:**
```bash
bash scripts/run_eval_v52_true_gt_h36m_test_a800.sh  # after editing paths/config inside to v2
```

Or directly:
```bash
python scripts/eval_v52_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v52_true_gt_v2_h36m_a800.pth \
    --config_json outputs/ablations/v52_true_gt_v2_h36m_a800.config.json \
    --val_stride 13 \
    --out_json outputs/eval_v52_true_gt_v2_h36m_test_a800.json
```

- [ ] v52 v2 training finished.
- [ ] v52 v2 S9/S11 test MPJPE/PA-MPJPE recorded.

### 5.4 v57 domain-conditional physical-space calibration (DC-PSC)

**Source script:** `scripts/run_v57_true_gt_medium_a800.sh`

**Changes for v2:**
- `--mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml`
- `--output outputs/ablations/v57_true_gt_v2_medium_a800.pth`
- Log: `outputs/ablations/v57_true_gt_v2_medium_a800.log`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v57_true_gt_medium_a800.sh
```

**Test eval:**
```bash
python scripts/eval_v57_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v57_true_gt_v2_medium_a800.pth \
    --config_json outputs/ablations/v57_true_gt_v2_medium_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v57_true_gt_v2_h36m_test_a800.json
```

- [ ] v57 v2 training finished.
- [ ] v57 v2 S9/S11 test MPJPE/PA-MPJPE recorded.

### 5.5 v80 view-reliability weighting

**Source script:** `scripts/run_v80_ablation_true_gt_regularization_a800.sh`

**Changes for v2:**
- `--mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml`
- `--output outputs/ablations/v80_true_gt_v2_regularization_a800.pth`
- Log: `outputs/ablations/v80_true_gt_v2_regularization_a800.log`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v80_ablation_true_gt_regularization_a800.sh
```

**Test eval:**
```bash
python scripts/eval_v80_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v80_true_gt_v2_regularization_a800.pth \
    --config_json outputs/ablations/v80_true_gt_v2_regularization_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v80_true_gt_v2_h36m_test_a800.json
```

- [ ] v80 v2 training finished.
- [ ] v80 v2 S9/S11 test MPJPE/PA-MPJPE recorded.

### 5.6 v81 temporal-pose-attention

**Source script:** `scripts/run_v81_true_gt_h36m_medium_a800.sh`

**Changes for v2:**
- `--mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml`
- `--output outputs/ablations/v81_true_gt_v2_h36m_medium_a800.pth`
- Log: `outputs/ablations/v81_true_gt_v2_h36m_medium_a800.log`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v81_true_gt_h36m_medium_a800.sh
```

**Test eval:**
```bash
python scripts/eval_v81_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v81_true_gt_v2_h36m_medium_a800.pth \
    --config_json outputs/ablations/v81_true_gt_v2_h36m_medium_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v81_true_gt_v2_h36m_test_a800.json
```

- [ ] v81 v2 training finished.
- [ ] v81 v2 S9/S11 test MPJPE/PA-MPJPE recorded.

### 5.7 v82 multi-scale temporal-pose-attention

v82 does not have a dedicated A800 training script in the current tree.  Create it by copying `scripts/run_v81_true_gt_h36m_medium_a800.sh` and adding the v82 flags (`--use_multiscale_temporal_pose_attention_v82` and its parameters).

**Source script:** create `scripts/run_v82_true_gt_h36m_medium_a800.sh` from v81 script.

**Changes for v2:**
- `--mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml`
- `--output outputs/ablations/v82_true_gt_v2_h36m_medium_a800.pth`
- Log: `outputs/ablations/v82_true_gt_v2_h36m_medium_a800.log`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v82_true_gt_h36m_medium_a800.sh
```

**Test eval:**
```bash
bash scripts/run_eval_v82_true_gt_h36m_test_a800.sh  # after editing paths to v2
```

- [ ] v82 v2 training script created.
- [ ] v82 v2 training finished.
- [ ] v82 v2 S9/S11 test MPJPE/PA-MPJPE recorded.

### 5.8 v85 random view dropout (sparse-view robustness)

**Source script:** `scripts/run_v85_random_view_dropout_medium_a800.sh`

v85 is already running on GPU 7.  When it finishes, the same script must be re-launched on the v2 split.

**Changes for v2:**
- `--mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml`
- `--output outputs/ablations/v85_random_view_dropout_v2_medium_a800.pth`
- Log: `outputs/ablations/v85_random_view_dropout_v2_medium_a800.log`

**Train:**
```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v85_random_view_dropout_medium_a800.sh
```

**Test eval:**
```bash
# Full 4-view S9/S11 test
python scripts/eval_v85_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v85_random_view_dropout_v2_medium_a800.pth \
    --config_json outputs/ablations/v85_random_view_dropout_v2_medium_a800.config.json \
    --s9  data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --val_stride 13 \
    --out_json outputs/eval_v85_true_gt_v2_h36m_test_a800.json

# Sparse-view no-fallback eval
bash scripts/run_eval_variable_views_v85_true_gt_stability_a800.sh  # adapt to v2

# Sparse-view DLT-fallback eval
bash scripts/eval_variable_views_v85_dlt_fallback_a800.sh  # adapt to v2
```

- [ ] v85 v2 training launched **after** the current v85 v1 job finishes.
- [ ] v85 v2 full-view test MPJPE/PA-MPJPE recorded.
- [ ] v85 v2 no-fallback variable-view k=2/3/4 MPJPE recorded.
- [ ] v85 v2 DLT-fallback variable-view k=2/3/4 MPJPE recorded.

## 6. Result aggregation

After each model finishes, update `docs/results_true_gt_h36m.md`:

- Replace the *historical true-GT v1* table with the v2 table.
- Keep v1 numbers in a clearly marked “Historical v1 (misaligned) results” section for reference.
- Add a new section “True-GT v2 Leaderboard” with the v2 numbers.

Suggested table columns:

| Method | S9 direct (mm) | S11 direct (mm) | Combined direct (mm) | Combined PA-MPJPE (mm) | Source |
|---|---:|---:|---:|---:|---|
| Iskakov ICCV 2019 | — | — | — | — | — |
| DLT (conf-weighted) | — | — | — | — | `outputs/h36m_true_gt_v2_baselines/dlt_baseline_h36m_true_gt_v2.json` |
| RANSAC/conf-DLT | — | — | — | — | `outputs/h36m_true_gt_v2_baselines/ransac_baseline_h36m_true_gt_v2.json` |
| v25 stability | — | — | — | — | `outputs/eval_v25_true_gt_v2_stability_h36m_test.json` |
| v46 SVG | — | — | — | — | `outputs/eval_v46_true_gt_v2_h36m_test_a800.json` |
| v52 UWT | — | — | — | — | `outputs/eval_v52_true_gt_v2_h36m_test_a800.json` |
| v57 DC-PSC | — | — | — | — | `outputs/eval_v57_true_gt_v2_h36m_test_a800.json` |
| v80 regularization | — | — | — | — | `outputs/eval_v80_true_gt_v2_h36m_test_a800.json` |
| v81 temporal-pose-attention | — | — | — | — | `outputs/eval_v81_true_gt_v2_h36m_test_a800.json` |
| v82 multi-scale temporal-pose-attention | — | — | — | — | `outputs/eval_v82_true_gt_v2_h36m_test_a800.json` |
| v85 random view dropout | — | — | — | — | `outputs/eval_v85_true_gt_v2_h36m_test_a800.json` |

- [ ] `docs/results_true_gt_h36m.md` v2 table populated.
- [ ] Sparse-view v2 tables added for v25/v81/v82/v85.

## 7. Risks and watch-outs

| Risk | Mitigation |
|---|---|
| v85 currently occupies GPU 7 and its eval occupies GPU 6. | Queue the v2 re-runs; do not stop the current jobs. |
| A800 disk is ~99 % full. | Run `scripts/cleanup_a800_safe.sh` dry-run before each new run; delete failed/abandoned runs first. |
| v82 has no A800 training script. | Create it from the v81 script before training. |
| Eval scripts default to v1 `.npz` paths. | Always pass the explicit `--s9` / `--s11` v2 paths or edit the script defaults. |
| Training scripts default to v1 manifest. | Always swap to `configs/splits/h36m_true_gt_v2_standard.yaml`. |
| Re-running 8 medium A800 jobs is a long queue. | Prioritise v25, v85, v81, v82 (best current models); v46/v52/v57/v80 can follow. |

## 8. Quick command summary

```bash
# 1. Generate v2 labels (local)
bash scripts/convert_all_h36m_true_gt_v2.sh
bash scripts/sync_h36m_true_gt_v2_to_a800.sh

# 2. Baselines
bash scripts/run_h36m_true_gt_v2_baselines.sh

# 3. Training (queue on free GPU 6/7)
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v25_ablation_true_gt_stability_a800.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v46_true_gt_h36m_a800.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v57_true_gt_medium_a800.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v80_ablation_true_gt_regularization_a800.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v81_true_gt_h36m_medium_a800.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v82_true_gt_h36m_medium_a800.sh  # create first
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v85_random_view_dropout_medium_a800.sh

# 4. Test eval (replace checkpoint/config paths with v2 variants)
python scripts/eval_v25_true_gt_h36m_test.py --checkpoint ... --s9 ... --s11 ... --out_json ...
python scripts/eval_v46_true_gt_h36m_test.py --checkpoint ... --s9 ... --s11 ... --out_json ...
python scripts/eval_v57_true_gt_h36m_test.py --checkpoint ... --s9 ... --s11 ... --out_json ...
python scripts/eval_v80_true_gt_h36m_test.py --checkpoint ... --s9 ... --s11 ... --out_json ...
python scripts/eval_v81_true_gt_h36m_test.py --checkpoint ... --s9 ... --s11 ... --out_json ...
python scripts/eval_v82_true_gt_h36m_test.py --checkpoint ... --s9 ... --s11 ... --out_json ...
python scripts/eval_v85_true_gt_h36m_test.py --checkpoint ... --s9 ... --s11 ... --out_json ...

# 5. Update docs/results_true_gt_h36m.md with v2 numbers
```

## 9. Definition of done

- [ ] All v2 `.npz` files generated, synced, and audited.
- [ ] DLT/RANSAC/Iskakov v2 baselines re-run and recorded.
- [ ] v25, v46, v52, v57, v80, v81, v82, v85 each trained once on `configs/splits/h36m_true_gt_v2_standard.yaml`.
- [ ] Each trained model evaluated on v2 S9/S11 test data.
- [ ] Sparse-view / DLT-fallback evals completed for v25, v81, v82, v85 on v2.
- [ ] `docs/results_true_gt_h36m.md` updated with a v2 leaderboard table.
- [ ] No running A800 job was interrupted or modified.
