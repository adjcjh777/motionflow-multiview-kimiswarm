# AIST++ Integration Plan — CVPR 2027

**Status:** Data ready, baselines complete, cross-eval scripts fixed and syntax-verified.  
**Last updated:** 2026-08-12

## 1. Goal

Make AIST++ a first-class, reproducible cross-domain benchmark for MotionFlow-MultiView:

1. Canonical AIST++ `.npz` are generated, audited, and stored in the project path.
2. Training, test-set evaluation, and cross-evaluation on H36M true-GT are scripted and documented.
3. Results are recorded in a form that can go directly into the CVPR 2027 paper.

## 2. Current State

### 2.1 Data

| Asset | Path | Status |
|-------|------|--------|
| Canonical AIST++ multi-view `.npz` | `data/webbridge/aistpp_canonical/` | **Ready** — 1,408 clips, 9 views, 17 joints (H36M skeleton), metres. |
| Train/val split | `configs/splits/webbridge_aistpp_train_val.yaml` | **Ready** — 1,280 train / 128 val. |
| Train/val/test split | `configs/splits/aistpp_train_val_test.yaml` | **Ready** — 1,280 train / 64 val / 64 test. |
| AIST++-only train/val (A800 medium) | `configs/splits/aistpp_only_train_val_a800.yaml` | **Ready** — used for the AIST++-only medium run. |

### 2.2 Completed Runs

| Run | Script | Key Results | Artifacts |
|-----|--------|-------------|-----------|
| AIST++-only medium fast v2 | `scripts/run_aistpp_only_medium_a800.sh` | Best val **91.43 mm** @ Epoch 4 | `outputs/ablations/aistpp_only_medium_a800_fast_v2_final.pth` |
| AIST++ → H36M cross-eval | `scripts/eval_aistpp_only_on_h36m_test_a800.sh` | S9 **98.17 mm**, S11 **89.70 mm** | `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json` |
| Full AIST++ DLT baseline | `experiments/run_aistpp_full_dlt_baseline.py` | Weighted MPJPE **15.93 mm**, PA-MPJPE **21.12 mm** | `outputs/aistpp_full_dlt_baseline_a800.json` |
| Iskakov ICCV 2019 baseline (local CPU) | `experiments/train_iskakov_aistpp_full.py` | Val direct MPJPE **29.27 mm** | `outputs/iskakov_learnable_tri_aistpp_full.pth` |

### 2.3 Known Issues Fixed in This Pass

- **Missing checkpoint symlink.**  The cross-eval scripts expected `outputs/ablations/aistpp_only_medium_a800_fast_v2.pth`, but only the `_final.pth` checkpoint existed (the symlink had been removed by cleanup).  The scripts now fall back to `_final.pth` automatically.
- **GPU policy violations.**  `scripts/run_aistpp_only_medium_a800.sh` and `scripts/run_v25_aistpp_full_medium_a800.sh` defaulted to GPU 4; they now default to GPU 7 and respect `CUDA_VISIBLE_DEVICES`.

## 3. File Inventory

### 3.1 Data / Splits

- `data/webbridge/aistpp_canonical/*.npz`
- `configs/splits/aistpp_train_val_test.yaml`
- `configs/splits/aistpp_only_train_val_a800.yaml`
- `configs/splits/webbridge_aistpp_train_val.yaml`
- `configs/splits/aistpp_train_val_test_mixed.yaml`

### 3.2 Training Scripts

| Script | Purpose | Default GPU |
|--------|---------|-------------|
| `scripts/run_aistpp_only_medium_a800.sh` | AIST++-only medium training | 7 (was 4) |
| `scripts/run_v25_aistpp_full_medium_a800.sh` | v25 full-medium on AIST++ | 7 (was 4) |
| `scripts/run_v57_aistpp_full_medium_a800.sh` | v57 on AIST++ | 6 |
| `scripts/run_v80_aistpp_full_medium_a800.sh` | v80 on AIST++ | 6 |
| `scripts/run_iskakov_aistpp_full_a800_gpu6.sh` | Iskakov baseline on AIST++ | 6 |
| `experiments/train_iskakov_aistpp_full.py` | Iskakov trainer (CPU/GPU) | — |

### 3.3 Evaluation / Cross-Evaluation Scripts

| Script | Purpose | Fixed in This Pass |
|--------|---------|--------------------|
| `scripts/eval_aistpp_only_on_h36m_test.sh` | Evaluate AIST++-only checkpoint on H36M true-GT S9/S11 | Yes — `_final.pth` fallback |
| `scripts/eval_aistpp_only_on_h36m_test_a800.sh` | Same as above, A800 variant | Yes — `_final.pth` fallback |
| `scripts/eval_aistpp_test_set.sh` | Evaluate a trained checkpoint on the AIST++ test split | Yes — `_final.pth` fallback |
| `experiments/eval_omniview_fusion_v5_aistpp.py` | Test-set evaluation driver | No change needed |
| `experiments/run_aistpp_full_dlt_baseline.py` | DLT baseline on AIST++ | No change needed |

### 3.4 Result Docs

- `docs/results_aistpp_dlt_baseline.md`
- `docs/results_aistpp_iskakov_full.md`
- `docs/aistpp_smoke_diagnosis.md`
- `docs/results_true_gt_h36m.md` (cross-eval numbers)

## 4. Verification Performed

1. **Bash syntax checks** passed for all modified scripts:
   - `scripts/eval_aistpp_only_on_h36m_test.sh`
   - `scripts/eval_aistpp_only_on_h36m_test_a800.sh`
   - `scripts/eval_aistpp_test_set.sh`
   - `scripts/run_aistpp_only_medium_a800.sh`
   - `scripts/run_v25_aistpp_full_medium_a800.sh`
2. **Python syntax checks** passed:
   - `scripts/eval_v25_true_gt_h36m_test.py`
   - `experiments/eval_omniview_fusion_v5_aistpp.py`
   - `experiments/train_iskakov_aistpp_full.py`
3. **Checkpoint fallback logic** tested in a temporary directory: when `.pth` is absent and `_final.pth` is present, the script resolves to `_final.pth`.
4. **File existence on A800** confirmed:
   - `outputs/ablations/aistpp_only_medium_a800_fast_v2_final.pth` exists.
   - `outputs/ablations/aistpp_only_medium_a800_fast_v2.config.json` exists.
   - `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json` exists.

> **Note:** A full end-to-end run of the cross-eval scripts was not repeated because the completed AIST++-only checkpoint is already evaluated and the A800 GPUs are occupied by v85 training/eval.  No running jobs were disturbed.

## 5. Open Tasks to Reach Paper Quality

| # | Task | Owner / Where | Priority |
|---|------|---------------|----------|
| 1 | Run v25/v57/v80 AIST++ full-medium to completion on A800 GPU 6/7 | `scripts/run_v25_aistpp_full_medium_a800.sh`, `scripts/run_v57_aistpp_full_medium_a800.sh`, `scripts/run_v80_aistpp_full_medium_a800.sh` | P0 |
| 2 | Run Iskakov AIST++ full on A800 GPU 6 | `scripts/run_iskakov_aistpp_full_a800_gpu6.sh` | P0 |
| 3 | Evaluate all finished checkpoints on the AIST++ test split | `scripts/eval_aistpp_test_set.sh` | P0 |
| 4 | Re-run AIST++ → H36M cross-eval for every finished model | `scripts/eval_aistpp_only_on_h36m_test_a800.sh` | P1 |
| 5 | Generate a consolidated results table for the paper | `docs/paper_results_table.md` or `docs/paper_draft_icra_cvpr_2027.md` | P1 |
| 6 | Add a regression test that asserts the fallback logic in the cross-eval scripts | `tests/` (new) | P2 |

## 6. Blockers and Risks

- **A800 disk is 99% full (~58 GB free).**  Training new models or keeping many checkpoints may fail.  Run `scripts/cleanup_a800_safe.sh` before launching new runs.
- **Only GPUs 6/7 are available.**  All new AIST++ training/eval jobs must use these GPUs.
- **v85 training/eval is still running on GPUs 6/7.**  Do not kill or interfere with those jobs.  Schedule new AIST++ runs only when a GPU is free.
- **AIST++-only checkpoint symlink was removed by cleanup.**  The scripts now fall back to `_final.pth`, but any hard-coded references elsewhere in docs or scripts should be checked before the next cleanup.

## 7. Recommended Next Smoke / Sanity Check

After the next AIST++ model finishes training, run:

```bash
# AIST++ test-set eval
bash scripts/eval_aistpp_test_set.sh v25

# H36M cross-eval
bash scripts/eval_aistpp_only_on_h36m_test_a800.sh
```

If both complete without `FileNotFoundError` and produce JSON outputs, the integration is healthy.
