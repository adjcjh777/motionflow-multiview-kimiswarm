# v57 H36M True-GT Medium Checkpoint Validation

> **Status: Rerun ready** — the underlying trainer bug (best checkpoint selected by loss instead of MPJPE) has been fixed; see `docs/v57_checkpoint_bug.md`.  
> **Run:** v57 H36M true-GT medium (`scripts/run_v57_h36m_true_gt_medium.sh`)  
> **Checkpoint:** `outputs/omniview_fusion_v57_h36m_true_gt_medium.pth`  
> **Config:** `outputs/omniview_fusion_v57_h36m_true_gt_medium.config.json`  
> **Log:** `outputs/omniview_fusion_v57_h36m_true_gt_medium.log`  
> **Validated:** 2026-08-11 06:05 UTC  
> **Validator:** coder subagent

## Summary

The v57 checkpoint is **loadable, structurally valid, and its state dict loads strictly into a model built from the saved config**. However, **it does not contain the best-validation-MPJPE model**. The saved checkpoint corresponds to **epoch 2 (val MPJPE 81.47 mm)**, whereas the run's true best validation MPJPE is **75.16 mm at epoch 3**.

The discrepancy is caused by the early-stopping/min-delta logic: the checkpoint is saved only when validation loss improves by at least `early_stopping_min_delta = 0.001`. The improvement from epoch 2 to epoch 3 (0.002474 → 0.002090, Δ = 0.000384) is smaller than that threshold, so the epoch-3 model was never written to disk. The printed "Best val MPJPE" line at the end of the log reports the best value seen in the run history, not the MPJPE of the file that was actually saved.

---

## GPU status at validation time

```text
GPU utilisation: 2 %
GPU memory used: 1,377 / 24,564 MiB
Status: free (no other GPU training/eval task active)
```

---

## Files checked

| File | Status | Size |
|---|---|---|
| `outputs/omniview_fusion_v57_h36m_true_gt_medium.pth` | ✅ exists | 55 MB (56,930,450 bytes) |
| `outputs/omniview_fusion_v57_h36m_true_gt_medium.config.json` | ✅ exists | 15 KB |
| `outputs/omniview_fusion_v57_h36m_true_gt_medium.log` | ✅ exists | 1.8 KB |
| `outputs/omniview_fusion_v57_h36m_true_gt_medium_final.pth` | ✅ exists | 55 MB (final epoch 5) |

---

## 1. Training log summary

```text
Epoch 1: train_loss=6.521304, val_loss=0.003491, val_MPJPE=98.11mm
Epoch 2: train_loss=6.914135, val_loss=0.002474, val_MPJPE=81.47mm
Epoch 3: train_loss=5.957940,  val_loss=0.002090, val_MPJPE=75.16mm  <-- run best
Epoch 4: train_loss=5.617853, val_loss=0.002179, val_MPJPE=76.60mm
Epoch 5: train_loss=5.343107, val_loss=0.002520, val_MPJPE=80.21mm
Early stopping at epoch 5 (no val_loss improvement for 3 epochs).
Best val MPJPE: 75.16mm -> outputs\omniview_fusion_v57_h36m_true_gt_medium.pth
```

* Log-reported best: **75.16 mm at epoch 3**.
* Final epoch: **80.21 mm at epoch 5**.

---

## 2. Checkpoint loadability and integrity

### 2.1 File-level checks

```bash
$ ls -lh outputs/omniview_fusion_v57_h36m_true_gt_medium.pth
-rw-r--r-- 1 tuml 1049089 55M Aug 11 13:38 outputs/omniview_fusion_v57_h36m_true_gt_medium.pth
```

### 2.2 PyTorch load test

```python
import torch
ckpt = torch.load("outputs/omniview_fusion_v57_h36m_true_gt_medium.pth", map_location="cpu")
print(ckpt.keys())
# ['epoch', 'model', 'optimizer', 'amp', 'history', 'scheduler', 'ema']
```

Result: **pass**. The file is a valid PyTorch checkpoint and contains the expected keys.

### 2.3 Model architecture match

The model was reconstructed from the saved config and the checkpoint `model` state dict was loaded with `strict=True`:

```python
from experiments.train_omniview_fusion_v5_webbridge_multi import build_model_from_args
import argparse, torch

cfg = json.load(open("outputs/omniview_fusion_v57_h36m_true_gt_medium.config.json"))
args = argparse.Namespace(**cfg)
model = build_model_from_args(args, n_joints=17, n_views=14, device="cpu")
print(sum(p.numel() for p in model.parameters()))  # 3,728,222 (matches log)

ckpt = torch.load("outputs/omniview_fusion_v57_h36m_true_gt_medium.pth", map_location="cpu")
model.load_state_dict(ckpt["model"], strict=True)  # succeeds
```

Result: **pass**. The state dict has 424 entries and matches the v57 architecture exactly. Total parameter count matches the log-reported `3,728,222`.

---

## 3. Checkpoint epoch vs. best val MPJPE

### 3.1 What the checkpoint actually contains

```python
ckpt = torch.load("outputs/omniview_fusion_v57_h36m_true_gt_medium.pth", map_location="cpu")
print("saved epoch:", ckpt["epoch"])        # 2  (0-indexed internally; log epoch 2)
print("history length:", len(ckpt["history"])) # 2
for h in ckpt["history"]:
    print(f"  epoch {h['epoch']}: val MPJPE = {h['val']['mpjpe']*1000:.2f} mm")
```

Output:

```text
saved epoch: 2
history length: 2
  epoch 1: val MPJPE = 98.11 mm
  epoch 2: val MPJPE = 81.47 mm
```

### 3.2 What the run history records as best

| Epoch | Val loss | Val MPJPE (mm) | Saved as best? |
|---:|---:|---:|---|
| 1 | 0.003491 | 98.11 | overwritten by epoch 2 |
| 2 | 0.002474 | 81.47 | **yes** (last time Δloss ≥ 0.001) |
| 3 | 0.002090 | 75.16 | **no** (Δloss = 0.000384 < 0.001) |
| 4 | 0.002179 | 76.60 | no |
| 5 | 0.002520 | 80.21 | no |

### 3.3 Conclusion on the mismatch

* The **best validation MPJPE achieved during training is 75.16 mm at epoch 3**.
* The **checkpoint saved on disk is from epoch 2, with validation MPJPE 81.47 mm**.
* Therefore, **the saved checkpoint does not match the best val MPJPE**.

### 3.4 Root cause

The training script's best-checkpoint criterion (in `motionflow_mv/training/trainer_v2.py`) is:

```python
if val_loss < best_metric - early_stopping_min_delta:
    best_metric = val_loss
    self.save_checkpoint(checkpoint_path)
```

For this run, `early_stopping_min_delta = 0.001` (from `configs/benchmark_v57_h36m_true_gt_medium.yaml`). The epoch-3 improvement (0.002474 → 0.002090) is smaller than 0.001, so no checkpoint was written. The final log line is the best value from the in-memory history, not an assertion about the saved file.

---

## 4. Recommendations

1. **For inference / leaderboard reporting:** Do not use `outputs/omniview_fusion_v57_h36m_true_gt_medium.pth` if the goal is the 75.16 mm model. Use one of:
   - Re-train v57 with the same seed and a smaller `early_stopping_min_delta` (e.g. `0.0001`) so the epoch-3 model is saved.
   - Patch the trainer to save the absolute best by val loss (or val MPJPE) regardless of the early-stopping delta.
   - If the final checkpoint is acceptable, `outputs/omniview_fusion_v57_h36m_true_gt_medium_final.pth` (epoch 5, 80.21 mm) is available, but it is worse than both the epoch-2 and epoch-3 checkpoints.

2. **For documentation / leaderboard:** Update `docs/results_true_gt_h36m.md` to clarify that the **best observed val MPJPE was 75.16 mm**, but the **retrievable checkpoint corresponds to 81.47 mm**.

3. **For future runs:** Consider changing the save criterion from a loss-improvement threshold to a direct `min(val_loss)` tracker so the true best model is always persisted.

---

## 5. Checklist

| Check | Result |
|---|---|
| Checkpoint file exists and is non-empty | ✅ yes (55 MB) |
| Checkpoint loads with `torch.load` | ✅ yes |
| State dict keys match a model built from saved config | ✅ yes (424 entries, strict load) |
| Parameter count matches training log | ✅ yes (3,728,222) |
| Saved epoch matches log-reported best epoch | ❌ no (saved: epoch 2, best: epoch 3) |
| Saved checkpoint MPJPE matches log-reported best MPJPE | ❌ no (saved: 81.47 mm, best: 75.16 mm) |
| Final checkpoint is available as fallback | ✅ yes (epoch 5, 80.21 mm) |

---

## Commands used for this validation

```bash
# GPU status
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits

# Inspect files
ls -lh outputs/omniview_fusion_v57_h36m_true_gt_medium.*

# Load and inspect checkpoint
python - <<'PY'
import torch
ckpt = torch.load("outputs/omniview_fusion_v57_h36m_true_gt_medium.pth", map_location="cpu")
print("keys:", list(ckpt.keys()))
print("epoch:", ckpt["epoch"])
print("history:", [(h["epoch"], h["val"]["mpjpe"]*1000) for h in ckpt["history"]])
PY

# Build model and load state dict
python - <<'PY'
import json, argparse, torch
from experiments.train_omniview_fusion_v5_webbridge_multi import build_model_from_args

cfg = json.load(open("outputs/omniview_fusion_v57_h36m_true_gt_medium.config.json"))
args = argparse.Namespace(**cfg)
model = build_model_from_args(args, 17, 14, "cpu")
ckpt = torch.load("outputs/omniview_fusion_v57_h36m_true_gt_medium.pth", map_location="cpu")
model.load_state_dict(ckpt["model"], strict=True)
print("params:", sum(p.numel() for p in model.parameters()))
PY
```
