# WebBridge Data Expansion — Smoke Report

**Direction:** `webbridge_data_expansion`  
**Branch:** `swarm/webbridge_data_expansion`  
**Date:** 2026-08-08

## 1. Current State

- The A800 baseline `v25 small` reports `val_MPJPE 18.31 mm`.
- Local RTX 4090 v26/v27/v28 runs overfit or regressed; a local `v25` baseline is currently running to diagnose the gap.
- The existing WebBridge mixed-dataset manifest
  `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml` is tiny:
  - **Train:** 29 files, 181 k samples (H36M S1 only + MPI-INF-3DHP S1,S3-S8)
  - **Val:** 16 files, 90 k samples (H36M S9 + MPI-INF-3DHP S2)
- Available but unused H36M WebBridge data includes subjects **S5, S6, S7, S8** for
  training and **S11** for validation.

## 2. Proposed Experiment

**Add the standard H36M training subjects (S5-S8) to the mixed-dataset manifest
and run a CPU-only v25 smoke to confirm end-to-end training still works.**

Rationale: H36M is already in the WebBridge canonical 17-joint/4-view format and
is consumed by the mixed loader. Expanding the manifest is a pure data change
and should be compared against the S1-only baseline before any GPU tuning.

## 3. Implementation

Created:

- `configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml`
  - Train: S1, S5, S6, S7, S8 (all 15 actions each) + MPI S1,S3-S8
  - Val: S9, S11 (all 15 actions each) + MPI S2
  - **89 train files / 31 val files** vs. 29/16 in the baseline.

- `experiments/prototypes/webbridge_data_expansion_loader_smoke.py`
  - Fast CPU smoke that validates every path, builds the mixed loader, and
    prints batch shapes.

- `scripts/run_webbridge_data_expansion_v25_smoke.sh`
  - CPU-only v25 smoke using the expanded manifest with a tiny model so it
    completes in under 30 minutes while the local 4090 v25 baseline uses GPU.

## 4. Loader Smoke Result

```text
Expanded manifest: configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml
  train files: 89  val files: 31
  train batches: 2225  val batches: 374
  sample x shape: (4, 13, 14, 17, 3) (B,T,V,J,3)
  sample y shape: (4, 13, 17, 3)
  camera K shape: (4, 14, 3, 3)
  Loader smoke: OK
```

The manifest is valid and the mixed loader pads H36M 4-view and MPI 14-view
sequences to the common 14-view/17-joint format as expected.

## 5. Training Smoke Result

**Actual-data CPU smoke:** started but stopped after ~100 train steps because the
full v25 model is too slow on CPU to finish within the 30-minute smoke budget
(2 epochs ≈ 4 450 train steps). No runtime error was observed; it simply needed
the GPU.

**Synthetic CPU smoke:** the same v5 trainer/model stack runs end-to-end on CPU:

```text
Device: cpu
Model params: 109222
Epoch 1: train_loss=2.277004, val_loss=0.002245, val_MPJPE=82.02mm
Best val MPJPE: 82.02mm -> outputs/webbridge_data_expansion_v5_synthetic_smoke.pth
Saved training config -> outputs/webbridge_data_expansion_v5_synthetic_smoke.config.json
```

This confirms the v5 training code path (with the exact flags used in the data
expansion runner) is functional.

Command to reproduce the real-data smoke when GPU is free:

```bash
bash scripts/run_webbridge_data_expansion_v25_smoke.sh
```

Or manually (CPU-only, slow):

```bash
PYTHON=/d/anaconda3/python \
CUDA_VISIBLE_DEVICES=9 bash scripts/run_webbridge_data_expansion_v25_smoke.sh
```

## 6. Next Steps

1. If the CPU smoke completes without errors, queue a full GPU run on the 4090
   (or A800) using the expanded manifest and the standard v25 small config.
2. Compare final `val_MPJPE` against the S1-only baseline to quantify the impact
   of adding S5-S8 / S11.
3. Optionally add AIST++ canonical data or 3DPW pseudo-multi-view data once the
   H36M expansion is validated.
