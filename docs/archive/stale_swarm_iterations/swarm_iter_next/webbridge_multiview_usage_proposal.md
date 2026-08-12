# WebBridge Multi-View Usage Proposal

**Generated:** 2026-08-09 10:09:57 UTC

**Root:** `data/webbridge`

**Constraint:** read-only; no dataset files were modified.

## 1. Proposal

Use the canonical WebBridge `.npz` files listed below for mixed-dataset multi-view training. All selected files are in **meter units** and follow the canonical `(T, V, J, 2)` layout. MPI-INF-3DHP uses its native 28-joint skeleton and is mapped to the common 17-joint layout by `WebBridgeCanonical17Dataset`.

## 2. Train/Val Splits

| Dataset | Train files | Val files | Notes |
|---------|------------|-----------|-------|
| aist | 1197 | 211 | Whole clip prefixes split 85/15 by deterministic hash. |
| campus | 1 | 1 | Explicit train/val files. |
| h36m | 90 | 15 | Subject 9 reserved for validation. |
| mpi | 20 | 1 | Subject 2 reserved for validation (standard MPI benchmark subject). |
| shelf | 1 | 1 | Explicit train/val files. |

## 3. Data Quality

- Total canonical `.npz` records inspected: **1538**
- Status distribution: `{'OK': 1538}`
- No load/validation issues detected among inspected files.

## 4. Next Steps

1. Run the loader smoke test with the generated YAML:

   ```bash

   python -m motionflow_mv.data.webbridge_mixed_dataset --smoke

   ```

2. If `OmniMultiViewFusionV2`/`V3` rejects variable view counts, add a    `view_mask` so that padded 14-view slots are ignored.

3. Train a small mixed-dataset model and measure cross-dataset MPJPE on    H36M subject 9 and MPI subject 2 before scaling up to A800.
