# WebBridge `.npz` Quality Audit Report

**Date:** 2026-08-05 20:40:05 UTC

**Root:** `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm\data\webbridge\mpi_inf_3dhp`

**Audit script:** `experiments/audit_webbridge_npz.py`

**Constraint:** read-only; no files were modified.

## 1. Summary

- Total ``.npz`` files scanned: **23**
- Total size: **708.51 MB**
- Fully canonical / healthy: **23**
- Status distribution:
  - `OK`: 23

## 2. Per-Dataset Summary

| Dataset | Files | OK | Warnings/Non-canonical | Total size |
|---------|-------|----|------------------------|-----------|
| webbridge | 23 | 23 | 0 | 708.51 MB |

## 3. Smoke Files

| File | Status | T | V | J | NaN frac | Zero-conf frac | Notes |
|------|--------|---|---|---|----------|----------------|-------|
| `s_01_seq_01_v14_multiview_m_smoke.npz` | `OK` | 250 | 14 | 28 | 0.00e+00 | 0.00% | — |
| `s_01_seq_02_v14_multiview_m_smoke.npz` | `OK` | 250 | 14 | 28 | 0.00e+00 | 0.00% | — |
| `s_02_seq_01_v14_multiview_m_smoke.npz` | `OK` | 500 | 14 | 28 | 0.00e+00 | 0.00% | — |

## 4. Full Per-File Status

| Path | Status | T | V | J | Missing keys | Notes |
|------|--------|---|---|---|--------------|-------|
| `s_01_seq_01_02_v14_multiview_m.npz` | `OK` | 18846 | 14 | 28 | — | — |
| `s_01_seq_01_02_v4_multiview_m.npz` | `OK` | 18846 | 4 | 28 | — | — |
| `s_01_seq_01_v14_multiview.npz` | `OK` | 6416 | 14 | 28 | — | — |
| `s_01_seq_01_v14_multiview_m.npz` | `OK` | 6416 | 14 | 28 | — | — |
| `s_01_seq_01_v14_multiview_m_smoke.npz` | `OK` | 250 | 14 | 28 | — | — |
| `s_01_seq_01_v4_multiview.npz` | `OK` | 6416 | 4 | 28 | — | — |
| `s_01_seq_01_v4_multiview_m.npz` | `OK` | 6416 | 4 | 28 | — | — |
| `s_01_seq_02_v14_multiview.npz` | `OK` | 12430 | 14 | 28 | — | — |
| `s_01_seq_02_v14_multiview_m.npz` | `OK` | 12430 | 14 | 28 | — | — |
| `s_01_seq_02_v14_multiview_m_smoke.npz` | `OK` | 250 | 14 | 28 | — | — |
| `s_01_seq_02_v4_multiview.npz` | `OK` | 12430 | 4 | 28 | — | — |
| `s_01_seq_02_v4_multiview_m.npz` | `OK` | 12430 | 4 | 28 | — | — |
| `s_02_seq_01_v14_multiview.npz` | `OK` | 6502 | 14 | 28 | — | — |
| `s_02_seq_01_v14_multiview_m.npz` | `OK` | 6502 | 14 | 28 | — | — |
| `s_02_seq_01_v14_multiview_m_smoke.npz` | `OK` | 500 | 14 | 28 | — | — |
| `s_03_seq_01_v14_multiview.npz` | `OK` | 12489 | 14 | 28 | — | — |
| `s_03_seq_01_v14_multiview_m.npz` | `OK` | 12489 | 14 | 28 | — | — |
| `s_03_seq_01_v4_multiview.npz` | `OK` | 12489 | 4 | 28 | — | — |
| `s_03_seq_01_v4_multiview_m.npz` | `OK` | 12489 | 4 | 28 | — | — |
| `s_03_seq_02_v14_multiview.npz` | `OK` | 12283 | 14 | 28 | — | — |
| `s_03_seq_02_v14_multiview_m.npz` | `OK` | 12283 | 14 | 28 | — | — |
| `s_03_seq_02_v4_multiview.npz` | `OK` | 12283 | 4 | 28 | — | — |
| `s_03_seq_02_v4_multiview_m.npz` | `OK` | 12283 | 4 | 28 | — | — |

## 5. Issues and Recommendations

- No load/validation errors detected.
- All scanned files contain the canonical keys.
- No NaN/Inf values observed in canonical arrays.
