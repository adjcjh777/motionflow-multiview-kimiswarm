# P07 WebBridge Data Manifest

**Generated:** 2026-08-07 02:59:11 UTC

**Branch:** `feat/swarm-iter18-omniview`

**Author:** Kimi Code subagent

**Root:** `data/webbridge`

**Constraint:** read-only; no files were modified.

## 1. Summary

| Dataset | Files | OK | Errors | Total size | Total frames | Views | Joints | Notes |
|---------|-------|----|--------|-----------|--------------|-------|--------|-------|
| MPI-INF-3DHP | 43 | 43 | 0 | 1.54 GB | 376,952 | 4, 14 | 28 | MPI-INF-3DHP canonical multi-view npz files |
| H36M | 45 | 45 | 0 | 238.02 MB | 203,824 | 4 | 17 | Human3.6M canonical multi-view npz files (meters) |
| AIST++ | 1408 | 1408 | 0 | 1.47 GB | 1,123,873 | 9 | 17 | AIST++ canonical multi-view npz files |
| Shelf/Campus | 4 | 4 | 0 | 9.69 MB | 4,623 | 3, 5 | 17 | Shelf and Campus canonical multi-view npz files |

**Total audited storage:** 3.25 GB

## 2. MPI-INF-3DHP

Canonical multi-view ``.npz`` files under `data/webbridge/mpi_inf_3dhp`.

### 2.1 Coverage

- Subject 1: sequences 01, 02
- Subject 2: sequences 01
- Subject 3: sequences 01, 02
- Subject 4: sequences 01, 02
- Subject 5: sequences 01, 02
- Subject 6: sequences 01, 02
- Subject 7: sequences 01, 02
- Subject 8: sequences 01, 02

### 2.2 Per-file inventory

| File | Status | T | V | J | Size | Notes |
|------|--------|---|---|---|------|-------|
| `s_01_seq_01_02_v14_multiview_m.npz` | `OK` | 18846 | 14 | 28 | 63.64 MB | — |
| `s_01_seq_01_02_v4_multiview_m.npz` | `OK` | 18846 | 4 | 28 | 26.03 MB | — |
| `s_01_seq_01_v14_multiview.npz` | `OK` | 6416 | 14 | 28 | 21.79 MB | — |
| `s_01_seq_01_v14_multiview_m.npz` | `OK` | 6416 | 14 | 28 | 21.84 MB | — |
| `s_01_seq_01_v14_multiview_m_smoke.npz` | `OK` | 250 | 14 | 28 | 2.03 MB | — |
| `s_01_seq_01_v4_multiview.npz` | `OK` | 6416 | 4 | 28 | 8.81 MB | — |
| `s_01_seq_01_v4_multiview_m.npz` | `OK` | 6416 | 4 | 28 | 8.87 MB | — |
| `s_01_seq_02_v14_multiview.npz` | `OK` | 12430 | 14 | 28 | 41.69 MB | — |
| `s_01_seq_02_v14_multiview_m.npz` | `OK` | 12430 | 14 | 28 | 41.80 MB | — |
| `s_01_seq_02_v14_multiview_m_smoke.npz` | `OK` | 250 | 14 | 28 | 2.03 MB | — |
| `s_01_seq_02_v4_multiview.npz` | `OK` | 12430 | 4 | 28 | 17.05 MB | — |
| `s_01_seq_02_v4_multiview_m.npz` | `OK` | 12430 | 4 | 28 | 17.17 MB | — |
| `s_02_seq_01_v14_multiview.npz` | `OK` | 6502 | 14 | 28 | 22.09 MB | — |
| `s_02_seq_01_v14_multiview_m.npz` | `OK` | 6502 | 14 | 28 | 22.15 MB | — |
| `s_02_seq_01_v14_multiview_m_smoke.npz` | `OK` | 500 | 14 | 28 | 4.06 MB | — |
| `s_03_seq_01_v14_multiview.npz` | `OK` | 12489 | 14 | 28 | 41.94 MB | — |
| `s_03_seq_01_v14_multiview_m.npz` | `OK` | 12489 | 14 | 28 | 101.39 MB | — |
| `s_03_seq_01_v4_multiview.npz` | `OK` | 12489 | 4 | 28 | 17.30 MB | — |
| `s_03_seq_01_v4_multiview_m.npz` | `OK` | 12489 | 4 | 28 | 34.69 MB | — |
| `s_03_seq_02_v14_multiview.npz` | `OK` | 12283 | 14 | 28 | 41.24 MB | — |
| `s_03_seq_02_v14_multiview_m.npz` | `OK` | 12283 | 14 | 28 | 99.71 MB | — |
| `s_03_seq_02_v4_multiview.npz` | `OK` | 12283 | 4 | 28 | 17.06 MB | — |
| `s_03_seq_02_v4_multiview_m.npz` | `OK` | 12283 | 4 | 28 | 34.11 MB | — |
| `s_04_seq_01_v14_multiview.npz` | `OK` | 6171 | 14 | 28 | 20.70 MB | — |
| `s_04_seq_01_v14_multiview_m.npz` | `OK` | 6171 | 14 | 28 | 50.10 MB | — |
| `s_04_seq_02_v14_multiview.npz` | `OK` | 6675 | 14 | 28 | 22.37 MB | — |
| `s_04_seq_02_v14_multiview_m.npz` | `OK` | 6675 | 14 | 28 | 54.19 MB | — |
| `s_05_seq_01_v14_multiview.npz` | `OK` | 12820 | 14 | 28 | 43.01 MB | — |
| `s_05_seq_01_v14_multiview_m.npz` | `OK` | 12820 | 14 | 28 | 104.07 MB | — |
| `s_05_seq_02_v14_multiview.npz` | `OK` | 12312 | 14 | 28 | 41.02 MB | — |
| `s_05_seq_02_v14_multiview_m.npz` | `OK` | 12312 | 14 | 28 | 99.95 MB | — |
| `s_06_seq_01_v14_multiview.npz` | `OK` | 6188 | 14 | 28 | 20.55 MB | — |
| `s_06_seq_01_v14_multiview_m.npz` | `OK` | 6188 | 14 | 28 | 50.24 MB | — |
| `s_06_seq_02_v14_multiview.npz` | `OK` | 6145 | 14 | 28 | 20.44 MB | — |
| `s_06_seq_02_v14_multiview_m.npz` | `OK` | 6145 | 14 | 28 | 49.89 MB | — |
| `s_07_seq_01_v14_multiview.npz` | `OK` | 6239 | 14 | 28 | 20.82 MB | — |
| `s_07_seq_01_v14_multiview_m.npz` | `OK` | 6239 | 14 | 28 | 50.65 MB | — |
| `s_07_seq_02_v14_multiview.npz` | `OK` | 6320 | 14 | 28 | 21.18 MB | — |
| `s_07_seq_02_v14_multiview_m.npz` | `OK` | 6320 | 14 | 28 | 51.31 MB | — |
| `s_08_seq_01_v14_multiview.npz` | `OK` | 6468 | 14 | 28 | 21.85 MB | — |
| `s_08_seq_01_v14_multiview_m.npz` | `OK` | 6468 | 14 | 28 | 52.51 MB | — |
| `s_08_seq_02_v14_multiview.npz` | `OK` | 6054 | 14 | 28 | 20.27 MB | — |
| `s_08_seq_02_v14_multiview_m.npz` | `OK` | 6054 | 14 | 28 | 49.15 MB | — |

## 3. Human3.6M

Canonical multi-view ``.npz`` files under `data/webbridge/h36m_meters`.

### 3.1 Coverage

- Subject 1: 15 action files (2–16)
- Subject 9: 15 action files (2–16)
- Subject 11: 15 action files (2–16)

### 3.2 Per-file inventory

| File | Status | T | V | J | Size | Notes |
|------|--------|---|---|---|------|-------|
| `s_01_acts_02_multiview_m.npz` | `OK` | 2995 | 4 | 17 | 3.50 MB | — |
| `s_01_acts_03_multiview_m.npz` | `OK` | 7657 | 4 | 17 | 8.94 MB | — |
| `s_01_acts_04_multiview_m.npz` | `OK` | 5078 | 4 | 17 | 5.93 MB | — |
| `s_01_acts_05_multiview_m.npz` | `OK` | 2414 | 4 | 17 | 2.82 MB | — |
| `s_01_acts_06_multiview_m.npz` | `OK` | 4902 | 4 | 17 | 5.72 MB | — |
| `s_01_acts_07_multiview_m.npz` | `OK` | 2159 | 4 | 17 | 2.52 MB | — |
| `s_01_acts_08_multiview_m.npz` | `OK` | 2222 | 4 | 17 | 2.60 MB | — |
| `s_01_acts_09_multiview_m.npz` | `OK` | 5916 | 4 | 17 | 6.91 MB | — |
| `s_01_acts_10_multiview_m.npz` | `OK` | 5765 | 4 | 17 | 6.73 MB | — |
| `s_01_acts_11_multiview_m.npz` | `OK` | 5089 | 4 | 17 | 5.94 MB | — |
| `s_01_acts_12_multiview_m.npz` | `OK` | 2110 | 4 | 17 | 2.47 MB | — |
| `s_01_acts_13_multiview_m.npz` | `OK` | 3232 | 4 | 17 | 3.77 MB | — |
| `s_01_acts_14_multiview_m.npz` | `OK` | 6610 | 4 | 17 | 7.72 MB | — |
| `s_01_acts_15_multiview_m.npz` | `OK` | 3439 | 4 | 17 | 4.02 MB | — |
| `s_01_acts_16_multiview_m.npz` | `OK` | 2506 | 4 | 17 | 2.93 MB | — |
| `s_09_acts_02_multiview_m.npz` | `OK` | 5055 | 4 | 17 | 5.90 MB | — |
| `s_09_acts_03_multiview_m.npz` | `OK` | 11179 | 4 | 17 | 13.05 MB | — |
| `s_09_acts_04_multiview_m.npz` | `OK` | 5349 | 4 | 17 | 6.25 MB | — |
| `s_09_acts_05_multiview_m.npz` | `OK` | 5422 | 4 | 17 | 6.33 MB | — |
| `s_09_acts_06_multiview_m.npz` | `OK` | 7140 | 4 | 17 | 8.34 MB | — |
| `s_09_acts_07_multiview_m.npz` | `OK` | 3932 | 4 | 17 | 4.59 MB | — |
| `s_09_acts_08_multiview_m.npz` | `OK` | 2755 | 4 | 17 | 3.22 MB | — |
| `s_09_acts_09_multiview_m.npz` | `OK` | 6033 | 4 | 17 | 7.04 MB | — |
| `s_09_acts_10_multiview_m.npz` | `OK` | 5864 | 4 | 17 | 6.85 MB | — |
| `s_09_acts_11_multiview_m.npz` | `OK` | 8711 | 4 | 17 | 10.17 MB | — |
| `s_09_acts_12_multiview_m.npz` | `OK` | 3795 | 4 | 17 | 4.43 MB | — |
| `s_09_acts_13_multiview_m.npz` | `OK` | 6624 | 4 | 17 | 7.73 MB | — |
| `s_09_acts_14_multiview_m.npz` | `OK` | 4058 | 4 | 17 | 4.74 MB | — |
| `s_09_acts_15_multiview_m.npz` | `OK` | 4454 | 4 | 17 | 5.20 MB | — |
| `s_09_acts_16_multiview_m.npz` | `OK` | 3388 | 4 | 17 | 3.96 MB | — |
| `s_11_acts_02_multiview_m.npz` | `OK` | 3104 | 4 | 17 | 3.63 MB | — |
| `s_11_acts_03_multiview_m.npz` | `OK` | 4882 | 4 | 17 | 5.70 MB | — |
| `s_11_acts_04_multiview_m.npz` | `OK` | 4478 | 4 | 17 | 5.23 MB | — |
| `s_11_acts_05_multiview_m.npz` | `OK` | 3503 | 4 | 17 | 4.09 MB | — |
| `s_11_acts_06_multiview_m.npz` | `OK` | 6882 | 4 | 17 | 8.04 MB | — |
| `s_11_acts_07_multiview_m.npz` | `OK` | 2888 | 4 | 17 | 3.37 MB | — |
| `s_11_acts_08_multiview_m.npz` | `OK` | 2066 | 4 | 17 | 2.41 MB | — |
| `s_11_acts_09_multiview_m.npz` | `OK` | 4036 | 4 | 17 | 4.71 MB | — |
| `s_11_acts_10_multiview_m.npz` | `OK` | 3845 | 4 | 17 | 4.49 MB | — |
| `s_11_acts_11_multiview_m.npz` | `OK` | 5177 | 4 | 17 | 6.05 MB | — |
| `s_11_acts_12_multiview_m.npz` | `OK` | 3535 | 4 | 17 | 4.13 MB | — |
| `s_11_acts_13_multiview_m.npz` | `OK` | 4542 | 4 | 17 | 5.30 MB | — |
| `s_11_acts_14_multiview_m.npz` | `OK` | 3258 | 4 | 17 | 3.81 MB | — |
| `s_11_acts_15_multiview_m.npz` | `OK` | 2622 | 4 | 17 | 3.06 MB | — |
| `s_11_acts_16_multiview_m.npz` | `OK` | 3153 | 4 | 17 | 3.68 MB | — |

## 4. AIST++

Canonical multi-view ``.npz`` files under `data/webbridge/aistpp_canonical`.

- **Total files:** 1,408
- **Unique clips:** 298
- **Healthy (OK):** 1,408
- **With errors:** 0
- **Total frames:** 1,123,873
- **Shape convention:** each file contains a single camera/channel; multiple `_chNN` files may be pooled to form a multi-view clip.

### 4.1 Genre distribution

| Genre | Files | Example |
|-------|-------|---------|
| BR | 141 | `gBR_sBM_cAll_d04_mBR0_ch01_multiview.npz` |
| HO | 141 | `gHO_sBM_cAll_d19_mHO0_ch01_multiview.npz` |
| JB | 141 | `gJB_sBM_cAll_d07_mJB0_ch01_multiview.npz` |
| JS | 141 | `gJS_sBM_cAll_d01_mJS0_ch01_multiview.npz` |
| KR | 141 | `gKR_sBM_cAll_d28_mKR0_ch01_multiview.npz` |
| LH | 141 | `gLH_sBM_cAll_d16_mLH0_ch01_multiview.npz` |
| LO | 141 | `gLO_sBM_cAll_d13_mLO0_ch01_multiview.npz` |
| MH | 141 | `gMH_sBM_cAll_d22_mMH0_ch01_multiview.npz` |
| PO | 140 | `gPO_sBM_cAll_d10_mPO0_ch01_multiview.npz` |
| WA | 140 | `gWA_sBM_cAll_d25_mWA0_ch01_multiview.npz` |

### 4.2 Per-clip aggregated inventory (first 20 of 298)

| Clip prefix | Files | Total frames | T | V | J | Notes |
|-------------|-------|--------------|---|---|---|-------|
| `gBR_sBM_cAll_d04_mBR0` | 10 | 7,200 | 720 | 9 | 17 | — |
| `gBR_sBM_cAll_d04_mBR1` | 10 | 6,400 | 640 | 9 | 17 | — |
| `gBR_sBM_cAll_d04_mBR2` | 10 | 5,760 | 576 | 9 | 17 | — |
| `gBR_sBM_cAll_d04_mBR3` | 10 | 5,240 | 524 | 9 | 17 | — |
| `gBR_sBM_cAll_d05_mBR0` | 10 | 7,200 | 720 | 9 | 17 | — |
| `gBR_sBM_cAll_d05_mBR1` | 10 | 6,400 | 640 | 9 | 17 | — |
| `gBR_sBM_cAll_d05_mBR4` | 10 | 4,800 | 480 | 9 | 17 | — |
| `gBR_sBM_cAll_d05_mBR5` | 10 | 4,430 | 443 | 9 | 17 | — |
| `gBR_sBM_cAll_d06_mBR2` | 10 | 5,760 | 576 | 9 | 17 | — |
| `gBR_sBM_cAll_d06_mBR3` | 10 | 5,240 | 524 | 9 | 17 | — |
| `gBR_sBM_cAll_d06_mBR4` | 10 | 4,800 | 480 | 9 | 17 | — |
| `gBR_sBM_cAll_d06_mBR5` | 10 | 4,430 | 443 | 9 | 17 | — |
| `gBR_sFM_cAll_d04_mBR0` | 1 | 2,878 | 2878 | 9 | 17 | — |
| `gBR_sFM_cAll_d04_mBR1` | 1 | 2,558 | 2558 | 9 | 17 | — |
| `gBR_sFM_cAll_d04_mBR2` | 1 | 2,302 | 2302 | 9 | 17 | — |
| `gBR_sFM_cAll_d04_mBR3` | 1 | 2,093 | 2093 | 9 | 17 | — |
| `gBR_sFM_cAll_d04_mBR4` | 2 | 3,838 | 1919 | 9 | 17 | — |
| `gBR_sFM_cAll_d04_mBR5` | 1 | 1,771 | 1771 | 9 | 17 | — |
| `gBR_sFM_cAll_d05_mBR1` | 1 | 2,558 | 2558 | 9 | 17 | — |
| `gBR_sFM_cAll_d05_mBR2` | 1 | 2,302 | 2302 | 9 | 17 | — |

> 278 additional clips omitted for brevity.


## 5. Shelf / Campus

Canonical multi-view ``.npz`` files under `data/webbridge/shelf_campus`.

| File | Dataset | Split | Views | T | J | Size | Status | Notes |
|------|---------|-------|-------|---|---|------|--------|-------|
| `campus_seq1_train_v3_multiview_m.npz` | Campus | train | 3 | 1138 | 17 | 1.77 MB | OK | — |
| `campus_seq1_val_v3_multiview_m.npz` | Campus | val | 3 | 285 | 17 | 456.20 KB | OK | — |
| `shelf_seq1_train_v5_multiview_m.npz` | Shelf | train | 5 | 2560 | 17 | 5.98 MB | OK | — |
| `shelf_seq1_val_v5_multiview_m.npz` | Shelf | val | 5 | 640 | 17 | 1.50 MB | OK | — |

## 6. Notes and blockers

- All reported sizes are on-disk ``.npz`` sizes.

- `OK` means the file contains all six canonical keys and the expected shapes.

- AIST++ files are stored per camera/channel; multi-view clips are built by grouping ``_chNN`` files with the same clip prefix.

- No validation errors were detected across the audited files.
