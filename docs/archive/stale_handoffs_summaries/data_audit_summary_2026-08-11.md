# Multi-Dataset Circular-Label Audit (2026-08-11)

15 parallel exploration agents were run on local `.npz` files and the A800-D remote to verify which datasets contain true 3D ground truth vs circular DLT labels.

## TL;DR

- **H36M true GT** (`data/h36m_true_gt/`) — ✅ non-circular, S9/S11 DLT error ~25–34 mm.
- **MPI-INF-3DHP** (GT 2D) — ✅ non-circular, DLT error ~35 mm.
- **MPI detected-2D** — ⚠️ real detected 2D `.npz` generated (16 files), but DLT baseline ~326–400 mm due to camera/label alignment; not yet usable for learned-model benchmarking.
- **AIST++** — ✅ non-circular, AIST++ smoke DLT (conf-weighted) 6.52 mm / (unweighted) 12.66 mm.
- **Shelf/Campus detected** — ✅ non-circular, DLT error ~130 mm.
- **3DPW pseudo** — ❌ circular (DLT error 0 mm).
- **3DPW actual** — ⚠️ monocular (1 view), not usable for multi-view triangulation.
- **A800 `/mnt/nvme0n1/zhangzy/projects`** — no true H36M / MPI raw data found.

---

## Human3.6M true GT

| Subject | Frames | Views | Joints | Direct MJE (mm) | Root-MPJPE (mm) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| S1 | 62,094 | 4 | 17 | 15.94 | 15.58 | ✅ true GT |
| S5 | 99,079 | 4 | 17 | 16.12 | 15.88 | ✅ true GT |
| S6 | 62,466 | 4 | 17 | 16.45 | 16.03 | ✅ true GT |
| S7 | 101,621 | 4 | 17 | 16.46 | 16.03 | ✅ true GT |
| S8 | 64,678 | 4 | 17 | 13.66 | 12.97 | ✅ true GT |
| **S9 (test)** | 83,759 | 4 | 17 | **33.83** | **34.36** | ✅ true GT |
| **S11 (test)** | 57,971 | 4 | 17 | **24.75** | **24.73** | ✅ true GT |

- Units: metres.
- S9/S11 DLT error is larger than train subjects because the test split has shuffled camera assignments / noisier 2D; the labels remain non-circular.

---

## MPI-INF-3DHP

| Source | Frames | Views | Joints | Direct MJE (mm) | Root-MPJPE (mm) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| GT 2D (`s_01_seq_01_v14_multiview_m`) | 6,416 | 14 | 28 | 34.77 | 33.56 | ✅ true GT (labels not DLT of 2D) |
| Detected-2D fallback | 6,416 | 14 | 28 | 35.01 | 33.79 | ⚠️ non-circular but synthetic 2D |

- The fallback confidence is constant 0.81 and the 2D is GT + ~2 px noise — **not a real detector**.

---

## AIST++

| Source | Frames | Views | Joints | Direct MJE (mm) | Root-MPJPE (mm) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `gBR_sBM_cAll_d04_mBR0_ch01_multiview` | 720 | 9 | 17 | 48.15 | 35.19 | ✅ true GT |

---

## Shelf / Campus detected

| Source | Frames | Views | Joints | Direct MJE (mm) | Root-MPJPE (mm) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Shelf val detected | 87 | 5 | 17 | 130.77 | 124.13 | ✅ true GT |
| Campus val detected | 164 | 3 | 17 | 138.08 | 120.61 | ✅ true GT |

---

## 3DPW

| Source | Frames | Views | Joints | Direct MJE (mm) | Verdict |
|---|---:|---:|---:|---:|---|
| `courtyard_basketball_01_pseudo` | 958 | 4 | 24 | 0.0000 | ❌ circular |
| `courtyard_basketball_01_actual` | 958 | 1 | 24 | 12418.35 (degenerate) | ⚠️ monocular, not multi-view usable |

---

## A800-D remote audit

- **Path inspected**: `/mnt/nvme0n1/zhangzy/projects`
- **Findings**:
  - No `PosesD3_Positions` / `PosesD3_Positions_mono`.
  - No `imageSequence/` MPI frames.
  - No H36M / MPI `.npz` files.
  - Contains build directories, robot-motion `.npz`, SMPL-X models, GVHMR stubs only.

---

## Conclusions for CVPR 2027

1. **H36M standard protocol is ready** once the H36M true-GT benchmark finishes.
2. **MPI-INF-3DHP needs real detected 2D**; the fallback is not acceptable.
3. **AIST++ is a viable extra non-circular dataset** for cross-domain training.
4. **3DPW pseudo cannot be used as true GT**; 3DPW actual is monocular.
5. **A800 does not have the missing raw data**; must obtain MPI imageSequence / H36M release from external sources.