# A800-D missing-data check

**Scope:** Read-only inspection of A800-D `/mnt/nvme0n1/zhangzy/projects` and the related motionflow checkout directories for MPI-INF-3DHP and AIST++ assets that could fill local gaps.

**Date:** 2026-08-11

## Summary

A800-D does **not** hold any real-detected MPI-INF-3DHP 2D data or any AIST++ training data. The only usable MPI assets are the same GT-projected canonical `.npz` files that already exist locally (and A800’s copy is less complete than the local copy).

## What was checked

- `/mnt/nvme0n1/zhangzy/projects` and the sibling motionflow directories on A800:
  - `motionflow-multiview-kimiswarm`
  - `motionflow-multiview-kimiswarm-iter20`
  - `motionflow-mv-detected-long`
  - `motionflow-mv-h36m-truegt`
- Searched for directories/files named `mpi*`, `3dhp`, `aist`, `aistpp`, `imageSequence`.
- Inspected the contents of any MPI `.npz` found.

## MPI-INF-3DHP

### Found on A800

`/mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm/data/webbridge/mpi_inf_3dhp/` contains 15 canonical multi-view `.npz` files:

```
s_01_seq_01_02_v14_multiview_m.npz
s_01_seq_01_v14_multiview_m.npz
s_01_seq_02_v14_multiview_m.npz
s_02_seq_01_v14_multiview_m.npz
s_03_seq_01_v14_multiview_m.npz
s_03_seq_02_v14_multiview_m.npz
s_04_seq_01_v14_multiview_m.npz
s_04_seq_02_v14_multiview_m.npz
s_05_seq_01_v14_multiview_m.npz
s_05_seq_02_v14_multiview_m.npz
s_06_seq_01_v14_multiview_m.npz
s_06_seq_02_v14_multiview_m.npz
s_07_seq_01_v14_multiview_m.npz
s_07_seq_02_v14_multiview_m.npz
s_08_seq_01_v14_multiview_m.npz
s_08_seq_02_v14_multiview_m.npz
```

Keys in each `.npz`:

```text
points_2d    (T, 14, 28, 2)
confidences  (T, 14, 28)
joints_3d    (T, 28, 3)
camera_K     (14, 3, 3)
camera_R     (14, 3, 3)
camera_t     (14, 3)
```

All `confidences` are exactly `1.0`, indicating the 2D points were derived directly from the official MPI annotations (`annot2`), not from a real detector.

### What A800 is missing

- **No `imageSequence/` raw video frames.**
- **No real-detected 2D keypoints** (e.g. CPN/HRNet/OpenPose/MediaPipe).
- **No MPI-INF-3DHP test set** (`test_set/TS*.npz`).
- **Missing `s_02_seq_02`** (subject 2, sequence 2).

### Comparison to local

Local `data/webbridge/mpi_inf_3dhp/` already contains the same canonical files plus additional variants (`_v4_`, `_smoke`, `_17j`, `s_02_seq_02`, `test_set/`, etc.). The local repository also already has `mpi_inf_3dhp_detected_2d/` work in progress. Therefore the A800 MPI copy does **not** add any missing data.

## AIST++

### Found on A800

- Split/config files reference `data/webbridge/aistpp_canonical/...`:
  - `configs/splits/webbridge_aistpp_train_val.yaml`
  - `configs/splits/webbridge_h36m_mpi_aist_mixed_train_val_expanded.yaml`
  - `scripts/download_aistpp.py`
  - `experiments/convert_aistpp_v1.py`

### What A800 is missing

- **No `data/webbridge/aistpp/` directory.**
- **No `data/webbridge/aistpp_canonical/` directory.**
- **No AIST++ `.npz` files anywhere** under the checked A800 motionflow trees.

Local `data/webbridge/aistpp_canonical/` already contains the canonical AIST++ multi-view `.npz` files, so A800 cannot fill this gap.

## Conclusion / blockers

| Dataset        | A800 asset useful? | Blocker                                                                 |
|----------------|--------------------|--------------------------------------------------------------------------|
| MPI-INF-3DHP   | No                 | A800 only has the same GT-projected `.npz`; real detected 2D is missing. |
| AIST++         | No                 | A800 has no AIST++ data files at all.                                  |

The local blockers documented in `AGENTS.md` remain:

- MPI-INF-3DHP still needs raw `imageSequence/` plus a real detector pass to generate real-detected 2D.
- AIST++ smoke integration is already present locally; a full medium run is pending GPU availability.

No files were modified on A800; this check was read-only.
