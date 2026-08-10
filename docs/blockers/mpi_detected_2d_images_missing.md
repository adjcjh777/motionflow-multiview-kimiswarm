# Blocker: MPI-INF-3DHP raw images missing

**Date:** 2026-08-10

## What we tried

Installed MediaPipe to generate real detected-2D keypoints for MPI-INF-3DHP:

```bash
pip install mediapipe
python scripts/generate_mpi_detected_2d.py \
    --input_dir data/webbridge/mpi_inf_3dhp \
    --output_dir data/webbridge/mpi_inf_3dhp_detected_2d \
    --detector mediapipe \
    --image_dir data/webbridge/mpi_inf_3dhp/raw
```

## What we found

`data/webbridge/mpi_inf_3dhp/raw/S*/Seq*/` only contains:
- `annot.mat`
- `camera.calibration`

No `imageSequence/` directory with raw `.jpg`/`.png` frames.

## Why it blocks

The standard MPI-INF-3DHP protocol requires detected 2D inputs (e.g., CPN / HRNet / OpenPose). Training on `annot2` (GT 2D) and comparing to literature is not apples-to-apples.

## How to unblock

Download the original MPI-INF-3DHP image archives and extract them under `data/webbridge/mpi_inf_3dhp/raw/S*/Seq*/imageSequence/`. Then rerun the MediaPipe detector script above.
