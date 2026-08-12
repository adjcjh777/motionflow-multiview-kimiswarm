# MPI-INF-3DHP Detected-2D Protocol

To match the standard literature protocol, the MPI-INF-3DHP inputs should come from an off-the-shelf 2D keypoint detector rather than the dataset's own `annot2` ground-truth 2D projections.

## Quick start (fallback / placeholder)

```bash
python scripts/generate_mpi_detected_2d.py \
    --input_dir data/webbridge/mpi_inf_3dhp \
    --output_dir data/webbridge/mpi_inf_3dhp_detected_2d \
    --detector fallback \
    --fallback_noise 2.0
```

This adds 2 px Gaussian noise to the official GT 2D as a placeholder. The
resulting `.npz` files have the same structure as the originals and can be used
for protocol-development smoke tests.

## Real detector pipeline

The skeleton script already supports two detector backends:

- `mediapipe` — runs MediaPipe Pose on the raw MPI frames.
- `openpose` — runs OpenPose via `cv2.dnn` if weights are placed under
  `models/openpose/`.

Install the dependency and run:

```bash
pip install mediapipe
python scripts/generate_mpi_detected_2d.py \
    --input_dir data/webbridge/mpi_inf_3dhp \
    --output_dir data/webbridge/mpi_inf_3dhp_detected_2d \
    --detector mediapipe \
    --image_dir data/webbridge/mpi_inf_3dhp/raw
```

## Split files

- `configs/splits/mpiinf3dhp_detected_2d.yaml` points to the generated directory.
  Update it if you change `output_dir`.
- `configs/splits/mpi_inf_3dhp_detected_2d_baseline.yaml` is a DLT-baseline split
  that follows the same train/val/test protocol and is consumed by
  `scripts/eval_mpi_detected_2d_baseline.py`.
- `configs/splits/mpi_inf_3dhp_detected_2d_baseline_smoke.yaml` is a small
  smoke split for quick local validation.

## DLT baseline

After generating (or downloading) the detected-2D `.npz` files, run the
confidence-weighted DLT baseline:

```bash
python scripts/eval_mpi_detected_2d_baseline.py
```

Options:

```bash
python scripts/eval_mpi_detected_2d_baseline.py \
    --config configs/splits/mpi_inf_3dhp_detected_2d_baseline.yaml \
    --output outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json \
    --device cpu
```

The script triangulates the stored 2D keypoints using the calibrated cameras,
compares the result to the true 3D ground truth, and writes per-file and
aggregate MPJPE/PA-MPJPE to JSON.

## Why this matters

The standard MPI-INF-3DHP test server expects submissions under the
"detected 2D" protocol. Training on GT 2D and comparing to methods that use CPN
or HRNet detections is not apples-to-apples. Generate (or download) real
keypoint detections before final model selection and before submitting to the
official test server.
