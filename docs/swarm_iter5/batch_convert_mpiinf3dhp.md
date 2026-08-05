# Batch Conversion of Additional MPI-INF-3DHP Subjects

**Goal:** Convert more MPI-INF-3DHP subjects/sequences to the canonical WebBridge `.npz` format, going beyond the initial S1/S2 samples.

## What was done

1. **Audited existing conversion code**
   - `motionflow_mv/data/webbridge_loader.py` already exposes `convert_mpiinf3dhp(...)`, which produces the canonical `(T, V, J, 2)` / `(T, J, 3)` arrays plus calibrated cameras.
   - Existing canonical files live under `data/webbridge/mpi_inf_3dhp/` and follow the naming pattern `s_{subject:02d}_seq_{seq:02d}_v{14|4}_multiview[_m].npz`.
   - The raw starter kit only contained S1/Seq1, S1/Seq2, and S2/Seq1. The full dataset has subjects S1–S8 with two sequences each.

2. **Created a batch converter/downloader**
   - New script: `experiments/batch_convert_mpiinf3dhp_v1.py`
     - Discovers raw `annot.mat` + `camera.calibration` pairs under `data/webbridge/mpi_inf_3dhp/raw`.
     - Optionally downloads missing annotation/calibration files from `https://vcai.mpi-inf.mpg.de/3dhp-dataset/` (image frames are skipped because they are not needed for the canonical `.npz`).
     - Produces both 14-view and 4-view subsets, in millimeters and meters.
   - New validation helper: `tmp/validate_mpiinf3dhp_npz.py` checks keys, shapes, and plausible camera/joint ranges.

3. **Downloaded and converted extra data**
   - Downloaded `S3/Seq1` and `S3/Seq2` annotation/calibration files (~196 MB per `annot.mat`).
   - Generated the following canonical files:
     - `s_03_seq_01_v14_multiview.npz` / `_m.npz`
     - `s_03_seq_01_v4_multiview.npz` / `_m.npz`
     - `s_03_seq_02_v14_multiview.npz` / `_m.npz`
     - `s_03_seq_02_v4_multiview.npz` / `_m.npz`

4. **Verified with a short training smoke test**
   - Ran `experiments/train_ray_attention_temporal_mpiinf3dhp.py` for 1 epoch using the newly converted S3/Seq1 data as extra training material.
   - Command:
     ```bash
     conda run -n mf python experiments/train_ray_attention_temporal_mpiinf3dhp.py \
       --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
              data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
       --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
       --clip_len 13 --epochs 1 --batch_size 8 --train_samples 1000 \
       --output outputs/ray_attention_temporal_mpiinf3dhp_smoke_v1.pth
     ```
   - Result: `Epoch 1: train_loss=0.000525, val_MPJPE=25.24mm (saved)`
   - The new `.npz` loads correctly and the pipeline reports a comparable validation error.

## Files touched

- `experiments/batch_convert_mpiinf3dhp_v1.py` (new)
- `tmp/validate_mpiinf3dhp_npz.py` (new)
- `requirements.txt` — added `scipy>=1.11.0` (required by the existing `webbridge_loader.py` `.mat` reader)
- `data/webbridge/mpi_inf_3dhp/s_03_seq_01_*.npz` (new, 4 files)
- `data/webbridge/mpi_inf_3dhp/s_03_seq_02_*.npz` (new, 4 files)
- `data/webbridge/mpi_inf_3dhp/raw/S3/Seq1/annot.mat`, `camera.calibration` (downloaded)
- `data/webbridge/mpi_inf_3dhp/raw/S3/Seq2/annot.mat`, `camera.calibration` (downloaded)
- `outputs/ray_attention_temporal_mpiinf3dhp_smoke_v1.pth` (smoke-test artifact)

## Dependency note

`scipy` was missing from the `mf` environment even though the existing `webbridge_loader.py` imports `scipy.io`. Installed locally with:

```bash
conda run -n mf pip install scipy
```

and recorded it in `requirements.txt`.

## How to use

Convert only the locally present sequences:

```bash
conda run -n mf python experiments/batch_convert_mpiinf3dhp_v1.py
```

Download and convert specific extra sequences:

```bash
conda run -n mf python experiments/batch_convert_mpiinf3dhp_v1.py \
  --subjects 3 4 --sequences 1 2 --download --yes-download
```

Convert every subject/sequence once the full raw dataset is available:

```bash
conda run -n mf python experiments/batch_convert_mpiinf3dhp_v1.py --subjects 1 2 3 4 5 6 7 8 --sequences 1 2
```

## Validation summary

| File | Shape `points_2d` | Shape `joints_3d` | `camera_t` norms (m) |
|------|-------------------|-------------------|----------------------|
| `s_03_seq_01_v14_multiview_m.npz` | (12489, 14, 28, 2) | (12489, 28, 3) | 0.004, 2.30, 5.25, 4.19, ... |
| `s_03_seq_01_v4_multiview_m.npz` | (12489, 4, 28, 2) | (12489, 28, 3) | 0.004, 2.30, 5.25, 4.19 |
| `s_03_seq_02_v14_multiview_m.npz` | (12283, 14, 28, 2) | (12283, 28, 3) | similar 14-camera layout |
| `s_03_seq_02_v4_multiview_m.npz` | (12283, 4, 28, 2) | (12283, 28, 3) | first 4 cameras of the above |

All files contain the expected keys (`points_2d`, `confidences`, `joints_3d`, `camera_K`, `camera_R`, `camera_t`), and the meters-scale variants have camera positions and joint coordinates in meters.

## Blockers

No blockers. The only dependency gap (`scipy`) was resolved locally and documented above.

## Future work / follow-up

- Download remaining subjects S4–S8 when disk/bandwidth budget allows (~1.6 GB of `annot.mat` files + ~8 GB of generated `.npz`s for all 16 sequences).
- Consider parallelizing the per-sequence conversion loop to reduce wall-clock time.
- Update `configs/train_ray_attention_reproducible.yaml` or training scripts to reference the expanded subject/sequence list.
