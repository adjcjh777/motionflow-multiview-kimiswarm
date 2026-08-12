# MPI-INF-3DHP Official Test-Server Submission Protocol

This document describes how to prepare a submission package for the MPI-INF-3DHP official test server using this repository. It is a focused operational guide; for background on how the test set is converted to canonical `.npz` files, see `docs/mpiinf3dhp_test_set_conversion.md`, and for the original detailed plan, see `docs/mpi_server_submission_plan.md`.

## Important: do not upload from this script

The script `scripts/prepare_mpi_official_submission.py` only **formats** and **packages** predictions. It does **not** upload anything. The actual submission is a separate manual step (email the evaluation CSVs or the zip, depending on the current server policy). See [Submitting to the server](#submitting-to-the-server) below.

## What the official server expects

The MPI-INF-3DHP test set consists of six sequences (`TS1`..`TS6`). Ground-truth 3D pose is not public; evaluation happens on the maintainers' side. The official evaluation code (the test-util scripts that ship with the dataset) accepts predictions in one of two ways:

1. **Per-sequence 3D predictions** — a `.zip` containing one `.mat` file per sequence. Each `.mat` should contain the predicted 3D joints, typically as a variable named `joint_3d_image` of shape `(T, 17, 3)` in millimetres, root-relative to the pelvis (joint 14 in the 17-joint "relevant" skeleton). The maintainer then runs the official evaluation on these coordinates.
2. **Pre-computed per-joint errors** — a single `mpii_3dhp_prediction.mat` with `sequencewise_per_joint_error` and `sequencewise_activity_labels`, exactly as the test-util scripts `mpii_test_predictions.m` / `mpii_evaluate_errors.m` expect. This is useful for local validation when you have the downloaded `annot_data.mat` files.

`prepare_mpi_official_submission.py` produces both: the per-sequence prediction `.mat` files are always included, and the local-evaluation `.mat` is included whenever the official test-set root (with `TS{i}/annot_data.mat`) is supplied.

## Submission format produced by the script

```text
mpi_official_submission.zip
├── TS1.mat
├── TS2.mat
├── TS3.mat
├── TS4.mat
├── TS5.mat
├── TS6.mat
├── mpii_3dhp_prediction.mat   # optional, if --test_root supplied
├── manifest.json
└── README.txt
```

Each `TS{i}.mat` contains:

| Variable | Shape | Units | Description |
|---|---:|---|---|
| `joint_3d_image` | `(T, 17, 3)` | mm | Predictions root-relative to the pelvis (joint 14). This is the variable the server evaluation code normally expects. |
| `joint_3d_image_abs` | `(T, 17, 3)` | mm | Absolute (camera/world) predictions, kept for debugging. |
| `pelvis_index` | scalar | — | Index of the pelvis (14, 0-based). |
| `unit` | string | — | `"mm"`. |

`mpii_3dhp_prediction.mat` contains the local-evaluation cell arrays described in `docs/mpi_server_submission_plan.md` and produced by `scripts/package_mpiinf3dhp_server_submission.py`.

## Prerequisites

1. **Downloaded test set.** The official MPI-INF-3DHP test set must be present locally, e.g.:
   ```text
   data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/
   ├── TS1/annot_data.mat
   ├── TS2/annot_data.mat
   ...
   └── TS6/annot_data.mat
   ```
2. **Canonical test-set `.npz` files.** Generate them with:
   ```bash
   bash scripts/convert_mpiinf3dhp_test_set_wsl.sh
   ```
   This writes `data/webbridge/mpi_inf_3dhp/test_set/TS{i}_v14_multiview.npz`.
3. **Predictions.** Either:
   - A trained `OmniMultiViewFusionV2` checkpoint, or
   - An existing predictions `.npz` in the format described above, or
   - Multi-view detected 2D for the test set so that DLT triangulation has at least two views per frame (the official release has only one annotated view).

## How to run the script

### 1. From a trained checkpoint

```bash
python scripts/prepare_mpi_official_submission.py \
    --mode checkpoint \
    --checkpoint outputs/omniview_fusion_v2_mpiinf3dhp.pth \
    --test_set_dir data/webbridge/mpi_inf_3dhp/test_set \
    --test_root data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set \
    --output_zip outputs/mpi_official_submission.zip \
    --method_name "MotionFlow-MultiView" \
    --device cuda
```

This runs `experiments/infer_mpiinf3dhp_test_set_omniview_v2.py` and then packages the result.

### 2. From an existing predictions .npz

```bash
python scripts/prepare_mpi_official_submission.py \
    --mode predictions \
    --npz outputs/omniview_fusion_v2_test_set_predictions.npz \
    --test_root data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set \
    --output_zip outputs/mpi_official_submission.zip
```

### 3. From DLT triangulation (requires multi-view detected 2D)

```bash
python scripts/prepare_mpi_official_submission.py \
    --mode dlt \
    --test_set_dir data/webbridge/mpi_inf_3dhp/test_set \
    --test_root data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set \
    --output_zip outputs/mpi_official_submission_dlt.zip
```

### Common arguments

| Argument | Default | Meaning |
|---|---|---|
| `--mode` | required | `predictions`, `dlt`, or `checkpoint` |
| `--npz` | — | Path to predictions `.npz` (mode `predictions`) |
| `--checkpoint` | — | Model checkpoint path (mode `checkpoint`) |
| `--test_set_dir` | `data/webbridge/mpi_inf_3dhp/test_set` | Directory with canonical `TS{i}_v14_multiview.npz` files |
| `--test_root` | `None` | Official test set root (enables `mpii_3dhp_prediction.mat`) |
| `--output_zip` | `outputs/mpi_official_submission.zip` | Output zip path |
| `--method_name` | `MotionFlow-MultiView` | Method name for the manifest |
| `--device` | `cpu` | PyTorch device for inference / DLT |

## Local sanity checks

After producing the zip, verify the contents without uploading:

```python
import zipfile
import scipy.io

with zipfile.ZipFile("outputs/mpi_official_submission.zip") as zf:
    print(zf.namelist())

# Load one sequence and check the variable.
mat = scipy.io.loadmat("outputs/mpi_official_submission_staging/TS1.mat")
print(mat["joint_3d_image"].shape)  # (T, 17, 3)
```

The pelvis joint (index 14) of `joint_3d_image` should be numerically zero after root-centering.

## Submitting to the server

- **Dataset homepage:** <https://vcai.mpi-inf.mpg.de/3dhp-dataset/> (also <http://gvv.mpi-inf.mpg.de/3dhp-dataset/>)
- **Official test set:** <http://gvv.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp_test_set.zip>
- **Submission mechanism:** As of the last update on the dataset homepage, submissions are handled by emailing the CSV files generated by the official test-util scripts to the maintainers. The script therefore does not attempt an automated upload.
- **Expected turnaround:** Typically a few business days to a couple of weeks, depending on the maintainers' availability. Plan accordingly before any deadline.

If a future version of the server adds an automated upload portal, update this document and the script to point to it.

## Pre-submission checklist

Before sending anything to the maintainers:

- [ ] Test set downloaded and converted to canonical `.npz` files.
- [ ] Predictions produced for **all six** sequences (`TS1`..`TS6`).
- [ ] Each `TS{i}.mat` contains `joint_3d_image` with shape `(T, 17, 3)` and units of **millimetres**.
- [ ] Predictions are root-relative to the pelvis (joint 14) — the pelvis channel in `joint_3d_image` should be zero.
- [ ] Local-evaluation `.mat` produced and validated with `mpii_evaluate_errors` (requires the official test-util scripts and the downloaded `annot_data.mat`).
- [ ] `manifest.json` inside the zip correctly lists the method name, timestamp, and per-sequence shapes.
- [ ] No ground-truth 3D from the test set leaked into the model or predictions.
- [ ] The zip has been uploaded **only to the maintainers** (or a server portal) and not committed to version control.

## Notes and caveats

- The official test-set release only contains a single annotated camera view per sequence. Running the script in `--mode dlt` directly on the released `.npz` files will fail with a "less than two active views" error because the remaining 13 views are zero-filled. You must either run a multi-view 2D detector on the raw test images or use the single-view inference mode (`--mode checkpoint`).
- The local-evaluation `mpii_3dhp_prediction.mat` uses the stored test-set ground truth and must therefore never be used for model selection; it is only for sanity-checking the submission package before emailing it.
- Keep the submission zip under the size limits requested by the maintainers. If the zip is too large, consider compressing the `.mat` files with `do_compression=True` (the default used by the script) or splitting by sequence.

## References

- `docs/mpi_server_submission_plan.md` — original full submission plan.
- `docs/mpiinf3dhp_test_set_conversion.md` — converting TS1–TS6 to canonical `.npz`.
- `scripts/package_mpiinf3dhp_server_submission.py` — helper that builds the local-evaluation `.mat`.
- `experiments/infer_mpiinf3dhp_test_set_omniview_v2.py` — inference script for OmniMultiViewFusionV2 checkpoints.
