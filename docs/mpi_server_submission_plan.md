# MPI-INF-3DHP Official Server Submission Plan

This document describes the exact steps to generate a submission for the MPI-INF-3DHP official test server, starting from the MotionFlow-MultiView project layout.

## Scope

- **Test set:** MPI-INF-3DHP official test sequences `TS1`–`TS6` (six sequences, single annotated view each).
- **Skeleton:** 17 joints in the MPI-INF-3DHP "relevant" order (matches Human3.6M CPM order).
- **Official metrics:** MPJPE, PCK@150 mm, AUC, reported sequence-wise and activity-wise.
- **Current project anchor:** `bayesian_tri_v2` ensemble, 8.35 mm MPJPE / 5.29 mm PA-MPJPE on S2/Seq1 (`docs/paper_story_system_v2.md`).

## Important caveats

1. The MPI-INF-3DHP dataset ships with local evaluation MATLAB scripts under `data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/test_util/`. The official leaderboard is updated by **emailing the CSV files produced by those scripts** to the dataset authors; there is no automated web-upload server. Verify the current contact address on the dataset homepage: <https://vcai.mpi-inf.mpg.de/3dhp-dataset/>.
2. The test-set `.npz` converter in this repo stores only the public 2D keypoints and calibration; the 3D ground truth in the downloaded `annot_data.mat` is used only for local validation and must **not** be used for training or model selection.
3. As of 2026-08-11, the MPI detected-2D files required for training a standard-protocol model are still being generated (`docs/cvpr2027_status.md`). This submission plan assumes a trained checkpoint already exists; it does not describe training.

---

## Current status (2026-08-11)

- **A800 disk:** 99% full (~46 GB free on `/mnt/nvme0n1p1`). Large checkpoint dumps or extracted frames should be avoided until cleanup.
- **MPI detected-2D regeneration:** Running on A800 GPU 7 (`generate_mpi_detected_2d.py --detector rtmpose`, PID 2041753). No output `.npz` files yet; a wrapper will re-run the DLT baseline after it finishes.
- **Test-set conversion:** Local `data/webbridge/mpi_inf_3dhp/test_set/TS{1..6}_v14_multiview.npz` already exists. A800 copy not yet converted.
- **Submission packaging:** `scripts/package_mpiinf3dhp_server_submission.py` is now in place.
- **DLT baseline on detected 2D:** Current MediaPipe-generated full set is still ~326–400 mm mean MPJPE due to camera/label misalignment; RTMPose smoke was ~62 mm on 10 frames of S1/Seq1.

---

## 1. Prerequisites

### 1.1 Downloaded data

You need the MPI-INF-3DHP **test set** locally:

```text
data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/
├── TS1/
│   ├── annot_data.mat
│   └── imageSequence/
├── TS2/
...
└── TS6/
```

And at least one reference `camera.calibration` from the training set, e.g.:

```text
data/webbridge/mpi_inf_3dhp/raw/S1/Seq1/camera.calibration
```

### 1.2 Trained checkpoint

A checkpoint that can run on 17-joint, 14-view MPI data, for example:

```text
outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth
outputs/bayesian_tri_v2_aug_mpiinf3dhp.pth
```

### 1.3 Python/MATLAB environment

- Python: project virtualenv with `torch`, `numpy`, `scipy`, `h5py`.
- MATLAB (or Octave): to run the official `mpii_test_predictions.m` / `mpii_evaluate_errors.m` scripts.

---

## 2. Convert the test set to canonical `.npz`

The inference scripts in this repo consume the canonical multi-view `.npz` format.

```bash
bash scripts/convert_mpiinf3dhp_test_set_wsl.sh
```

This writes:

```text
data/webbridge/mpi_inf_3dhp/test_set/
├── TS1_v14_multiview.npz
├── TS2_v14_multiview.npz
├── TS3_v14_multiview.npz
├── TS4_v14_multiview.npz
├── TS5_v14_multiview.npz
└── TS6_v14_multiview.npz
```

Each file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `points_2d` | `(T, 14, 17, 2)` | Image-space 2D keypoints; only the single annotated view is non-zero. |
| `confidences` | `(T, 14, 17)` | Per-frame/view/joint confidence. |
| `joints_3d` | `(T, 17, 3)` | Placeholder zeros; public 3D GT is withheld. |
| `camera_K` | `(14, 3, 3)` | Intrinsic calibration matrices. |
| `camera_R` | `(14, 3, 3)` | World-to-camera rotations. |
| `camera_t` | `(14, 3)` | World-to-camera translations. |

**Note:** Because the test set provides annotations for only one physical camera view per sequence, the 2D data are placed into view slot 0 (`--camera_index 0`) and the other 13 views are zero-filled. The model must therefore be evaluated in a **single-view fallback / variable-view** mode unless you run a real 2D detector on the raw test images.

---

## 3. Run inference on TS1–TS6

### 3.1 Using the OmniMultiViewFusionV2 inference script

```bash
python experiments/infer_mpiinf3dhp_test_set_omniview_v2.py \
    --checkpoint outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth \
    --test_set_dir data/webbridge/mpi_inf_3dhp/test_set \
    --out_npz outputs/omniview_fusion_v2_test_set_predictions.npz \
    --clip_len 13 \
    --batch_size 8 \
    --stride 1
```

The output `outputs/omniview_fusion_v2_test_set_predictions.npz` contains one array per sequence, keyed by the file stem:

```python
import numpy as np
data = np.load("outputs/omniview_fusion_v2_test_set_predictions.npz")
print(data.files)          # ['TS1_v14_multiview', 'TS2_v14_multiview', ...]
print(data['TS1_v14_multiview'].shape)  # (T, 17, 3)
```

Units are metres (the model is trained on metre-scale MPI data). The official evaluation expects millimetres.

### 3.2 Using a different model

If your model is not `OmniMultiViewFusionV2`, create an inference script that:

1. Loads each `data/webbridge/mpi_inf_3dhp/test_set/TS{i}_v14_multiview.npz`.
2. Produces per-frame 3D pose predictions of shape `(T, 17, 3)` in **metres**.
3. Saves them to a single `.npz` with keys `TS1_v14_multiview` … `TS6_v14_multiview`.

---

## 4. Convert project predictions to official submission format

### 4.1 Required MATLAB variables

The official test-util scripts expect a `.mat` file (v7.3 or older, not HDF5) containing:

```matlab
sequencewise_per_joint_error  % 6x1 cell array
sequencewise_activity_labels  % 6x1 cell array
```

where for test sequence `i`:

- `sequencewise_per_joint_error{i}` is a `(17, 1, N_i)` double array of **per-joint Euclidean errors in millimetres** for each valid frame.
- `sequencewise_activity_labels{i}` is a `(N_i, 1)` int array of activity labels for each valid frame (copied from `activity_annotation(valid_frame==1)`).

`N_i` is the number of valid frames in `TS{i}`.

### 4.2 Reference: official MATLAB evaluation snippet

From `data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/test_util/mpii_test_predictions.m`:

```matlab
[~,o1,o2,relevant_labels] = mpii_get_joints('relevant');
for i = 1:length(test_subject_id)
   dat = load([data_base_path int2str(test_subject_id(i)) filesep 'annot_data.mat']);
   num_test_points = sum(dat.valid_frame(:));
   per_joint_error = zeros(17,1,num_test_points);
   pje_idx = 1;
   sequencewise_activity_labels{i} = dat.activity_annotation(dat.valid_frame == 1);

   for j = 1:length(dat.valid_frame)
       if dat.valid_frame(j)
           % pred_p must be 3x17, mm, root-relative to pelvis (joint 15 in relevant order)
           P = dat.univ_annot3(:,:,:,j) - repmat(dat.univ_annot3(:,15,:,j), 1, 17);
           pred_p = P;  % REPLACE WITH MODEL PREDICTION
           error_p = sqrt(sum((pred_p - P).^2, 1));
           per_joint_error(:,:,pje_idx) = error_p(:);
           pje_idx = pje_idx + 1;
       end
   end
   sequencewise_per_joint_error{i} = per_joint_error;
end

save('mpii_3dhp_prediction.mat', 'sequencewise_per_joint_error', 'sequencewise_activity_labels');
[seq_table, activity_table] = mpii_evaluate_errors(sequencewise_per_joint_error, sequencewise_activity_labels);
```

### 4.3 Python helper to build the submission `.mat`

Create `scripts/package_mpiinf3dhp_server_submission.py`:

```python
#!/usr/bin/env python3
"""Package per-sequence 3D predictions into the official MPI-INF-3DHP .mat format.

Input
-----
npz_path : str
    Path to the project's test-set predictions .npz, with keys
    "TS1_v14_multiview" .. "TS6_v14_multiview", each shape (T, 17, 3) in metres.

test_root : str
    Path to the downloaded MPI-INF-3DHP test set root (contains TS1..TS6).

output_mat : str
    Destination .mat file (v7.2 / older, not v7.3).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


PELVIS_RELEVANT_IDX = 14  # 0-based pelvis in the 17-joint "relevant" skeleton


def load_test_annotations(mat_path: Path) -> tuple:
    """Return (univ_annot3, valid_frame, activity_annotation)."""
    try:
        import h5py

        with h5py.File(mat_path, "r") as f:
            # (3, 17, 1, T) in mm
            univ_annot3 = f["univ_annot3"][:]
            valid_frame = f["valid_frame"][:, 0].astype(bool)
            activity_annotation = f["activity_annotation"][:, 0].astype(int)
    except Exception:
        from scipy.io import loadmat

        data = loadmat(mat_path)
        univ_annot3 = data["univ_annot3"]
        valid_frame = data["valid_frame"].astype(bool).ravel()
        activity_annotation = data["activity_annotation"].ravel().astype(int)
    return univ_annot3, valid_frame, activity_annotation


def package_submission(npz_path: str, test_root: str, output_mat: str) -> None:
    preds = np.load(npz_path)
    test_root = Path(test_root)

    per_joint_error_cells = []
    activity_label_cells = []

    for i in range(1, 7):
        key = f"TS{i}_v14_multiview"
        pred_m = preds[key]  # (T, 17, 3) in metres
        pred_mm = pred_m * 1000.0  # convert to mm

        mat_path = test_root / f"TS{i}" / "annot_data.mat"
        univ_annot3, valid_frame, activity_annotation = load_test_annotations(mat_path)

        # Ensure correct shape (T, 17, 3) in mm
        if univ_annot3.shape[0] == 3:
            # shape is (3, 17, 1, T)
            gt = univ_annot3[:, :, 0, :].transpose(2, 1, 0)
        else:
            gt = univ_annot3

        # Root-relative: subtract pelvis joint
        root = gt[:, PELVIS_RELEVANT_IDX, :][:, None, :]  # (T, 1, 3)
        gt_rel = gt - root
        pred_rel = pred_mm - root

        # Per-joint Euclidean error (mm), shape (T, 17)
        per_joint_err = np.linalg.norm(pred_rel - gt_rel, axis=-1)

        # Keep only valid frames
        per_joint_err = per_joint_err[valid_frame]  # (N_i, 17)
        activity_labels = activity_annotation[valid_frame]  # (N_i,)

        # MATLAB cell entry must be (17, 1, N_i) double
        per_joint_error_cells.append(per_joint_err.T[:, None, :])
        activity_label_cells.append(activity_labels[:, None])

    sio.savemat(
        output_mat,
        {
            "sequencewise_per_joint_error": np.array(per_joint_error_cells, dtype=object),
            "sequencewise_activity_labels": np.array(activity_label_cells, dtype=object),
        },
        do_compression=True,
    )
    print(f"Saved submission file: {output_mat}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package MPI-INF-3DHP server submission.")
    parser.add_argument("--npz", required=True, help="Path to test-set predictions .npz.")
    parser.add_argument("--test_root", required=True, help="Path to MPI test set root.")
    parser.add_argument("--output", default="outputs/mpii_3dhp_prediction.mat")
    args = parser.parse_args()

    package_submission(args.npz, args.test_root, args.output)
```

Run it:

```bash
python scripts/package_mpiinf3dhp_server_submission.py \
    --npz outputs/omniview_fusion_v2_test_set_predictions.npz \
    --test_root data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set \
    --output outputs/mpii_3dhp_prediction.mat
```

### 4.4 Alternative: produce the submission entirely in MATLAB

If your inference pipeline can write per-sequence `.npy` or `.mat` predictions, you can replace the Python packaging step with a MATLAB script that mirrors `mpii_test_predictions.m`. This is the most direct path because it uses the exact reference evaluation code.

---

## 5. Validate the submission locally

### 5.1 Run the official evaluation scripts

In MATLAB, with the test-util directory on the path:

```matlab
addpath('data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/test_util');
load('outputs/mpii_3dhp_prediction.mat');
[seq_table, activity_table] = mpii_evaluate_errors(sequencewise_per_joint_error, sequencewise_activity_labels);
```

This produces:

- Sequence-wise MPJPE, PCK@150 mm, AUC for each of TS1–TS6.
- Activity-wise MPJPE, PCK@150 mm, AUC for each of the 7 activities.
- Overall MPJPE, PCK@150 mm, AUC.

### 5.2 Sanity checks

- `sequencewise_per_joint_error{i}` should be `(17, 1, N_i)` double, all non-negative.
- `sequencewise_activity_labels{i}` should be `(N_i, 1)` int in range `[1, 7]`.
- Mean MPJPE should be comparable to the S2/Seq1 validation number (anchor: ~8–9 mm for the best ensemble).
- If errors are all zero, you leaked the test-set 3D ground truth into the predictions.

---

## 6. Submit to the official server

### 6.1 Generate the official CSV report

After running `mpii_evaluate_errors`, the reference script writes:

```text
<net_base>/<net_name>/mpii_3dhp_evaluation_<iter>_sequencewise.csv
<net_base>/<net_name>/mpii_3dhp_evaluation_<iter>_activitywise.csv
```

These are the files the MPI-INF-3DHP maintainers use to update the leaderboard.

### 6.2 Send the results

Per the dataset homepage (<https://vcai.mpi-inf.mpg.de/3dhp-dataset/>):

> If you would like the updated results from your method to be reflected here, please send us the CSV files generated from the evaluation script, along with information regarding what test time augmentation was employed, as well as a link to your project page.

Typical submission email contents:

- **Subject:** `MPI-INF-3DHP leaderboard submission — <MethodName>`
- **Body:**
  - Method name and paper/project link.
  - Whether test-time augmentation was used.
  - Attached `*_sequencewise.csv`, `*_activitywise.csv`, and optionally the raw `mpii_3dhp_prediction.mat`.
- **Recipient:** check the current dataset homepage for the active contact email (historically the first author of the 3DV 2017 paper).

**Note:** There is no known automated upload portal. The submission is manual email. If an online submission server exists, it will be linked from the dataset homepage and the plan should be updated accordingly.

---

## 7. Known blockers and dependencies

| # | Blocker | Status | Impact |
|---|---------|--------|--------|
| 1 | **MPI detected-2D training data** — real detected 2D for all train sequences must be generated before a standard-protocol model can be selected for the server. RTMPose generation is running on A800 GPU 7 (PID 2041753, started 2026-08-11 10:00). No `.npz` outputs yet; a wrapper will re-run the DLT baseline after generation finishes. | In progress | High — training on GT 2D is not comparable to literature. |
| 2 | **Single-view test input** — the test set converter only populates view slot 0; models that require ≥2 views need real 2D detections on the raw test images or a single-view inference path. | Open | High for multi-view models. |
| 3 | **Checkpoint readiness** — the best existing checkpoints (Bayesian Tri v2 ensemble) are trained/validated on GT 2D, not detected 2D. Their server performance is unknown under the detected-2D protocol. | Open | High for leaderboard relevance. |
| 4 | **Server contact verification** — the official submission is email-based; the current contact address must be verified on the dataset homepage before sending. | Open | Low (informational). |
| 5 | **Packaging script** — `scripts/package_mpiinf3dhp_server_submission.py` has been created from the template in §4.3. | Done | Low — ready for use once predictions exist. |

---

## 8. Quick checklist

- [ ] Test set downloaded to `data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/TS*`.  
- [ ] Reference `camera.calibration` available (e.g. `data/webbridge/mpi_inf_3dhp/raw/S1/Seq1/camera.calibration`).  
- [ ] `bash scripts/convert_mpiinf3dhp_test_set_wsl.sh` run successfully and produced `data/webbridge/mpi_inf_3dhp/test_set/TS*_v14_multiview.npz`.  
- [ ] Inference script run and `outputs/omniview_fusion_v2_test_set_predictions.npz` produced.  
- [x] `scripts/package_mpiinf3dhp_server_submission.py` created. Run it to produce `outputs/mpii_3dhp_prediction.mat` once inference outputs are ready.  
- [ ] Local MATLAB validation with `mpii_evaluate_errors` executed and numbers are reasonable.  
- [ ] CSV report generated.  
- [ ] Submission email sent to the dataset maintainers with CSVs and method description.  

---

## 9. References

- `docs/mpiinf3dhp_test_set_conversion.md` — converting TS1–TS6 to canonical `.npz`.  
- `experiments/infer_mpiinf3dhp_test_set_omniview_v2.py` — inference script.  
- `data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/test_util/` — official MATLAB evaluation scripts.  
- `docs/mpi_detected_2d_protocol.md` — standard detected-2D protocol.  
- Dataset homepage: <https://vcai.mpi-inf.mpg.de/3dhp-dataset/>  
