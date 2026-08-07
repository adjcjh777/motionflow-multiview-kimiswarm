# MPI-INF-3DHP Test Set (TS1-TS6) Canonical Conversion

The official MPI-INF-3DHP test set contains six sequences (`TS1`..`TS6`).  Each folder includes a single-view image sequence and an `annot_data.mat` file.  The 3D ground truth inside that file is intended for the official evaluation server and is **not** written into the project's canonical `.npz` files.

## Conversion

Run the WSL wrapper:

```bash
bash scripts/convert_mpiinf3dhp_test_set_wsl.sh
```

Or call the Python script directly:

```bash
python experiments/prototypes/swarm_iter18/convert_mpiinf3dhp_test_set.py \
    --test_root data/webbridge/mpi_inf_3dhp/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set \
    --calib data/webbridge/mpi_inf_3dhp/raw/S1/Seq1/camera.calibration \
    --out_dir data/webbridge/mpi_inf_3dhp/test_set \
    --camera_index 0
```

## Output

Per-sequence `.npz` files are written to `data/webbridge/mpi_inf_3dhp/test_set/` (e.g. `TS1_v14_multiview.npz`).

Each file contains the canonical keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `points_2d` | `(T, 14, 17, 2)` | Image-space 2D keypoints |
| `confidences` | `(T, 14, 17)` | Per-frame/view/joint confidence |
| `joints_3d` | `(T, 17, 3)` | **Placeholder** — all zeros because public 3D GT is withheld |
| `camera_K` | `(14, 3, 3)` | Intrinsic calibration matrices |
| `camera_R` | `(14, 3, 3)` | World-to-camera rotations |
| `camera_t` | `(14, 3)` | World-to-camera translation |

Because the released test set provides annotations for only one physical camera view per sequence, the 2D data are placed into the slot selected by `--camera_index` (default 0).  All other 13 views are zero-filled and have confidence 0.  The 14-camera calibration is read from the reference `camera.calibration` file of any MPI-INF-3DHP training sequence.

## Notes

- Joints follow the 17-joint test-set skeleton, not the 28-joint training skeleton.
- `valid_frame` from `annot_data.mat` is mapped to the confidence mask; invalid frames have confidence 0 for the annotated view.
- Do not use `annot3` / `univ_annot3` from the test set for training; they are reserved for official evaluation.
