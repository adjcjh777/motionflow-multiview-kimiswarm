# MPI-INF-3DHP Official Test-Server Submission Checklist

> Operational checklist derived from `docs/mpi_official_submission_protocol.md`,
> `docs/mpi_server_submission_plan.md` and `docs/mpiinf3dhp_test_set_conversion.md`.
> Last updated: 2026-08-13

## 1. Official evaluation flow

1. The MPI-INF-3DHP test set contains six sequences `TS1`–`TS6`.
2. Ground-truth 3D is **not** public; evaluation is performed by the maintainers.
3. The server/leaderboard accepts predictions as per-sequence `.mat` files.
4. Each `.mat` must contain a variable `joint_3d_image` of shape `(T, 17, 3)` in **millimetres**, root-relative to the pelvis (joint 14, 0-based).
5. Submit either by email (historical) or via the official portal at <https://mpi-inf-3dhp.is.tuebingen.mpg.de/> (verify current policy on the dataset homepage).

## 2. Required files

| File / directory | Purpose | Where it comes from |
|------------------|---------|---------------------|
| `data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/TS{i}/annot_data.mat` (i=1..6) | Official test set with GT withheld | Download from MPI-INF-3DHP homepage |
| `data/webbridge/mpi_inf_3dhp/test_set/TS{i}_v14_multiview.npz` | Canonical multi-view `.npz` consumed by project inference scripts | `scripts/convert_mpiinf3dhp_test_set_wsl.sh` |
| Trained checkpoint `.pth` or a predictions `.npz` | Source of 3D pose predictions | Training or inference run |
| `scripts/prepare_mpi_submission.py` | Full v5 checkpoint → submission package | Repository |
| `scripts/prepare_mpi_official_submission.py` | Generic `.npz` / DLT / v2 checkpoint → submission package | Repository |
| `scripts/convert_npz_to_mpi_submission.py` | Lightweight project `.npz` → official per-sequence `.mat` + zip | This checklist |

## 3. Step-by-step submission

1. **Prepare data**
   - Download the official test set and place it under `data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/`.
   - Run `bash scripts/convert_mpiinf3dhp_test_set_wsl.sh` to create the canonical `.npz` files.

2. **Generate predictions**
   - From a v5 checkpoint:
     ```bash
     bash scripts/run_mpi_submission_a800.sh \
         --config configs/ablations/v25_true_gt_v2_medium_a800.yaml \
         --checkpoint outputs/ablations/v25_true_gt_v2_medium_a800.pth \
         --method_name v25_true_gt_v2_medium
     ```
   - From an existing project-format `.npz`:
     ```bash
     python scripts/convert_npz_to_mpi_submission.py \
         --npz outputs/my_mpi_predictions.npz \
         --test_root data/webbridge/mpi_inf_3dhp/raw_test/mpi_inf_3dhp_test_set/ \
         --output outputs/my_mpi_submission.zip
     ```

3. **Verify the package**
   ```bash
   python scripts/verify_mpi_submission_format.py --zip outputs/my_mpi_submission.zip
   ```

4. **Submit**
   - Email the CSVs produced by the official MATLAB evaluation scripts, or upload via the official portal (verify current mechanism on the dataset homepage).

## 4. Current `outputs/` status

- No project-format MPI test-set prediction `.npz` is present yet (checked 2026-08-13).
- Existing `outputs/*mpi*.json` files contain evaluation **metrics**, not submittable 3D predictions.
- Therefore, predictions must first be generated before any submission package can be built.

## 5. Common pitfalls

- **Single-view test set:** the released test set only has one annotated camera view. Multi-view models must either use detected 2D on the raw images or run in single-view fallback mode.
- **Units:** predictions inside the `.npz` are stored in **metres**; the official `.mat` requires **millimetres**.
- **Root:** the submission must be root-relative to the pelvis (joint 14). The scripts handle this automatically.
- **Skeleton:** use the 17-joint "relevant" skeleton, not the 28-joint training skeleton.
