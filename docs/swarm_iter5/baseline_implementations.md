# Baseline Implementations

## What was produced

`experiments/baselines.py` — a self-contained, NumPy-only script that benchmarks
four classic triangulation baselines on every prepared `.npz` dataset:

1. **DLT** — confidence-weighted Direct Linear Transform.
2. **RANSAC-DLT** — minimal-view-set sampling with inlier scoring; falls back to
   plain DLT when there are too few views for meaningful sampling.
3. **Robust** — iteratively re-weighted DLT with Huber-like residual weighting.
4. **Temporal** — DLT per frame followed by a centered moving-average smoother.

The script discovers all `data/**/*.npz` files (or accepts explicit paths),
expects the standard keys `points_2d`, `confidences`, `joints_3d`, `camera_K`,
`camera_R`, `camera_t`, and reports MPJPE, PA-MPJPE and mean reprojection error
in pixels.  It writes an optional JSON results table.

## Quick verification run

```bash
# Windows conda environment used in this repo
export PATH="/d/anaconda3/envs/jz_py310/Library/bin:$PATH"
export KMP_DUPLICATE_LIB_OK=TRUE

python experiments/baselines.py --max_frames 200 --output outputs/baseline_results.json
```

Results on the prepared datasets (first 200 frames, native dataset units):

| dataset                                  | method      | MPJPE  | PA-MPJPE | Reproj(px) |
|------------------------------------------|-------------|--------|----------|------------|
| s_01_act_02_multiview.npz              | dlt         | 1.93   | 3.35     | 2.01       |
| s_01_act_02_multiview.npz              | ransac_dlt  | 1.93   | 3.35     | 2.01       |
| s_01_act_02_multiview.npz              | robust      | 3.56   | 6.91     | 1.96       |
| s_01_act_02_multiview.npz              | temporal    | 5.03   | 6.22     | 2.09       |
| s_01_acts_02_..._16_multiview.npz      | dlt         | 1.93   | 3.35     | 2.01       |
| s_09_acts_02_multiview.npz             | dlt         | 513.27 | 622.34   | 225.99     |
| Shelf_Seq1/pseudogt.npz                | dlt         | 290.17 | 54.07    | 786.25     |

H36M subject 1 is internally consistent: DLT re-creates the per-frame 3D GT
with ~1.9 mm MPJPE, matching the DLT baseline reported in `docs/design_v3.md`.

## Key findings

* **DLT is the strong geometric baseline** on clean, internally consistent
  data.  It should be the reference against which `ray_attention_v3` is judged.
* **RANSAC-DLT now samples 3-view subsets even with only 4 total views.** After
  removing the `V <= 4` fallback in `experiments/baselines.py`, the canonical
  unweighted RANSAC-DLT gives **~28.4 mm** on the H36M true-GT test set (S9/S11,
  4 views).  A confidence-weighted random-subset variant gives **~26.5 mm**, the
  closest reproducible result to the historical 26.61 mm reference (now superseded).  See
  `docs/results_true_gt_h36m.md` for the full discrepancy note.
* **Robust Huber weighting stays close to DLT** on clean data (~3.6 mm) but can
  degrade when the initial DLT estimate is already poor (see Shelf pseudogt and
  H36M subject 9).
* **Temporal smoothing trades per-frame accuracy for smoothness**, which is
  expected when the ground truth is itself per-frame triangulation rather than
  an independently captured smooth reference.
* **Shelf pseudogt and H36M s_09 show large absolute 3D errors** because the
  stored 3D ground truth and the bundled camera parameters are not on the same
  scale/coordinate frame.  The script still runs, but those datasets are not yet
  suitable for benchmarking geometric accuracy without additional alignment.

## Environment notes

The script is PyTorch-free, but NumPy 2.x on the Windows conda environment needs
`<env>/Library/bin` in `PATH` at runtime for the BLAS/LAPACK DLLs.  The
`KMP_DUPLICATE_LIB_OK=TRUE` workaround is also needed to avoid the libomp vs.
libiomp5md conflict in this environment.

## Next steps

* Use `experiments/baselines.py` to generate the DLT reference for every
  prepared H36M/Shelf split before training `ray_attention_v3`.
* Add optional synthetic 2D noise / outlier injection to the script so that
  RANSAC and robust baselines can be compared under controlled corruption.
* Investigate the camera/3D mismatch in `s_09_acts_02_multiview.npz` and in the
  Shelf pseudogt; likely the extrinsics need a per-subject or per-sequence
  scale/alignment check.
