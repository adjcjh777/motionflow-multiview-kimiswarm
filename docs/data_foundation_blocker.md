# Data Foundation Blocker (2026-08-10)

> TL;DR: H36M labels in `data/h36m_hf/*_multiview.npz` are **exactly the unweighted DLT triangulation of the input 2D keypoints**, making the validation MPJPE a meaningless measure of pose accuracy. The current model-development loop (v25–v79) has been optimising a circular metric. All training/polling is paused until the data foundation is fixed.

## Diagnostic evidence

Run the diagnostic script:

```bash
python scripts/diagnose_circular_labels.py data/h36m_hf/s_01_act_02_multiview.npz
```

Observed output (2026-08-10):

```text
  direct MJE (no root align): 0.0000 mm
  root-aligned MPJPE:       0.0000 mm
  max per-joint error:      0.0000 mm
```

The stored 3D labels are identical to `triangulate_dlt(points_2d, cameras)` up to numerical precision.

## Where the circular label comes from

`motionflow_mv/data/webbridge_loader.py:182`:

```python
j3d = _triangulate_joints(p2d, cameras)
```

The converter triangulates the input 2D detections and stores the result as the ground-truth 3D.  Any architecture that contains an explicit (or easily learned) weighted DLT layer can trivially reproduce the label, which explains why v25 (which has a weighted DLT fusion layer) dominates the leaderboard and why modules that perturb the DLT output (physical calibration, graph refinement, etc.) appear to hurt.

## Impact on previous experiments

- v25 "17.17 mm" is not a meaningful pose-estimation result; it is the model learning to copy the label-generation function.
- v31–v43 graph/iterative modules look worse because they move the output away from the exact DLT label, not because they are worse pose estimators.
- v53–v79 physical-space calibration modules were dropped because they intentionally modify the DLT output, which the circular metric penalises.
- Tiny/medium smoke decisions (50–200 samples) were made on a noise-laden circular validation, so keep/drop decisions are unreliable.

## What needs to change

1. **Obtain real H36M 3D GT.** The raw pickle `data/h36m_hf/h36m_sh_conf_cam_source_final.pkl.zip` only contains `joint3d_image` (per-camera image-space coordinates `(u, v, z)`). This is **not** true world-space 3D and cannot be converted to a consistent world coordinate across cameras. A real H36M 3D source (e.g., the original Human3.6M release with mocap world coordinates) is required.
2. **Adopt a fallback dataset if true H36M is unavailable.** MPI-INF-3DHP has real 3D (`univ_annot3`), but the standard protocol uses detected 2D, not GT 2D. Shelf/Campus has real 3D but is small and uses GT 2D projections. The fastest reliable baseline may come from synthetic data with known 3D.
3. **Regenerate canonical `.npz` files.** All `data/h36m_hf/*_multiview.npz` need to be re-generated with the corrected labels once a true 3D source is found.
4. **Re-evaluate baselines.** Re-run DLT / v25 / v46 / v57 on the corrected labels under a standard protocol (e.g., H36M S1,S5,S6,S7,S8 train → S9,S11 test, or the MPI-INF-3DHP standard split).
5. **Adopt a realistic target.** On a correct protocol, numbers will likely land in the 15–30 mm range, comparable to multi-view H36M SOTA. The paper contribution should pivot from “precision record” to **sparse-view / cross-domain robustness**, which the existing v46-SVG, v52-UWT, and `eval_variable_views.py` `MPJPE@k` infrastructure already support.
6. **Clean up related work.** Fix the fabricated citations noted in the swarm reports and add real SOTA baselines (Iskakov, VoxelPose, etc.).

## Local data audit

| Dataset | True 3D? | Notes |
|---------|----------|-------|
| H36M (pkl / `data/h36m_hf`) | **No** | `joint3d_image` is image-space; labels are DLT(p2d, cameras) |
| `data/webbridge/h36m` | **No** | Same circular `.npz` |
| `data/webbridge/h36m_meters` | **No** | Same circular `.npz` in metres |
| MPI-INF-3DHP | Real 3D | Not fully downloaded locally; standard protocol uses detected 2D |
| Shelf/Campus | Real 3D | Small; 2D inputs are GT projections (still circular but with noise) |
| `data/h36m_mirror/SemGCN/h36m.zip` | **No** | Only contains 2D StackedHourglass poses and `cameras.h5` |
| A800-D `/mnt/nvme0n1/zhangzy/projects` | No true H36M 3D found | Read-only; no original `.mat`/`.cdf` files present |

## Timeline

- **ICRA 2027 (~2026-09):** too tight; do not target.
- **CVPR 2027 (~2026-11):** achievable if the data foundation is fixed within the next ~1–2 weeks and baselines are re-run shortly after.

## Immediate next steps

- [x] Stop all training/polling, including A800 tmux training sessions.
- [x] Write and run the DLT diagnostic.
- [x] Confirm `joint3d_image` is not usable as true world GT.
- [ ] Repair the data foundation for CVPR 2027.
  - [ ] Decide whether to download/obtain original H36M 3D or pivot to MPI/Shelf/synthetic.
  - [ ] Regenerate affected `.npz` files with corrected labels.
- [ ] Re-run the diagnostic and confirm MPJPE between DLT and corrected labels is non-zero.
- [ ] Re-run v25 baseline on the corrected protocol.
