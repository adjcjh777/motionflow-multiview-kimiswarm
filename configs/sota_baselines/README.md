# configs/sota_baselines

SOTA baseline configs and launchers for the corrected H36M true-GT v2 protocol
(`data/h36m_true_gt_v2/`, S1,5,6,7,8 → S9/S11).

## Files

- `voxelpose_h36m_true_gt_v2.yaml` — VoxelPose runtime config passed to
  `run/train_3d.py` in the upstream Microsoft VoxelPose repo.
- `voxelpose_h36m_true_gt_v2_prep.yaml` — Prep config consumed by
  `scripts/sota_baselines/convert_to_voxelpose_format.py`.
- `run_voxelpose_h36m_true_gt_v2.sh` — A800 launcher. It exports the v2 data,
  converts to VoxelPose format, applies the H36M adapter overlay, waits for GPU
  6 or 7 to be free, and launches training.
- `mvpose_h36m_true_gt_v2.yaml` — MVPose config consumed by the converter and
  by the geometry-only adapter.
- `run_mvpose_h36m_true_gt_v2.sh` — Launcher that exports the v2 data, converts
  to MVPose format, runs the geometry-only triangulation adapter on the test
  split, and evaluates.

## Usage

```bash
# A800 VoxelPose launch
bash configs/sota_baselines/run_voxelpose_h36m_true_gt_v2.sh

# MVPose launch (CPU, does not touch project GPUs)
bash configs/sota_baselines/run_mvpose_h36m_true_gt_v2.sh
```

All outputs are written to `tmp/sota_baselines/*_v2/` and
`outputs/sota_baselines/*_v2.*` so they do not overwrite the original v1
baseline artifacts.
