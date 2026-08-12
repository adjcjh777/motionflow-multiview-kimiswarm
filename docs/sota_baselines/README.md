# SOTA Baselines on H36M True-GT

Results for third-party SOTA baselines on the corrected, non-circular H36M
true-GT protocol (S1,5,6,7,8 train → S9/S11 test).

*Last verified: 2026-08-12 07:48 UTC.*

## MVPose (zju3dv/mvpose)

MVPose was run on the full H36M true-GT S9/S11 validation pickle using the
geometry-only top-down triangulation kernel from `zju3dv/mvpose`, bypassing the
2D detector and Re-ID backend. Ground-truth 2D projections were fed to the
adapter.

### Commands run

```bash
# Run inference (CPU / local GPU)
python scripts/sota_baselines/mvpose_h36m_adapter.py \
    --input_pkl tmp/sota_baselines/mvpose_data_a800/h36m_true_gt_val.pkl \
    --output_dir tmp/sota_baselines/mvpose_predictions_full

# Evaluate predictions
python scripts/sota_baselines/eval_mvpose_predictions.py \
    --input_pkl tmp/sota_baselines/mvpose_data_a800/h36m_true_gt_val.pkl \
    --pred_dir tmp/sota_baselines/mvpose_predictions_full \
    --out_json outputs/sota_baselines/mvpose_h36m_true_gt_metrics.json
```

### Results

| Subject | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| S9 | **29.19** | **31.90** | 83,759 frames |
| S11 | **21.54** | **23.15** | 57,971 frames |
| **Combined** | **26.06** | **28.32** | frame-weighted over 141,730 frames |

- Body-12 (non-face) subset: **31.13 mm** MPJPE / **34.45 mm** PA-MPJPE combined.
- Source JSON: `outputs/sota_baselines/mvpose_h36m_true_gt_metrics.json`
- MVPose is competitive with the confidence-weighted DLT baseline
  (25.67 mm) on this true-GT protocol.

## VoxelPose

Not yet trained. The Microsoft `voxelpose-pytorch` repo has been cloned, the H36M
true-GT adapter overlay is ready, and the converted data exists on A800. The
A800 launcher (`scripts/run_voxelpose_h36m_true_gt_a800.sh`) will start training
once GPU 6 or 7 is free. See `docs/sota_voxelpose_h36m_setup.md` for setup
instructions.

## True-GT v2 configs and launchers

New SOTA baseline configs and launchers targeting the corrected
`data/h36m_true_gt_v2/` protocol are in `configs/sota_baselines/`:

- `configs/sota_baselines/voxelpose_h36m_true_gt_v2.yaml` — VoxelPose runtime config
- `configs/sota_baselines/voxelpose_h36m_true_gt_v2_prep.yaml` — VoxelPose prep config
- `configs/sota_baselines/run_voxelpose_h36m_true_gt_v2.sh` — A800 launcher
- `configs/sota_baselines/mvpose_h36m_true_gt_v2.yaml` — MVPose config
- `configs/sota_baselines/run_mvpose_h36m_true_gt_v2.sh` — MVPose launcher

These use `configs/splits/h36m_true_gt_v2_standard.yaml` and write to
`tmp/sota_baselines/*_v2/` and `outputs/sota_baselines/*_v2.*` so they do not
clobber the original v1 baseline outputs.

### MVPose true-GT v2 result (geometry-only fallback)

| Subject | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| S9 | **31.73** | **36.48** | 83,759 frames |
| S11 | **23.76** | **26.58** | 57,971 frames |
| **Combined** | **28.47** | **32.43** | frame-weighted over 141,730 frames |

- Body-12 subset: **35.21 mm** MPJPE / **39.86 mm** PA-MPJPE combined.
- Source JSON: `outputs/sota_baselines/mvpose_h36m_true_gt_v2_metrics.json`
- The v2 raw MPJPE is slightly higher than v1 because the v2 labels/cameras
  are now physically consistent; the geometry-only triangulation still serves
  as a strong baseline.
