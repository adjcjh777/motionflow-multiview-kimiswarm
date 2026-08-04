# A800-D Read-Only Audit Report

Date: 2026-08-04
Scope: `/mnt/nvme0n1/zhangzy/projects` and the running MotionFlow Docker service.
Constraint: read-only operations only.

## SSH Access

- Host: `a800-D` (configured in `~/.ssh/config`)
- User: `zhangzy`
- Authentication: `~/.ssh/id_ed25519`
- Disk: `/dev/nvme0n1p1` 3.5T, 98% used (3.2T used, 95G free)

## Running Docker Service

A `motionflow` container is currently running:

```
NAMES        IMAGE                                  STATUS       PORTS
motionflow   elf3-trainer:20260729-auto-equal-sample   Up 5 days    0.0.0.0:8000->8000/tcp, :::8000->8000/tcp, 8080/tcp
```

Earlier containers (all `Exited`):
- `motionflow-pre-auto-equal-sample-20260729T085958Z`
- `motionflow-pre-equal-sample-hotfix-20260729T083057Z`
- `motionflow-pre-multigpu-tuning-20260729T075059Z`
- `motionflow-pre-live-log-eta-v2-20260728T063526Z`
- ...and more.

## Standard 3D-GT Datasets

No Human3.6M, CMU Panoptic, Shelf, Campus, 3DPW, or AMASS data directories were found under `/mnt/nvme0n1/zhangzy/projects` or `/mnt/nvme0n1` in general.

Search command used:

```bash
find /mnt/nvme0n1 -maxdepth 4 -type d \
  \( -iname '*h36m*' -o -iname '*human3.6m*' -o -iname '*S1' -o -iname '*S5' \
     -o -iname '*S9' -o -iname '*S11' -o -iname '*shelf*' -o -iname '*campus*' \
     -o -iname '*panoptic*' -o -iname '*3dpw*' -o -iname '*amass*' \)
```

Result: empty.

## Available Assets

### SMPL / SMPL-X Body Models (GVHMR project)

`/mnt/nvme0n1/zhangzy/projects/GVHMR/inputs/checkpoints/body_models/` contains:
- `smpl/SMPL_NEUTRAL.pkl`
- `smpl/SMPL_MALE.pkl`
- `smpl/SMPL_FEMALE.pkl`
- `smplx/SMPLX_NEUTRAL.npz`
- `smplx/SMPLX_MALE.npz`
- `smplx/SMPLX_FEMALE.npz`

These can be read-only copied for local testing.

### GVHMR Demo Outputs

`/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/` contains multiple single-view demo results, e.g.:
- `block/hmr4d_results.pt` (152 frames, ~652 KB)
- `squat/hmr4d_results.pt`
- `walk/hmr4d_results.pt`
- ...and many more.

These are real GVHMR outputs useful for testing the `gvhmr_pt_to_ir` adapter. They are single-view, so they cannot directly exercise multi-view fusion, but they validate the IR pipeline with real data.

### Motion Capture NPZ Files

`/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/vendor/mjlab-elf3_beyongmimic/npz/` contains:
- `taiji_elf3_smooth.npz`
- `dance1_subject1_BXI.npz`
- `dance2_final_slow.npz`

These may contain already-retargeted robot motion and are not directly 3D human GT for fusion training.

## Read-Only Copy Performed

Copied for local testing:

```
a800-D:/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/block/hmr4d_results.pt
  -> data/gvhmr_demo/hmr4d_results.pt
```

## Implications for Next Iteration

- Real 3D-GT datasets are **not present** on A800-D. The fastest path to 3D-supervised training is still to register/download Human3.6M or generate AMASS synthetic data.
- GVHMR demo outputs are immediately useful for IR adapter validation and single-view demos.
- SMPL body models on A800-D can be reused for any SMPL-based fitting/evaluation.
