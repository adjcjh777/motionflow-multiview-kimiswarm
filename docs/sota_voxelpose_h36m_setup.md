# VoxelPose H36M True-GT Baseline Setup

This document records how the Microsoft VoxelPose baseline is wired to the
corrected H36M true-GT protocol (`data/h36m_true_gt/`).

## Repository

Upstream repo cloned locally for inspection:

```
models/voxelpose-pytorch   # https://github.com/microsoft/voxelpose-pytorch
```

The A800 launcher (`scripts/run_voxelpose_h36m_true_gt_a800.sh`) will clone the
same repo on the A800 training host if it is not already there.

## Auto-launch after the v85 eval suite

A monitor script is provided so VoxelPose starts automatically once the v85
random-view-dropout training and its post-training evaluation suite finish and a
GPU becomes free:

```bash
nohup bash scripts/monitor_v85_evalsuite_then_launch_voxelpose.sh 2072251 \
    > outputs/sota_baselines/monitor_v85_evalsuite_then_launch_voxelpose_nohup.log 2>&1 &
```

- It waits for the v85 post-training eval-suite monitor (PID `2072251` by default,
  overridable as the first argument).
- It then waits for GPU 6 or 7 to be free and launches
  `scripts/run_voxelpose_h36m_true_gt_a800.sh` on the first available one.
- The old `scripts/monitor_v85_then_launch_voxelpose.sh` is superseded because it
  only watched the original no-fallback eval PID, which is no longer running.

## Python / environment requirements

The upstream `requirements.txt` pins:

```
tqdm==4.29.1
json_tricks==3.13.2
torch==1.4.0
opencv_python==4.0.0.21
prettytable==0.7.2
scipy==1.4.1
torchvision==0.5.0
numpy==1.16.2
matplotlib==2.0.2
easydict==1.9
PyYAML==5.4
tensorboardX==2.1
```

**Compatibility warnings:**

* `torch==1.4.0` is CUDA 10.1 era code and relies on APIs that are deprecated
  or removed in PyTorch 2.x.
* The local WSL repo runs Python 3.13 + torch 2.7.1, so the upstream code will
  not run here without a dedicated Python ≤3.8 environment.
* The A800 host currently has Python 3.10 + torch 2.13.0+cu130. Running the
  original VoxelPose code as-is will likely fail because of API changes.
* The A800 Ampere GPUs do not support CUDA 10.1. To use the original repo, you
  would either need a PyTorch 1.4/CUDA 10.1 container with software emulation,
  or (more realistically) port the code to a newer PyTorch.

For a true SOTA run, the recommended next step is to create a dedicated conda
environment on A800 (Python 3.8, PyTorch ≥1.10, matching CUDA driver) and patch
any incompatibilities, rather than using the system torch.

A ready-to-use setup script is now provided:

- `scripts/sota_baselines/setup_voxelpose_env_a800.sh`

It creates a conda environment (`voxelpose_py38_pt112` by default) with
Python 3.8, PyTorch 1.12.1, torchvision 0.13.1, and cudatoolkit 11.6, then
installs the remaining VoxelPose dependencies and applies the H36M true-GT
adapter overlay. It does **not** start training.

Run it on A800 from the repo root:

```bash
bash scripts/sota_baselines/setup_voxelpose_env_a800.sh
```

Override versions if needed:

```bash
PYTORCH_VERSION=1.8.2 TORCHVISION_VERSION=0.9.1 TORCHAUDIO_VERSION=0.8.1 \
  CUDA_VERSION=11.1 bash scripts/sota_baselines/setup_voxelpose_env_a800.sh
```

## Joint mapping to H36M

The corrected H36M true-GT files (`data/h36m_true_gt/s_*_multiview_m.npz`)
contain 17 joints in the following order:

| Index | Name        | VoxelPose equivalent (Panoptic 15) |
|-------|-------------|------------------------------------|
| 0     | Hip         | mid-hip (root)                     |
| 1     | RHip        | r-hip                              |
| 2     | RKnee       | r-knee                             |
| 3     | RFoot       | r-ankle                            |
| 4     | LHip        | l-hip                              |
| 5     | LKnee       | l-knee                             |
| 6     | LFoot       | l-ankle                            |
| 7     | Spine       | —                                  |
| 8     | Thorax      | —                                  |
| 9     | Neck        | neck                               |
| 10    | Head        | nose / head                        |
| 11    | LShoulder   | l-shoulder                         |
| 12    | LElbow      | l-elbow                            |
| 13    | LWrist      | l-wrist                            |
| 14    | RShoulder   | r-shoulder                         |
| 15    | RElbow      | r-elbow                            |
| 16    | RWrist      | r-wrist                            |

**Mapping strategy:** we train VoxelPose directly on the H36M 17-joint skeleton
rather than mapping to the Panoptic/COCO skeleton. This is possible because the
adapter (`scripts/sota_baselines/voxelpose_h36m_true_gt_a800_overlay/h36m_true_gt.py`)
generates the per-view input heatmaps from the GT 2D points, so the 2D backbone
(backbone_model) is disabled and there is no pretrained COCO-joint mismatch.

## Data flow

```
data/h36m_true_gt/s_*_multiview_m.npz
    |
    v
common_export_h36m_true_gt.py
    |
    v
tmp/sota_baselines/h36m_true_gt_baseline_format.pkl
    |
    v
convert_to_voxelpose_format.py
    |
    v
tmp/sota_baselines/voxelpose_data/h36m_true_gt_annotations.pkl
    + blank_h36m.png (placeholder image)
```

The adapter reads `h36m_true_gt_annotations.pkl` and builds per-view records
containing:

* `image`: path to the blank placeholder image.
* `joints_3d` / `joints_3d_vis`: single-person GT pose.
* `joints_2d` / `joints_2d_vis`: projected GT 2D points.
* `camera`: VoxelPose-format camera dict (`R`, `T`, `fx/fy/cx/cy`, `k`, `p`).
* `pred_pose2d`: GT 2D points used as the multi-view input heatmaps.

## Camera convention

Our project stores cameras as `x = K (R X + t)` with `X` in metres. VoxelPose
expects `x_cam = R (X - T)` where `T` is the camera centre in world coordinates.
The adapter converts with:

```
T = -R^T * t
```

No unit scaling is applied; both 2D coordinates and camera intrinsics remain in
pixels, and 3D coordinates remain in metres.

## Run script / config

* `scripts/sota_baselines/voxelpose_h36m_true_gt_a800.yaml` — A800 VoxelPose
  experiment config.
* `scripts/run_voxelpose_h36m_true_gt_a800.sh` — A800 launcher that exports
  data, clones the upstream repo, applies the adapter overlay, and runs training.
* `scripts/sota_baselines/voxelpose_h36m_true_gt_a800_overlay/` — adapter files
  and the `function.py` patcher.

To launch (from the local WSL repo, or directly on a800-D):

```bash
ssh a800-D 'bash /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/scripts/run_voxelpose_h36m_true_gt_a800.sh'
```

The script refuses to start if GPU memory is already in use. As of the latest
status check, GPU 7 is busy with v85 random view dropout training and GPU 6 is
running lightweight DLT-fallback evaluations, so VoxelPose training has **not yet
been launched**. The project GPU policy limits MotionFlow runs to GPUs 6 and 7.

## Known limitations / blockers

1. **Environment:** the upstream torch 1.4.0 dependency is incompatible with
   the A800's current torch 2.13.0. Run
   `scripts/sota_baselines/setup_voxelpose_env_a800.sh` to create the
   recommended Python 3.8 + PyTorch 1.12.1 + CUDA 11.6 environment before
   launching training.
2. **Raw H36M frames:** the project does not store raw H36M video frames. The
   adapter uses GT 2D points as the 2D input, which makes this an *oracle* 2D
   baseline rather than a full image-based VoxelPose run. It still tests the
   3D triangulation/voxel aggregation component.
3. **GPU availability:** GPUs 6 and 7 are currently occupied. VoxelPose must
   wait for a free GPU per project policy; never use GPUs 0–5.
