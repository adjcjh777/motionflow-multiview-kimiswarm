# VoxelPose & MVPose on H36M True-GT

Scripts to prepare, run, and evaluate the VoxelPose and MVPose SOTA baselines
on the corrected, non-circular H36M true-GT protocol (`S1,5,6,7,8 → S9/S11`).

> **GPU work — start only when the local RTX 4090 is idle.**
> The export step is CPU-only and can be run at any time.

## Status summary

| Method | Upstream repo | Data prep | Training/eval | Notes |
|--------|---------------|-----------|---------------|-------|
| VoxelPose | [microsoft/voxelpose-pytorch](https://github.com/microsoft/voxelpose-pytorch) cloned to `models/voxelpose-pytorch` | Ready | **Not started** | Adapter overlay + H36M true-GT dataset class are in place; waiting for a free GPU |
| MVPose | zju3dv/mvpose cloned to `tmp/sota_baselines/mvpose` | Ready | **Run complete** | H36M true-GT S9/S11: **26.06 mm** MPJPE / **28.32 mm** PA-MPJPE (all-17) |

VoxelPose now uses the official Microsoft PyTorch implementation. The old
`susuhustle/VoxelPose` and `jiajunhua/MvPose` URLs are no longer referenced.
Microsoft's repo clones successfully and the H36M true-GT adapter overlay has
been written (see `voxelpose_h36m_true_gt_a800_overlay/`).

## Layout

```
scripts/sota_baselines/
├── README.md                           # this file
├── check_gpu_free.sh                   # helper: exit 0 iff the local GPU is idle
├── common_export_h36m_true_gt.py       # export our *_m.npz to a common baseline format
├── convert_to_voxelpose_format.py      # common pickle → VoxelPose input
├── convert_to_mvpose_format.py         # common pickle → MVPose input
├── eval_sota_baseline.py               # compute MPJPE/PA-MPJPE on SOTA predictions
├── voxelpose_h36m_config.yaml          # VoxelPose data / repo / output settings
├── voxelpose_h36m_run_config.yaml      # VoxelPose runtime config template
├── mvpose_h36m_config.yaml             # MVPose data / repo / output settings
├── prepare_voxelpose_h36m.sh           # clone / export / convert / train-or-eval
├── prepare_mvpose_h36m.sh              # clone / export / convert / train-or-eval
└── setup_voxelpose_env_a800.sh         # build Python 3.8 / PyTorch 1.12 / CUDA 11.x env
```

Top-level convenience wrappers in `scripts/`:

- `scripts/run_voxelpose_h36m_true_gt.sh`
- `scripts/run_mvpose_h36m_true_gt.sh`

## Quick start

1. **Export H36M true-GT once** (CPU-only):

   ```bash
   python scripts/sota_baselines/common_export_h36m_true_gt.py
   ```

2. **Prepare converted data** (does not require a GPU):

   ```bash
   bash scripts/sota_baselines/prepare_voxelpose_h36m.sh
   bash scripts/sota_baselines/prepare_mvpose_h36m.sh
   ```

   These scripts will warn that the upstream repos are missing and only
   generate the method-specific input files.

3. **Evaluate an upstream method's predictions** (once available):

   ```bash
   python scripts/sota_baselines/eval_sota_baseline.py \
       --pred tmp/sota_baselines/voxelpose_data/h36m_true_gt_pred_s9.npz \
       --gt data/h36m_true_gt/s_09_acts_02_03_..._multiview_m.npz \
       --out_json outputs/sota_baselines/voxelpose_h36m_s9_metrics.json
   ```

## What the scripts do

- `common_export_h36m_true_gt.py` loads `configs/splits/h36m_true_gt_standard.yaml`
  and writes `tmp/sota_baselines/h36m_true_gt_baseline_format.pkl`.
- `prepare_*_h36m.sh` would clone the upstream repo into `tmp/sota_baselines/`,
  convert the common pickle to the method-specific input format, then train
  or evaluate depending on whether a checkpoint exists. **The clone step is
  currently skipped because the configured URLs are unreachable.**
- `check_gpu_free.sh` aborts if GPU memory is in use or if a Python process is
  holding GPU memory.
- `eval_sota_baseline.py` computes MPJPE and PA-MPJPE from any upstream method's
  3D predictions.

## Outputs

- `tmp/sota_baselines/h36m_true_gt_baseline_format.pkl`
- `tmp/sota_baselines/voxelpose_data/` and `tmp/sota_baselines/mvpose_data/`
- `tmp/sota_baselines/VoxelPose/` (after a working repo is cloned)
- `tmp/sota_baselines/MVPose/` (after a working repo is cloned)
- `outputs/sota_baselines/voxelpose_h36m_true_gt_run.log`
- `outputs/sota_baselines/mvpose_h36m_true_gt_run.log`
- Method metrics JSONs under `outputs/sota_baselines/`

## Important constraints

- **Do not run on A800-D or any read-only mount.** The prep scripts exit early
  if the host is `a800-D*` or `/mnt/nvme0n1p1/zhangzy/projects` exists.
- **Do not start these while another training/evaluation run is active.** The
  wrapper scripts gate on `check_gpu_free.sh`, but double-check GPU status
  first.
- These baselines depend on third-party repositories; an internet connection
  is required the first time. **The configured repositories are currently
  unreachable and must be replaced.**

## A800 true-GT baseline

A separate A800 run script and config are provided for VoxelPose on the
corrected H36M true-GT protocol:

- `scripts/sota_baselines/voxelpose_h36m_true_gt_a800.yaml`
- `scripts/run_voxelpose_h36m_true_gt_a800.sh`
- `scripts/sota_baselines/voxelpose_h36m_true_gt_a800_overlay/`
- `docs/sota_voxelpose_h36m_setup.md`

True-GT v2 protocol (corrected, non-circular labels):

- `configs/sota_baselines/voxelpose_h36m_true_gt_v2.yaml`
- `scripts/run_voxelpose_true_gt_v2_a800.sh`

These files are ready to launch once an A800 GPU is free and the old PyTorch
environment issue is resolved. See `docs/sota_voxelpose_h36m_setup.md` for the
environment requirements, joint mapping, and run instructions.

## Missing pieces / next steps

1. **VoxelPose training/evaluation**
   - The Microsoft VoxelPose repo is cloned and the H36M true-GT adapter overlay
     is in place. The next step is to launch `scripts/run_voxelpose_h36m_true_gt_a800.sh`
     on A800 when GPU 6 or 7 is free (project policy: only GPUs 6/7 may be used).
   - A conda environment with Python 3.8 + PyTorch 1.12.1 + CUDA 11.6 can be
     created with `scripts/sota_baselines/setup_voxelpose_env_a800.sh`.
   - The adapter feeds ground-truth 2D points as input heatmaps and uses a blank
     placeholder image, so no raw H36M RGB frames are required.
   - **Auto-launch monitor:** `scripts/monitor_v85_evalsuite_then_launch_voxelpose.sh`
     waits for the v85 post-training eval suite to finish, then automatically
     launches VoxelPose on the first free GPU (6 or 7). The older
     `scripts/monitor_v85_then_launch_voxelpose.sh` is superseded because it
     only watched the old no-fallback eval PID.

2. **Camera convention validation**
   - The common format stores `K`, `R`, `t` in the project's convention and the
     adapter converts them to the VoxelPose format. This must be validated by
     running the first training epoch and checking that MPJPE is reasonable.

3. **Disk and GPU constraints**
   - A800 `/mnt/nvme0n1p1` is ~99 % full. Before launching VoxelPose training,
     ensure there is enough free space for checkpoints and logs.
   - Do not launch VoxelPose while v85 (GPU 7) or the manifest-based DLT-fallback
     evals (GPU 6) are still active.

## MVPose-specific notes

### Upstream implementation

The original `emredog/mvposeestim` (sometimes written `mvposestim`) is no longer
reachable. The closest public implementation is
[`zju3dv/mvpose`](https://github.com/zju3dv/mvpose)
(Dong et al., *Fast and Robust Multi-Person 3D Pose Estimation from Multiple
Views*, CVPR 2019 / T-PAMI 2021).

`zju3dv/mvpose` is an **inference-only**, multi-person method designed for the
Campus and Shelf datasets. Running it on H36M therefore requires a custom
adapter; the A800 config and run script document the joint mapping and stop at
data preparation until that adapter is written.

### Requirements (from upstream `requirements.txt`)

| Package | Pinned version |
|---------|----------------|
| Python | ~3.5–3.6 era |
| torch | 1.0.1.post2 |
| torchvision | 0.2.2 |
| tensorflow_gpu | 1.9.0 |
| numpy | 1.16.2 |
| scipy | 1.2.1 |
| opencv_python | 4.0.0.21 |
| Cython | 0.29.6 |
| setuptools | 39.1.0 |
| easydict | 1.9 |
| seaborn | 0.9.0 |
| requests | 2.21.0 |
| prettytable | 0.7.2 |
| tqdm | 4.29.1 |
| coloredlogs | 10.0 |
| setproctitle | 1.1.10 |
| matplotlib | 2.0.2 |
| visdom | 0.1.8.8 |

**Blocker for A800:** TensorFlow 1.9 / PyTorch 1.0 are CUDA 9-era releases and
are incompatible with Ampere GPUs. A separate environment (CUDA 9 container or
older GPU) is needed before the upstream 2D detector backend can run. A pure-GT
adapter that bypasses the 2D detector would avoid this.

### Joint mapping (H36M true-GT → COCO17)

`zju3dv/mvpose` internally uses COCO17 ordering. The H36M true-GT npz has 
17 body joints, so the five COCO17 facial keypoints are approximated from the
Head joint and marked with zero confidence.

| COCO17 index | COCO17 name | H36M source |
|--------------|-------------|-------------|
| 0 | nose | Head |
| 1 | left_eye | Head |
| 2 | right_eye | Head |
| 3 | left_ear | Head |
| 4 | right_ear | Head |
| 5 | left_shoulder | LShoulder |
| 6 | right_shoulder | RShoulder |
| 7 | left_elbow | LElbow |
| 8 | right_elbow | RElbow |
| 9 | left_wrist | LWrist |
| 10 | right_wrist | RWrist |
| 11 | left_hip | LHip |
| 12 | right_hip | RHip |
| 13 | left_knee | LKnee |
| 14 | right_knee | RKnee |
| 15 | left_ankle | LFoot |
| 16 | right_ankle | RFoot |

The exact map is encoded in `scripts/sota_baselines/mvpose_h36m_a800_config.yaml`
and applied by `scripts/sota_baselines/convert_to_mvpose_format.py` when a
`joint_mapping` block is present.

## Verification commands

```bash
# Check whether the export worked and data keys are present.
python - <<'PY'
import pickle
from pathlib import Path
pkl = Path('tmp/sota_baselines/h36m_true_gt_baseline_format.pkl')
if not pkl.exists():
    print('ERROR: export not found')
else:
    data = pickle.load(open(pkl, 'rb'))
    print('train sequences:', len(data['train']))
    print('val sequences:', len(data['val']))
    print('joint names:', data['joint_names'])
PY

# Check converted data.
ls tmp/sota_baselines/voxelpose_data/
ls tmp/sota_baselines/mvpose_data/
ls tmp/sota_baselines/mvpose_data_a800/

# Run a CPU smoke test of the MVPose H36M adapter on a few frames.
python scripts/sota_baselines/mvpose_h36m_adapter.py \
    --input_pkl tmp/sota_baselines/mvpose_data_a800/h36m_true_gt_val.pkl \
    --output_dir tmp/sota_baselines/mvpose_predictions_smoke \
    --max_frames 10

# Pure-geometry fallback (no upstream import required):
python scripts/sota_baselines/mvpose_h36m_adapter.py \
    --input_pkl tmp/sota_baselines/mvpose_data_a800/h36m_true_gt_val.pkl \
    --output_dir tmp/sota_baselines/mvpose_predictions_smoke \
    --max_frames 10 --fallback_only
```
