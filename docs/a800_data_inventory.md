# A800-D Data Inventory: MotionFlow-MultiView

Host: `a800-D`  
Scanned path: `/mnt/nvme0n1p1/zhangzy/projects`  
Inventory date: `2026-08-09T03:46:24+00:00`

This is a read-only snapshot of datasets, checkpoints, and related project directories on the A800-D `projects` mount that are used by or relevant to the MotionFlow-MultiView (ICRA/CVPR 2027) work.

## Top-level MotionFlow-related directories

| Path | Total size | Last modified | Notes |
|------|------------|---------------|-------|
| `/mnt/nvme0n1p1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles` | 157M | 2026-07-28 | Research repo for multi-view EasyMocap robot profiles |
| `/mnt/nvme0n1p1/zhangzy/projects/motionflow-6df139c-build` | 299M | 2026-07-29 | MotionFlow build artifact (frontend + backend) |
| `/mnt/nvme0n1p1/zhangzy/projects/motionflow-f49d93e-build-KieqEr` | 11M | 2026-07-23 | Earlier MotionFlow build artifact |
| `/mnt/nvme0n1p1/zhangzy/projects/GVHMR` | 7.1G | 2026-05-27 | GVHMR repo with pretrained checkpoints and demo outputs |
| `/mnt/nvme0n1p1/zhangzy/projects/GMR` | 2.3G | 2026-06-22 | General motion retargeting repo and saved motion data |
| `/mnt/nvme0n1p1/zhangzy/projects/gmr-motionlab` | 1.8G | 2026-06-23 | gmr-motionlab repo and assets |
| `/mnt/nvme0n1p1/zhangzy/projects/smplx` | 6.1M | 2026-05-21 | SMPL-X body-model utilities and transfer data |

## `motionflow-research-multiview-easymocap-robot-profiles`

The repo itself is 157M; most assets are vendored.

### Vendored components

| Subdirectory | Size | Notes |
|--------------|------|-------|
| `vendor/GMR` | 25M | General motion retargeting code/data |
| `vendor/GVHMR` | 15M | GVHMR source used as a dependency |
| `vendor/mjlab-elf3_beyongmimic` | 112M | MuJoCo motion data and RL checkpoints |
| `vendor/smplx` | 3.3M | SMPL-X utilities |

### Key data files in the vendored tree

| Size | Date | Path |
|------|------|------|
| 11.2M | 2026-07-28 | `vendor/mjlab-elf3_beyongmimic/npz/dance1_subject1_BXI.npz` |
| 2.3M | 2026-07-28 | `vendor/mjlab-elf3_beyongmimic/npz/dance2_final_slow.npz` |
| 3.6M | 2026-07-28 | `vendor/mjlab-elf3_beyongmimic/npz/taiji_elf3_smooth.npz` |
| 1.9M | 2026-07-28 | `vendor/smplx/transfer_data/support_data/github_data/amass_sample.npz` |

## `GVHMR` pretrained checkpoints

`/mnt/nvme0n1p1/zhangzy/projects/GVHMR/inputs/checkpoints` totals **6.2G**.

| Subdirectory | Size | Contains |
|--------------|------|----------|
| `inputs/checkpoints/hmr2` | 2.6G | HMR2 checkpoint (`epoch=10-step=25000.ckpt`, ~2.7 GB) |
| `inputs/checkpoints/vitpose` | 2.4G | ViTPose checkpoint (`vitpose-h-multi-coco.pth`, ~2.5 GB) |
| `inputs/checkpoints/body_models` | 1.0G | SMPL/SMPL-X body models |
| `inputs/checkpoints/gvhmr` | 156M | GVHMR pretrained model (`gvhmr_siga24_release.ckpt`) |
| `inputs/checkpoints/yolo` | 131M | YOLOv8x detection model |
| `inputs/checkpoints/dpvo` | 14M | DPVO checkpoint |

### Individual body-model files

| Size | Date | Path |
|------|------|------|
| 247.1M | 2026-05-11 | `inputs/checkpoints/body_models/smpl/SMPL_FEMALE.pkl` |
| 246.6M | 2026-05-11 | `inputs/checkpoints/body_models/smpl/SMPL_MALE.pkl` |
| 247.2M | 2026-05-11 | `inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl` |
| 108.8M | 2026-05-09 | `inputs/checkpoints/body_models/smplx/SMPLX_FEMALE.npz` |
| 108.8M | 2026-05-09 | `inputs/checkpoints/body_models/smplx/SMPLX_MALE.npz` |
| 108.8M | 2026-05-21 | `inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` |

### Pretrained model files

| Size | Date | Path |
|------|------|------|
| 163.5M | 2026-05-11 | `inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt` |
| 2.54G | 2026-05-09 | `inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt` |
| 2.55G | 2026-05-09 | `inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth` |
| 136.9M | 2026-05-11 | `inputs/checkpoints/yolo/yolov8x.pt` |
| 14.2M | 2026-05-11 | `inputs/checkpoints/dpvo/dpvo.pth` |

## `GVHMR` demo outputs

`/mnt/nvme0n1p1/zhangzy/projects/GVHMR/outputs/demo` totals **613M** and holds per-video GVHMR inference results. Each demo directory usually contains `hmr4d_results.pt`, `hmr4d_results.betas10.pt`, and a `preprocess/` folder with `bbx.pt`, `vit_features.pt`, and `vitpose.pt`.

| Demo directory | Size | Last modified |
|----------------|------|---------------|
| `outputs/demo/wushu` | 186M | 2026-07-06 |
| `outputs/demo/unitree_classic_movie_dancing` | 57M | 2026-06-29 |
| `outputs/demo/pufu` | 55M | 2026-07-14 |
| `outputs/demo/xiaoxuanfeng` | 36M | 2026-06-30 |
| `outputs/demo/xiaoxaunf` | 36M | 2026-07-03 |
| `outputs/demo/From-standing-to-crawling-forward` | 36M | 2026-07-10 |
| `outputs/demo/taiji` | 34M | 2026-07-07 |
| `outputs/demo/galbot` | 26M | 2026-07-01 |
| `outputs/demo/love_waltz` | 25M | 2026-07-01 |
| `outputs/demo/7motion` | 22M | 2026-06-03 |
| `outputs/demo/freemocap_rotated` | 20M | 2026-07-22 |
| `outputs/demo/beatutiful_mythos` | 19M | 2026-07-01 |
| `outputs/demo/1783905861112_VID_442` | 11M | 2026-07-13 |
| `outputs/demo/walk` | 9.1M | 2026-06-05 |
| `outputs/demo/tennis` | 8.8M | 2026-05-21 |
| `outputs/demo/quanji` | 8.8M | 2026-07-06 |
| `outputs/demo/test1` | 6.3M | 2026-05-13 |
| `outputs/demo/squat2` | 4.5M | 2026-05-29 |
| `outputs/demo/hi` | 4.3M | 2026-06-10 |
| `outputs/demo/squat` | 4.0M | 2026-05-27 |
| `outputs/demo/stand` | 3.7M | 2026-06-11 |
| `outputs/demo/block_long` | 3.7M | 2026-06-01 |
| `outputs/demo/block` | 2.2M | 2026-05-29 |

## Other related datasets / saved motions

| Path | Size | Date | Notes |
|------|------|------|-------|
| `/mnt/nvme0n1p1/zhangzy/projects/GMR/save` | 21M | various | Saved `.pkl`/`.npz` motion clips for retargeting |
| `/mnt/nvme0n1p1/zhangzy/projects/gmr-motionlab/assets` | 1.5G | 2026-06-23 | Body models and retargeting assets |
| `/mnt/nvme0n1p1/zhangzy/projects/gmr-motionlab/sample/Extended_3_stageii.pkl` | 448K | 2026-06-22 | Sample motion data |
| `/mnt/nvme0n1p1/zhangzy/projects/smplx/transfer_data` | 2.0M | 2026-05-21 | SMPL-X transfer/support data |

## Notes / caveats

* This inventory covers only `/mnt/nvme0n1p1/zhangzy/projects`. The active MotionFlow-MultiView training repository and its `outputs/` directory live under `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20` (outside `projects`) and are not included here.
* All operations were read-only; no files on `a800-D` were modified.
* Sizes are from `du -sh` and file sizes; dates are `mtime`.
