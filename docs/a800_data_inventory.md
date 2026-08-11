# A800-D Data Inventory: `/mnt/nvme0n1/zhangzy/projects`

Host: `a800-D` (read-only audit)
Scanned path: `/mnt/nvme0n1/zhangzy/projects`
Inventory date: `2026-08-11T02:43:39Z`

This is a read-only snapshot of datasets, checkpoints, and related project directories on the A800-D `projects` mount that are used by or relevant to the MotionFlow-MultiView (ICRA/CVPR 2027) work.

## Executive summary

- `/mnt/nvme0n1/zhangzy/projects` is **not** the primary home of the MotionFlow-MultiView training repository or its human-pose datasets (H36M, MPI-INF-3DHP, Shelf/Campus). Those live under `/mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm*` (outside `projects`).
- Within `projects`, the MotionFlow-related assets are dominated by **robot motion reference data** used by the `motionflow-6df139c-build` ELF3 pipeline, plus pretrained checkpoints for GVHMR/HMR2/ViTPose and SMPL/SMPL-X body models.
- This audit did not modify any files on `a800-D`.

## Top-level project directories (MotionFlow-relevant)

| Path | Total size | Notes |
|------|------------|-------|
| `motionflow-6df139c-build` | 299M | MotionFlow/ELF3 build artifact (frontend + backend + vendored motion data) |
| `motionflow-research-multiview-easymocap-robot-profiles` | 157M | Research repo for multi-view EasyMocap robot profiles |
| `motionflow-f49d93e-build-KieqEr` | 11M | Earlier MotionFlow build artifact (no vendored datasets) |
| `GVHMR` | 7.1G | GVHMR repo with pretrained checkpoints and demo outputs |
| `GMR` | 2.3G | General motion retargeting repo and saved motion data |
| `gmr-motionlab` | 1.8G | gmr-motionlab repo and assets |
| `mjlab_elf3` | 3.5G | MuJoCo/ELF3 robot training repo (motion data + trained policies) |
| `elf3-video-to-policy` | 855M | ELF3 video-to-policy repo with vendored motion data |
| `summercamp` | 672M | Summer-camp ELF3 RL controller examples |
| `smplx` | 6.1M | SMPL-X body-model utilities and transfer data |

## Robot motion reference datasets

### 1. `motionflow-6df139c-build/vendor/mjlab-elf3_beyongmimic/npz`

Path on disk: `/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/vendor/mjlab-elf3_beyongmimic/npz`
Total size: **17 MB**

| File | Size | Shape / contents |
|------|------|------------------|
| `dance1_subject1_BXI.npz` | 12M | Reference dance motion |
| `dance2_final_slow.npz` | 2.3M | Reference dance motion |
| `taiji_elf3_smooth.npz` | 3.5M | Reference taiji motion |

Format (verified on `taiji_elf3_smooth.npz`):

```text
Keys: fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w,
      body_ang_vel_w, qpos_elf3, qpos_elf3_columns, joint_names, body_names
fps              ()
joint_pos        (1934, 29)
joint_vel        (1934, 29)
body_pos_w       (1934, 30, 3)
body_quat_w      (1934, 30, 4)
body_lin_vel_w   (1934, 30, 3)
body_ang_vel_w   (1934, 30, 3)
qpos_elf3        (1934, 36)
qpos_elf3_columns (36,)
joint_names      (29,)
body_names       (30,)
```

These files are duplicated (same names/sizes) in:

- `/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/vendor/mjlab-elf3_beyongmimic/npz` (17M)
- `/mnt/nvme0n1/zhangzy/projects/elf3-video-to-policy/vendor/mjlab-elf3_beyongmimic/npz` (18M; also contains `cmu_walk.npz`)

### 2. `mjlab_elf3/data/motions`

Path on disk: `/mnt/nvme0n1/zhangzy/projects/mjlab_elf3/data/motions`
Total size: **58 MB**

| File | Size | Notes |
|------|------|-------|
| `unitree_classic_movie_dancing_elf3.npz` | 4.1M | Dance reference |
| `7ms31_elf3.npz` | 4.6M | Motion reference |
| `7ms_elf3.npz` | 3.9M | Motion reference |
| `7msnew_elf3.npz` | 3.9M | Motion reference |
| `g7ms_elf3.npz` | 3.9M | Motion reference |
| `block_long_elf3.npz` | 732K | Motion reference |
| `walk_20260623_073019.npz` | 424K | Walk reference |
| `walk.npz` | 224K | Walk reference |
| `data_elf3.npz` | 908K | Motion reference |
| `squat_newelf3.npz` | 440K | Squat reference |
| `squat_down_good.npz` | 292K | Squat reference |
| `squat2_elf3.npz` | 520K | Squat reference |
| `squat_elf3.npz` | 440K | Squat reference |
| `block_elf3.npz` | 440K | Block reference |
| `block_newelf3.npz` | 440K | Block reference |

Format: `fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w` (some files also include `joint_names`, `body_names`). These are raw MuJoCo body/joint trajectories for the ELF3 robot.

### 3. `GMR/save`

Path on disk: `/mnt/nvme0n1/zhangzy/projects/GMR/save`
Total size: **21 MB**

Contains paired `.pkl`/`.npz` motion clips produced by the General Motion Retargeting (GMR) pipeline. Examples:

| File | Size | Notes |
|------|------|-------|
| `dance1_subject1_BXI.npz` | 12M | GMR-processed dance motion |
| `data_g1.npz` / `data_g1.pkl` | 898K / 88K | G1 robot retargeted motion |
| `data_elf3.npz` / `data_elf3.pkl` | 928K / 90K | ELF3 retargeted motion |
| `unitree_classic_movie_dancing_elf3.pkl` | 340K | Processed dance clip |
| `taiji_elf3.pkl` | 240K | Processed taiji clip |
| `freemocap_rotated_elf3.pkl` | 208K | Processed freemocap clip |
| `wushu_elf3.pkl` | 228K | Processed wushu clip |
| ... | ... | Additional squat/stand/walk/block clips |

### 4. `summercamp/bxi_rl_controller_ros2_example/src/bxi_example_py_elf3/data`

Path on disk: `/mnt/nvme0n1/zhangzy/projects/summercamp/bxi_rl_controller_ros2_example/src/bxi_example_py_elf3/data`
Total size: **81 MB**

Contains reference motion `.npz` files and exported ONNX policy files for the BXI/ELF3 RL controller:

| File | Size | Notes |
|------|------|-------|
| `dance.npz` | 2.6M | Dance reference |
| `block.npz` | 2.0M | Block reference |
| `block_newelf3.npz` | 449K | Block reference |
| `recover.npz` | 3.0M | Recover reference |
| `forward_flip.npz` | 1.0M | Flip reference |
| `squat_policy.npz` | 1.6M | Squat reference |
| `squat_policy1.npz` | 440K | Squat reference |
| `squat_policy2.npz` | 520K | Squat reference |
| `applause.pkl` | 772K | Applause motion |
| `naotou.pkl` | 109K | Motion clip |
| `*.onnx` | various | Exported RL policies |

## Human body model / pretrained checkpoint datasets

### `GVHMR/inputs/checkpoints`

Path on disk: `/mnt/nvme0n1/zhangzy/projects/GVHMR/inputs/checkpoints`
Total size: **6.2 GB**

| Subdirectory | Size | Contains |
|--------------|------|----------|
| `body_models/smpl` | ~711M | SMPL models (`SMPL_FEMALE.pkl`, `SMPL_MALE.pkl`, `SMPL_NEUTRAL.pkl`) |
| `body_models/smplx` | ~313M | SMPL-X models (`SMPLX_FEMALE.npz`, `SMPLX_MALE.npz`, `SMPLX_NEUTRAL.npz`) |
| `hmr2` | 2.6G | HMR2 checkpoint `epoch=10-step=25000.ckpt` |
| `vitpose` | 2.4G | ViTPose checkpoint `vitpose-h-multi-coco.pth` |
| `gvhmr` | 156M | GVHMR pretrained model `gvhmr_siga24_release.ckpt` |
| `yolo` | 131M | YOLOv8x detection model |
| `dpvo` | 14M | DPVO checkpoint |

### `smplx` utilities

Path on disk: `/mnt/nvme0n1/zhangzy/projects/smplx`
Total size: **6.1 MB**

- SMPL-X transfer / support data in `transfer_data/support_data/github_data/amass_sample.npz` (2.0M)
- Config files for SMPL/SMPL-X/SMPL-H conversions

## GVHMR demo outputs

Path on disk: `/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo`
Total size: **613 MB**

Per-video inference results. Each directory typically contains `hmr4d_results.pt`, `hmr4d_results.betas10.pt`, and a `preprocess/` folder with `bbx.pt`, `vit_features.pt`, `vitpose.pt`.

| Demo directory | Size |
|----------------|------|
| `wushu` | 186M |
| `unitree_classic_movie_dancing` | 57M |
| `pufu` | 55M |
| `xiaoxuanfeng` | 36M |
| `xiaoxaunf` | 36M |
| `From-standing-to-crawling-forward` | 36M |
| `taiji` | 34M |
| `galbot` | 26M |
| `love_waltz` | 25M |
| `7motion` | 22M |
| `freemocap_rotated` | 20M |
| `beatutiful_mythos` | 19M |
| (remaining demos) | < 15M each |

## What is **not** in `/mnt/nvme0n1/zhangzy/projects`

The main MotionFlow-MultiView research repository and its human-pose datasets live outside the `projects` directory:

| Repository path | Data present | Notes |
|-----------------|--------------|-------|
| `/mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm` | `data/webbridge` (1.6G) | Active local repo |
| `/mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm-iter20` | `data/webbridge` (1.6G) | Iter-20 repo |
| `/mnt/nvme0n1/zhangzy/motionflow-mv-h36m-truegt` | `data/h36m_true_gt` (290M) | True-GT H36M mocap world coordinates |
| `/mnt/nvme0n1/zhangzy/motionflow-mv-detected-long` | `data/webbridge` (848K) | Detected-2D variant |

If the goal is to locate the **human multi-view pose datasets** (H36M, MPI-INF-3DHP, Shelf/Campus), those should be inventoried from `/mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm*` rather than from `projects`.

## Notes / caveats

- All operations were read-only; no files on `a800-D` were created, modified, or deleted.
- Sizes are from `du -sh` / `ls -l` and are approximate.
- The `projects` tree contains large duplicated robot-motion `.npz` files across `motionflow-6df139c-build`, `motionflow-research-multiview-easymocap-robot-profiles`, and `elf3-video-to-policy`.
