# CMU Panoptic Dataset: Multiview Structure & Small-Subset Usage

## TL;DR

CMU Panoptic is a massively multiview capture dataset (480 VGA + 31 HD + 10 Kinect2 cameras) for social motion and 3D pose analysis. For a quick, low-cost start, use the official `panoptic-toolbox` repo and download the small sample sequence `171204_pose1_sample` with only a few HD views. Data is organized per sequence: per-view videos, a calibration JSON, and per-frame 3D skeleton/face/hand JSON files.

## Key Conclusions

- **Capture layout**: 480 VGA cameras (640×480 @ 25 fps), 31 HD cameras (1920×1080 @ 29.97 fps), 10 Kinect v2 sensors (RGB-D). All hardware-synchronized.
- **Public data**: ~65 sequences, ~5.5 hours, ~1.5M 3D skeletons. Research-only license.
- **Per-sequence structure** (after download):
  ```
  <seq>/hdVideos/hd_00_XX.mp4              # 31 HD videos
  <seq>/vgaVideos/KINECTNODE%d/vga_XX_XX.mp4  # 480 VGA videos
  <seq>/calibration_<seq>.json             # camera intrinsics/extrinsics/distortion
  <seq>/hdPose3d_stage1_coco19.tar       # per-frame 3D body keypoints (COCO19)
  <seq>/hdFace3d.tar / hdHand3d.tar        # per-frame 3D face/hand keypoints
  ```
- **Calibration JSON** contains a list of cameras with `panel`, `node`, `K`, `distCoef`, `R`, `t`. Camera identity is the tuple `(panel, node)`; first HD camera is `(0,0)`.
- **3D body skeleton** is one JSON per frame. Each person has `id` and `joints19` in `[x,y,z,c, ...]` for 19 joints: Neck, Nose, BodyCenter, shoulders, elbows, wrists, hips, knees, ankles, eyes, ears.
- **Small-subset workflow**:
  1. Clone the toolbox:
     ```bash
     git clone https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox.git
     cd panoptic-toolbox
     ```
  2. Download the small sample with only a few HD views (no VGA):
     ```bash
     ./scripts/getData.sh 171204_pose1_sample 0 5
     ```
     The arguments are `sequenceName vgaCount hdCount`. Default is `0 31`; adjust to limit download size.
  3. Extract frames and annotations (requires `ffmpeg`):
     ```bash
     ./scripts/extractAll.sh 171204_pose1_sample
     ```
  4. If the CMU server is slow, use the SNU mirror:
     ```bash
     ./scripts/getData.sh 171204_pose1_sample 0 5 --snu-endpoint
     ```
- **Minimal Python snippet** to load calibration and project a 3D point:
  ```python
  import json, numpy as np

  seq = '171204_pose1_sample'
  with open(f'{seq}/calibration_{seq}.json') as f:
      calib = json.load(f)
  cameras = {(c['panel'], c['node']): c for c in calib['cameras']}
  cam = cameras[(0, 0)]
  K, R, t = np.array(cam['K']), np.array(cam['R']), np.array(cam['t']).reshape(3, 1)
  # X: 3xN world points
  # Xc = R @ X + t
  # x = K @ Xc
  ```
- **Demo dependencies**: `numpy`, `matplotlib`, `ffmpeg`; optional `pyopengl` for the 3D viewer.

## Reference Links / Code

- [CMU Panoptic dataset homepage](http://domedb.perception.cs.cmu.edu/)
- [panoptic-toolbox (download & demo scripts)](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox)
- [List of released sequences v1.2](https://docs.google.com/spreadsheets/d/1eoe74dHRtoMVVFLKCTJkAtF8zqxAnoo2Nt15CYYvHEE/edit#gid=1333444170)
- Key scripts in the toolbox:
  - `scripts/getData.sh` – download a subset of a sequence
  - `scripts/extractAll.sh` – extract frames and 3D annotations
  - `python/panutils.py` – helpers for projection and camera ordering

## Citation (recommended)

```bibtex
@article{Joo_2017_TPAMI,
  title={Panoptic Studio: A Massively Multiview System for Social Interaction Capture},
  author={Joo, Hanbyul and Simon, Tomas and Li, Xulong and Liu, Hao and Tan, Lei and Gui, Lin and Banerjee, Sean and Godisart, Timothy Scott and Nabbe, Bart and Matthews, Iain and Kanade, Takeo and Nobuhara, Shohei and Sheikh, Yaser},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2017}
}
```
