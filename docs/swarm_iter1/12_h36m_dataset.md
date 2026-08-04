# Human3.6M for Multi-view 3D Pose

## TL;DR

Human3.6M is a large indoor mocap dataset for multi-view 3D human pose: 3.6 million frames, 11 actors, 17 actions, captured by 4 calibrated 50 Hz cameras. Ground truth is 3D joint positions (32 joints, world coordinates in mm). For multi-view work you need: the raw videos/frames, the per-subject D3_Positions CDF files, and the camera intrinsics/extrinsics. Use the official site + registration for download; use cdflib or the VideoPose3D preprocessing script to convert CDFs to .npz arrays.

## Key Findings

### Dataset structure

- Subjects: 11 actors (S1, S5, S6, S7, S8, S9, S11 are standard train/test; S2-S4 are sometimes used as unlabeled data).
- Views: 4 fixed, calibrated cameras.
- Actions: 17 daily actions (e.g. Walking, Eating, Discussion, Smoking, Photo, Greeting).
- GT format: MyPoseFeatures/D3_Positions/<Action>.cdf stores the pose variable Pose with shape (frames, 96), which is reshaped to (frames, 32, 3) (mm in world coordinates).
- Skeleton: 32 joints; common pipelines reduce to 17 active joints by removing static/end-site joints.
- Cameras: Each subject has 4 camera IDs (54138969, 55011271, 58860488, 60457274). Intrinsic and extrinsic parameters are distributed in many public repos (e.g. VideoPose3D). Note that public hard-coded extrinsics are available for S1, S5, S6, S7, S8, S9, S11; S2-S4 are often omitted.

### Download

1. Register at the official Human3.6M site: http://vision.imar.ro/human3.6m/
2. Download Poses -> D3 Positions for the desired subjects (Poses_D3_Positions_S*.tgz).
3. (Optional) Download videos for visualization/frame extraction.
4. Accept the license and cite the TPAMI 2014 paper.

### Preprocessing (minimal pipeline)

~~~bash
# Extract downloaded archives
mkdir -p h36m && tar xzf Poses_D3_Positions_S1.tgz -C h36m

# Optional: extract frames from a video for a specific view
# ffmpeg -i S1/Videos/Walking.54138969.mp4 -vf "fps=50" frames/%04d.jpg
~~~

Convert CDF ground truth to .npz:

~~~python
import cdflib
import numpy as np

cdf = cdflib.CDF("h36m/S1/MyPoseFeatures/D3_Positions/Walking.cdf")
# Pose shape: (frames, 32*3) -> (frames, 32, 3)
positions = cdf["Pose"].reshape(-1, 32, 3)  # mm, world coordinates
positions_m = positions / 1000.0             # convert to meters

np.savez_compressed("data_3d_h36m.npz",
                    positions_3d={"S1": {"Walking": positions_m}})
~~~

### Multi-view GT format and projection

- 3D GT lives in a common world coordinate frame (mm by default; converted to meters in most pipelines).
- For each camera you need:
  - orientation: quaternion (world -> camera rotation)
  - translation: camera position in world (mm)
  - focal_length, center (principal point), radial_distortion, tangential_distortion
- A 3D point X_world is projected to a given camera with:

~~~python
import numpy as np
from common.camera import world_to_camera, project_to_2d

# cam is a dict with orientation, translation, intrinsic vector
X_cam = world_to_camera(X_world, cam["orientation"], cam["translation"])
X_2d  = project_to_2d(X_cam, cam["intrinsic"])
~~~

- Typical intrinsic vector layout: [fx, fy, cx, cy, k1, k2, k3, p1, p2].

## References / Code

- Official website and download: http://vision.imar.ro/human3.6m/
- Paper: Ionescu et al., "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments," TPAMI 2014.
- Preprocessing/loader code: https://github.com/facebookresearch/VideoPose3D/blob/main/common/h36m_dataset.py
- Camera helpers and projection: https://github.com/facebookresearch/VideoPose3D/blob/main/common/camera.py
- CDF conversion script: https://github.com/facebookresearch/VideoPose3D/blob/main/data/prepare_data_h36m.py
