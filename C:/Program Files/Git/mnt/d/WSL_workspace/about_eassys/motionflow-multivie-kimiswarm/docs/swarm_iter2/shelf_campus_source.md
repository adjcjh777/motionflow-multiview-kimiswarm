# Shelf / Campus Multi-View Human Pose Dataset — Download & Format Notes

> Scope: document how to obtain the Shelf and Campus multi-view human pose datasets, what the extracted data looks like, and how to read the calibration/ground-truth files.  No large files are downloaded here.

## 1. Overview

These two datasets are the standard benchmarks used in multi-view 3D human pose papers such as:

- **VoxelPose** (Tu et al., ECCV 2020) — [microsoft/voxelpose-pytorch](https://github.com/microsoft/voxelpose-pytorch)
- **Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS** (Chen et al., CVPR 2020) — [longcw/crossview_3d_pose_tracking](https://github.com/longcw/crossview_3d_pose_tracking)
- **Microsoft Graph-based Multi-view 3D Human Pose Estimation**

They are often distributed together as “Campus” and “Shelf”.

---

## 2. Direct Download Sources / Mirrors

### 2.1 Recommended one-click mirror (Google Drive)
The authors of the Cross-View Tracking paper provide a single Google Drive folder containing **Campus, Shelf, StoreLayout1 and StoreLayout2** already organized:

- **Google Drive:** https://drive.google.com/drive/folders/1LJGcP2v0aQDmetnCzO2PiRP1v4jU6sFC?usp=drive_link
- Source: [longcw/crossview_3d_pose_tracking#dataset](https://github.com/longcw/crossview_3d_pose_tracking#dataset)

This is currently the most reliable direct mirror; the original TUM hosting page for Shelf is no longer active (it redirects to a generic department landing page).

### 2.2 Original / alternate sources

| Dataset | Original source | Status |
|---------|-----------------|--------|
| **Campus** | https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/ | Still referenced |
| **Shelf**  | http://campar.in.tum.de/Chair/MultiHumanPose | **Dead / redirected** as of 2024–2025 |

Because the Shelf original link is dead, use the Google Drive mirror above, or obtain the data from the VoxelPose repository (which also ships pre-processed camera parameters, see §5).

---

## 3. Extracted Folder Structure

After extraction you should see two top-level folders, e.g.:

```text
Campus_Seq1/
├── annotation_2d.json
├── annotation_3d.json
├── calibration.json
├── detection.json
├── frames/
│   ├── Camera0/
│   ├── Camera1/
│   └── Camera2/
│       ├── 0060.720.jpg
│       ├── 0060.760.jpg
│       └── ...
└── result_3d.json

Shelf_Seq1/
├── annotation_2d.json
├── annotation_3d.json
├── calibration.json
├── detection.json
├── frames/
│   ├── Camera0/
│   ├── Camera1/
│   ├── Camera2/
│   ├── Camera3/
│   └── Camera4/
└── result_3d.json
```

- `frames/Camera<N>/` — JPEG image sequences, named by timestamp in seconds (e.g. `0060.720.jpg`).
- `calibration.json` — camera intrinsics/extrinsics (see §4).
- `annotation_2d.json` / `annotation_3d.json` — 2D and 3D ground-truth poses.
- `detection.json` — 2D pose detections (CPN).
- `result_3d.json` — 3D tracking result from the paper.

---

## 4. Camera Calibration Format (`calibration.json`)

### 4.1 Format
The calibration file is a JSON object keyed by camera id (`"0"`, `"1"`, …).  Each entry contains:

```json
{
  "0": {
    "R": [[...], [...], [...]],   // 3x3 rotation matrix (world -> cam)
    "T": [[...], [...], [...]],   // 3x1 translation vector (mm)
    "fx": 1063.512085,            // focal length x
    "fy": 1071.863647,            // focal length y
    "cx": 511.738251,             // principal point x
    "cy": 350.088287,             // principal point y
    "k": [[0.0], [0.0], [0.0]],   // radial distortion coeffs (k1, k2, k3)
    "p": [[0.0], [0.0]]           // tangential distortion coeffs (p1, p2)
  },
  ...
}
```

- `R` and `T` together form the camera pose / extrinsics.
- `fx, fy, cx, cy` are the pinhole intrinsics.
- Distortion coefficients are usually all zeros in the pre-packaged versions; the provided image data is already undistorted or the effect is negligible for these datasets.

### 4.2 How to read it

**Using plain Python / OpenCV:**

```python
import json, numpy as np

with open('Shelf_Seq1/calibration.json') as f:
    calib = json.load(f)

cam0 = calib['0']
K = np.array([[cam0['fx'], 0.0,       cam0['cx']],
              [0.0,        cam0['fy'], cam0['cy']],
              [0.0,        0.0,       1.0]])
R = np.array(cam0['R'])
T = np.array(cam0['T']).reshape(3)

# world-to-camera transform (4x4)
Tw = np.eye(4)
Tw[:3, :3] = R
Tw[:3,  3] = T

# projection matrix P = K [I | 0] Tw^-1
P = K @ np.hstack((np.eye(3), np.zeros((3, 1)))) @ np.linalg.inv(Tw)
```

**Using the reference loader (crossview repo):**

```python
from crossview_dataset.calib.calibration import Calibration
calibration = Calibration.from_json('Shelf_Seq1/calibration.json')
P = calibration.get_projection_matrix(camera_id='0')
```

### 4.3 VoxelPose pre-processed calibration
If you use the VoxelPose repo, it ships its own camera parameters in a slightly different JSON layout:

- `data/Shelf/calibration_shelf.json`
- `data/CampusSeq1/calibration_campus.json`

These have the same per-camera keys (`R`, `T`, `fx`, `fy`, `cx`, `cy`, `k`, `p`) and can be read with the same snippet above.  See [microsoft/voxelpose-pytorch](https://github.com/microsoft/voxelpose-pytorch).

---

## 5. 3D Ground-Truth Format

### 5.1 File
`annotation_3d.json` contains a list of frames keyed by timestamp:

```json
[
  {
    "timestamp": 6.08,
    "poses": [
      {
        "id": 10159970873491820000,
        "points_3d": [[x, y, z], [x, y, z], ...],
        "scores": [1.0, 1.0, ...]
      }
    ]
  }
]
```

- `timestamp` — time in seconds, matching the image filename (e.g. `0060.720.jpg` → 60.720 s).
- `points_3d` — `N x 3` array of 3D joint positions in the same world coordinate system as the calibration.
- `scores` — per-joint confidence/visibility.

### 5.2 Skeleton keypoint order (annotation)

The **annotated** poses use **14 keypoints**:

```text
0:  r-ankle
1:  r-knee
2:  r-hip
3:  l-hip
4:  l-knee
5:  l-ankle
6:  r-wrist
7:  r-elbow
8:  r-shoulder
9:  l-shoulder
10: l-elbow
11: l-wrist
12: bottom-head
13: top-head
```

Detections/results in `detection.json` / `result_3d.json` may use a **17 keypoint COCO-like** order (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).  Check the file you consume.

### 5.3 Loading example

```python
import json
import numpy as np

with open('Shelf_Seq1/annotation_3d.json') as f:
    gt3d = json.load(f)

for frame in gt3d:
    ts = frame['timestamp']
    for person in frame['poses']:
        points = np.array(person['points_3d'])  # (J, 3)
        scores = np.array(person['scores'])
```

---

## 6. 2D Counterpart

`annotation_2d.json` follows a similar structure but is organized by image filename:

```json
{
  "image_wh": [360, 288],
  "frames": {
    "Camera0/0060.720.jpg": {
      "camera": "Camera0",
      "timestamp": 60.72,
      "poses": [
        {
          "id": -1,
          "points_2d": [[x, y], ...],
          "scores": [1.0, ...]
        }
      ]
    }
  }
}
```

---

## 7. Preprocessing / Reference Scripts

### 7.1 Cross-View Tracking repo (display + evaluation)
Official scripts for visualization and evaluation:

```bash
# 2D annotation visualization
python display.py \
  --frame-root /data/3DPose_pub/Campus_Seq1/frames \
  --calibration /data/3DPose_pub/Campus_Seq1/calibration.json \
  --pose-file /data/3DPose_pub/Campus_Seq1/annotation_2d.json \
  --pose-type 2d

# 3D annotation visualization
python display.py \
  --frame-root /data/3DPose_pub/Campus_Seq1/frames \
  --calibration /data/3DPose_pub/Campus_Seq1/calibration.json \
  --pose-file /data/3DPose_pub/Campus_Seq1/annotation_3d.json \
  --pose-type 3d

# Evaluation
python evaluate.py \
  --annotation /data/3DPose_pub/Campus_Seq1/annotation_3d.json \
  --result /data/3DPose_pub/Campus_Seq1/result_3d.json
```

Source: [longcw/crossview_3d_pose_tracking](https://github.com/longcw/crossview_3d_pose_tracking)

### 7.2 VoxelPose repo
- Provides pre-pro
