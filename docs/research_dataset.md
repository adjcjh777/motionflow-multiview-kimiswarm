# Dataset & Metrics Recommendation

## 1. Primary recommendation: Shelf + Campus (small, calibrated, multi-view)

For fast iteration on an RTX 4090 / A800, the **Shelf** and **Campus** datasets are the best starting point. They are small, public, have calibrated multi-camera rigs, and are standard benchmarks for multi-view 3D human pose estimation.

### Shelf Dataset
- **Paper**: Belagiannis et al., "3D Pictorial Structures for Multiple Human Pose Estimation," CVPR 2014. [https://cvn.ecp.fr/person/452/1868.pdf](https://cvn.ecp.fr/person/452/1868.pdf)
- **Size**: ~3,000 frames, 4 calibrated cameras, 2 actors.
- **Annotations**: 3D ground-truth joint positions for a fixed set of joints.
- **Why use it**: Very small, so a full training/evaluation cycle fits in minutes on a 4090 or A800. Widely used by VoxelPose, Learnable Triangulation, and other multi-view baselines.

### Campus Dataset
- **Paper**: A subset derived from the same 3D pictorial structures line of work (Belagiannis et al., CVPR 2014).
- **Size**: ~2,000 frames, 3 calibrated cameras, 3 actors.
- **Annotations**: 3D ground-truth joint positions.
- **Why use it**: Slightly more actors and views than Shelf, complementary scenarios, same small scale.

### Download sources
- Shelf/Campus data is often mirrored in multi-view pose repositories:
  - `https://github.com/microsoft/voxelpose` (VoxelPose repo contains data links and preprocessing scripts)
  - `https://github.com/zju3dv/mvpose` (multi-view pose evaluation utilities)
- Search term: "Shelf dataset multi view human pose download".

---

## 2. Secondary / scale option: Human3.6M subset

- **Paper**: Ionescu et al., "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments," TPAMI 2014.
- **Size**: 3.6 million video frames, 4 views, 11 actors.
- **Why not first choice**: Too large for quick iteration. Use only a **small subset** (e.g., one action, one subject, downsampled frames) when you need a larger sanity check after validating on Shelf/Campus.

---

## 3. Preprocessing pipeline (minimal)

1. **Download**: images/frames + camera calibration files + 3D GT annotations.
2. **Extract frames** if videos are provided (e.g., `ffmpeg -i vid.mp4 frame_%04d.jpg`).
3. **Parse calibration**: load intrinsics `K`, distortion `D`, and extrinsics `R, t` for each camera.
4. **Synchronize frames**: use provided frame indices so all views share the same timestamp.
5. **Run per-view estimator**: feed each view into MotionFlow (or the chosen baseline) to get 2D/3D joints per view.
6. **Transform to common skeleton**: map baseline output joints to the dataset's joint set (e.g., 17 COCO-Human joints or 14/15 H36M-style joints).
7. **World-space fusion**: triangulate / fuse per-view predictions using calibration into the dataset world coordinate system.
8. **Cache**: store per-view 2D/3D predictions and fused 3D results as `.npz` or `.pkl` for fast reloading.

Recommended cache layout:
```
data/
├── shelf/
│   ├── calibration/      # camera params
│   ├── images/           # synced frames
│   ├── annotations/      # 3D GT
│   └── cache/            # predictions + fused poses
```

---

## 4. Evaluation metrics

Use the standard 3D pose metrics below. Report **MPJPE** and **PA-MPJPE** as the main numbers; add PCK/AUC for completeness.

| Metric | Description | Use case |
|--------|-------------|----------|
| **MPJPE** | Mean Per Joint Position Error (mm) between predicted and GT 3D joints after rigid alignment (root at pelvis). | Primary accuracy metric. |
| **PA-MPJPE** | MPJPE after Procrustes analysis (scale, rotation, translation). | Removes global pose ambiguity; common for monocular methods. |
| **PCK** | Percentage of Correct Keypoints at a threshold (e.g., 150 mm). | Good for comparing across datasets. |
| **AUC** | Area Under the Curve of PCK over a range of thresholds. | Single scalar summary of localization quality. |
| **Per-joint error** | MPJPE broken down by joint. | Diagnose which body parts are hardest. |

### Reference implementations
- `mpii_human_pose_eval` and many repos have Python implementations.
- VoxelPose / MvPose repos include PCK/AUC code for Shelf/Campus.

---

## 5. Fast-iteration checklist

- [ ] Download Shelf (and optionally Campus).
- [ ] Verify camera calibration loads correctly and triangulates GT 3D points.
- [ ] Run the MotionFlow baseline on a 30–60 frame clip first; confirm output format.
- [ ] Convert baseline joints to dataset skeleton.
- [ ] Compute MPJPE / PA-MPJPE on the same test split used by VoxelPose / MvPose for comparability.
- [ ] Log per-frame inference time on RTX 4090 and A800.

---

## 6. Key references

1. Belagiannis et al., "3D Pictorial Structures for Multiple Human Pose Estimation," CVPR 2014. [paper](https://cvn.ecp.fr/person/452/1868.pdf)
2. Ionescu et al., "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments," TPAMI 2014. [project](http://vision.imar.ro/human3.6m/)
3. Tu et al., "VoxelPose: Towards Multi-camera 3D Human Pose Estimation in Wild Environment," CVPR 2020. [code](https://github.com/microsoft/voxelpose)
