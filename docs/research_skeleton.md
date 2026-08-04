# Project Skeleton Proposal: motionflow-multiview-kimiswarm

> Minimal layout, modules, dependency file, and first-PR plan for extending a monocular MotionFlow pipeline to multi-view human-motion fusion.

## 1. Baseline assumption

The existing **MotionFlow** baseline is treated as an external black-box module that consumes a monocular video and produces per-frame 2D/3D human pose (exact checkpoint/output format will be pinned in `docs/research_motionflow.md`).
We wrap it rather than vendoring it, so the skeleton stays decoupled from baseline internals and license details.

## 2. Design principles

- **No over-engineering**: start with a deterministic fusion step (DLT triangulation) and add learned components only after validation.
- **Single command**: one script runs the full multi-view inference end-to-end.
- **Tested from day one**: every non-trivial module has a matching unit test.
- **Clear interfaces**: the baseline wrapper exposes a small, stable API so the fusion module does not depend on baseline internals.

## 3. Directory layout

```text
motionflow-multiview-kimiswarm/
├── motionflow_mv/              # core package
│   ├── __init__.py
│   ├── baseline/
│   │   ├── __init__.py
│   │   └── wrapper.py          # MotionFlowBaseline interface
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── camera.py           # intrinsics / extrinsics helpers
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── triangulator.py     # DLT + confidence-weighted triangulation
│   ├── pipeline.py             # end-to-end multi-view inference
│   └── utils/
│       ├── __init__.py
│       └── io.py               # video/keypoint/calib I/O
├── experiments/
│   ├── config.yaml
│   └── run_multiview.py        # CLI entry point
├── tests/
│   ├── test_baseline_wrapper.py
│   ├── test_camera.py
│   ├── test_triangulator.py
│   └── test_pipeline.py
├── docs/                       # research notes (this file, etc.)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 4. Core modules and APIs

### `motionflow_mv.baseline.wrapper.MotionFlowBaseline`

```python
class MotionFlowBaseline:
    def __init__(self, config_path: str, device: str = "cuda"): ...
    def infer(self, video_path: str) -> dict:
        """
        Returns {
            "keypoints_2d": np.ndarray,   # (T, J, 3)  x, y, confidence
            "keypoints_3d": np.ndarray,   # (T, J, 3)  optional camera-relative 3D
        }
        """
```

### `motionflow_mv.calibration.camera.Camera`

Stores intrinsics `K`, distortion `D`, and extrinsics `R, t`.
Provides `project(points_3d) -> points_2d` and `pose_in_world` helpers.

### `motionflow_mv.fusion.triangulator.DLTTriangulator`

```python
class DLTTriangulator:
    def triangulate(self, points_2d: dict[str, np.ndarray],
                    cameras: dict[str, Camera],
                    confidences: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """
        points_2d: view_name -> (T, J, 2)
        cameras:   view_name -> Camera
        confidences: optional view_name -> (T, J) for weighted DLT
        returns: (T, J, 3) in the common world frame
        """
```

### `motionflow_mv.pipeline.MultiViewPipeline`

Orchestrates: per-view baseline inference → optional temporal smoothing → multi-view triangulation → output `.npy`/`.mp4`.

## 5. Dependency file

`requirements.txt` (start minimal; add only when needed):

```text
numpy>=1.23
opencv-python>=4.7
pyyaml
torch>=2.0       # pin to whatever MotionFlow requires
pytest
```

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "motionflow_mv"
version = "0.1.0"
dependencies = [
    "numpy>=1.23",
    "opencv-python>=4.7",
    "pyyaml",
    "torch>=2.0",
]

[project.optional-dependencies]
dev = ["pytest"]
```

## 6. First PR content

**Title**: `feat: minimal multi-view fusion skeleton`

**What it adds**:

1. `motionflow_mv/baseline/wrapper.py` – abstract/initial wrapper around MotionFlow.
2. `motionflow_mv/calibration/camera.py` – camera model and parameter loading.
3. `motionflow_mv/fusion/triangulator.py` – DLT triangulation with optional confidence weighting.
4. `motionflow_mv/pipeline.py` + `experiments/run_multiview.py` – end-to-end CLI.
5. `tests/test_*.py` – unit tests for camera, triangulator, and pipeline with synthetic data.
6. `requirements.txt` / `pyproject.toml` and a short usage section in `README.md`.

**What it explicitly does NOT add**: a trained fusion network, real dataset loaders, or any vendored baseline code.

## 7. Usage sketch

```bash
# 1. Run baseline on each view
python -m motionflow_mv.baseline.wrapper --video view1.mp4 --out view1_pose.npz

# 2. Triangulate multi-view pose
python experiments/run_multiview.py \
    --videos view1.mp4 view2.mp4 view3.mp4 \
    --calib calib.json \
    --output output_3d.npy
```

## 8. References

- Project repo: https://github.com/adjcjh777/motionflow-multiview-kimiswarm
- Iskakov et al., "Learnable Triangulation of Human Pose", ICCV 2019. Code: https://github.com/karfly/learnable-triangulation-pytorch
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2004. (DLT triangulation)
- OpenCV calib3d: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
