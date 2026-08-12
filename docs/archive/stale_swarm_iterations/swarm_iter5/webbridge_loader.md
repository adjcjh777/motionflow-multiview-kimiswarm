# WebBridge Canonical Loader (swarm iter 5)

## Deliverable

`motionflow_mv/data/webbridge_loader.py` – a single module that converts each
supported WebBridge dataset into the canonical multi-view `.npz` format used by
the `ray_attention` training/evaluation pipeline.

## Canonical `.npz` format

```text
points_2d:   (T, V, J, 2)  per-view 2D keypoint detections
confidences: (T, V, J)     detection confidence / visibility
joints_3d:   (T, J, 3)    3D ground-truth joint positions
camera_K:    (V, 3, 3)    intrinsic calibration matrices
camera_R:    (V, 3, 3)    rotation (world-to-camera)
camera_t:    (V, 3)       translation (world-to-camera)
```

## Supported converters

| Dataset          | Function                  | Status      | Required assets                                                   |
|------------------|---------------------------|-------------|-------------------------------------------------------------------|
| Human3.6M      | `convert_human36m`        | implemented | `camera_params.json`, `h36m_sh_conf_cam_source_final.pkl.zip`     |
| Shelf/Campus     | `convert_shelf_campus`    | implemented | `calibration.json`, `annotation_3d.json`                        |
| Synthetic/AMASS  | `convert_synthetic_amass` | implemented | `smplx`, `data/smpl/SMPL_NEUTRAL.pkl` (calls existing generator) |
| CMU Panoptic     | `convert_panoptic`        | stub        | raw dataset not yet present                                       |
| 3DPW             | `convert_3dpw`            | stub        | raw dataset not yet present                                       |

## Usage

```python
from pathlib import Path
from motionflow_mv.data.webbridge_loader import convert_human36m

out = convert_human36m(
    data_root=Path("data/h36m_hf"),
    subject=1,
    actions=[2, 3, 4],
    split="train",
    out_dir=Path("data/h36m_hf"),
)
```

CLI:

```bash
python motionflow_mv/data/webbridge_loader.py human36m \
    --data_root data/h36m_hf \
    --out data/h36m_hf/s_01_acts_02_03_04_multiview.npz \
    --subject 1 --actions 2 3 4
```

## Important findings

- The loader reuses existing project components:
  - `motionflow_mv.calibration.camera.Camera`
  - `motionflow_mv.fusion.triangulation.triangulate_dlt`
  - `motionflow_mv.data.shelf_loader.build_shelf_dataset`
  - `experiments/generate_synthetic_multiview_dataset.py`
- The H36M conversion logic mirrors `experiments/prepare_h36m_multiview.py` but is
  wrapped as a reusable function in the `motionflow_mv` package.
- Runtime verification is currently blocked by a Windows fatal exception
  (`code 0xc06d007f`) inside `numpy.linalg`/BLAS when the `jz_py310` environment
  is invoked from the Git Bash shell. The same exception occurs with existing
  project tests (`tests/test_triangulation.py`). Non-BLAS parts (file I/O,
  `.npz` writing) were verified successfully.
