# AIST++ → Canonical WebBridge `.npz` Conversion

**Date:** 2026-08-04  
**Scope:** Download AIST++ multi-view dance annotations and convert them into the project's canonical multi-view `.npz` format so they can be consumed by the existing `RayAttentionFusionModelTemporal` training pipeline.

## Summary

- Implemented `motionflow_mv/data/webbridge_loader.py::convert_aistpp`.
- Added `scripts/download_aistpp.py` to fetch the official annotation archives.
- Added `experiments/convert_aistpp_v1.py` to drive the conversion.
- Downloaded the four annotation archives from the AIST++ GitHub release and extracted them under `data/webbridge/aistpp`.
- Converted all 1,408 AIST++ sequences to canonical `.npz` files.
- Verified shapes, reprojection, forward pass, and a 2-epoch smoke training run.

## Files Added / Modified

- `motionflow_mv/data/webbridge_loader.py`
  - New helpers: `_rodrigues_to_matrix`, `_load_aistpp_cameras`, `_parse_aistpp_mapping`.
  - New converter: `convert_aistpp(...)`.
  - CLI: added `aistpp` choice with `--split_file`, `--max_seqs`, `--scale_factor`, `--meters`.
- `scripts/download_aistpp.py` — downloads `keypoints2d.zip`, `keypoints3d.zip`, `motions.zip`, `cameras.zip` from the official AIST++ release and extracts them.
- `experiments/convert_aistpp_v1.py` — wrapper script to convert all or a subset of AIST++ sequences.

## AIST++ Annotation Layout

After download and extraction (`data/webbridge/aistpp`):

```text
data/webbridge/aistpp/
  keypoints2d/<seq>.pkl   # (V, T, 17, 3)  last dim = (x, y, confidence)
  keypoints3d/<seq>.pkl   # (T, 17, 3)     COCO 17 3D keypoints
  motions/<seq>.pkl       # SMPL motion parameters (not used for canonical format)
  cameras/
    mapping.txt             # seq_name -> setting_name
    setting*.json           # 9-camera rigs (intrinsics + Rodrigues extrinsics)
```

## Canonical `.npz` Output

Per sequence, one file named `<seq_name>_multiview.npz` containing:

```text
points_2d:   (T, V, J, 2)
confidences: (T, V, J)
joints_3d:   (T, J, 3)
camera_K:    (V, 3, 3)
camera_R:    (V, 3, 3)
camera_t:    (V, 3)
```

For AIST++: T = 720 (60 fps × 12 s), V = 9, J = 17 (COCO).

## Units

AIST++ raw 3D keypoints and camera translations are stored in **centimeters**. The converter defaults to `scale_factor=0.01`, so the canonical arrays are written in **meters** to match the MPI-INF-3DHP `_m` convention. Use `--raw` or `scale_factor=None` to keep centimeters.

Quick sanity check on the first sequence (head-to-hip distance ≈ 0.67 m) confirms the metric scaling is sensible.

## Verification

### Shape & Reprojection Check

```bash
conda run -n mf python - <<'PY'
import numpy as np
from pathlib import Path
from motionflow_mv.calibration.camera import Camera

p = Path('data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch01_multiview.npz')
data = np.load(p)
print(data['points_2d'].shape)   # (720, 9, 17, 2)
print(data['confidences'].shape)  # (720, 9, 17)
print(data['joints_3d'].shape)    # (720, 17, 3)

cam = Camera(K=data['camera_K'][0], R=data['camera_R'][0], t=data['camera_t'][0])
X = data['joints_3d'][0]
P = cam.projection_matrix
Xh = np.hstack([X, np.ones((17, 1))])
x = (P @ Xh.T).T
x = x[:, :2] / x[:, 2:]
err = np.linalg.norm(x - data['points_2d'][0, 0], axis=-1).mean()
print(f"Mean reprojection error: {err:.2f} px")
PY
```

Result:

```text
(720, 9, 17, 2)
(720, 9, 17)
(720, 17, 3)
Mean reprojection error: 3.26 px
```

### Forward Pass

```bash
conda run -n mf python - <<'PY'
import numpy as np
import torch
from pathlib import Path
from motionflow_mv.fusion.ray_attention_temporal_model import RayAttentionFusionModelTemporal

p = Path('data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch01_multiview.npz')
data = np.load(p)
clip = torch.from_numpy(np.concatenate([
    data['points_2d'][:13],
    data['confidences'][:13, ..., None],
], axis=-1)).float().unsqueeze(0)
K = torch.from_numpy(data['camera_K']).float()
R = torch.from_numpy(data['camera_R']).float()
t = torch.from_numpy(data['camera_t']).float()
model = RayAttentionFusionModelTemporal(j=17, d=64, n_views=9, n_heads=4,
                                        n_joint_layers=1, n_temporal_layers=2)
with torch.no_grad():
    X, w = model(clip, K=K, R=R, t=t)
print(X.shape, w.shape)  # torch.Size([1, 13, 17, 3]) torch.Size([1, 13, 9, 17])
PY
```

Succeeds with the expected output shapes.

### Smoke Training

```bash
conda run -n mf python experiments/train_ray_attention_temporal_mpiinf3dhp.py \
  --train data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch01_multiview.npz \
         data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch02_multiview.npz \
  --val data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch03_multiview.npz \
  --clip_len 13 --epochs 2 --batch_size 8 --train_samples 500 \
  --output outputs/ray_attention_temporal_aistpp_smoke.pth
```

Result:

```text
Device: cuda
n_views=9, j=17, clip_len=13, d=64
Model params: 197345
Epoch 1: train_loss=0.000906, val_MPJPE=15.39mm (saved)
Epoch 2: train_loss=0.000174, val_MPJPE=22.11mm
Best val MPJPE: 15.39mm -> outputs\ray_attention_temporal_aistpp_smoke.pth
```

The model trains on AIST++ without modification.

## Known Limitations / Notes

- AIST++ contains 1,408 sequences with **13 different camera rigs** (`setting*.json`). The current temporal training script handles per-sample rigs, but downstream mixed-dataset training should account for the varying camera setups (e.g., per-sequence rig or dataset-specific normalization).
- The 2D keypoints come from an off-the-shelf detector and therefore have detection noise; the reprojection error of ~3 px reflects this.
- No new Python dependencies were required beyond what is already in the `mf` environment (`numpy`, `scipy`, `torch`).

## Dataset Size

- Raw AIST++ annotations: ~5.5 GB (`data/webbridge/aistpp`)
- Canonical `.npz` outputs: ~1.5 GB (`data/webbridge/aistpp_canonical`, 1,408 files)

## Next Steps / Follow-ups

- Optionally generate train/val/test split files based on AIST++ genres for reproducible experiments.
- Consider a small per-genre validation hold-out for cross-genre evaluation.
