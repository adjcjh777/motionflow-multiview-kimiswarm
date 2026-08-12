# H36M True-GT Converter v2

## Problem

The existing `data/h36m_hf/*.npz` files have **circular labels**: `joints_3d` is the DLT triangulation of the input `points_2d`, so `scripts/diagnose_circular_labels.py` reports `direct MJE = 0.0000 mm`.

The `data/h36m_true_gt/*_multiview.npz` files (millimetre convention) also contain a coordinate/unit mismatch: the diagnostic reports `direct MJE ≈ 16668 mm`, i.e. the stored 3D does not match the stored cameras/2D.

The `data/h36m_true_gt/*_multiview_m.npz` files (metre convention) are internally consistent, but they are scattered across subjects and mixed with the broken non-`_m` files. A single reproducible converter is needed.

## Fix: `scripts/convert_h36m_true_gt_v2.py`

The new converter builds canonical `.npz` files from the official mocap release while keeping the real 2D detections from the preprocessed archive.

What it does:

1. Loads the official 3D mocap ground truth from `data/h36m_true_gt/data_3d_h36m.npz` (VideoPose3D/MHFormer release).
2. Uses the 17-joint H36M subset `[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]`.
3. Aligns the mocap frames to the frame order of `data/h36m_hf/h36m_sh_conf_cam_source_final.pkl.zip` (via reprojection-based disambiguation when multiple subactions exist).
4. Uses the official camera parameters from `data/h36m_hf/camera_params.json`.
5. Keeps the preprocessed 2D keypoints and confidences from the pkl archive (realistic detector inputs).
6. Stores `joints_3d` = true mocap 3D, `points_2d` = pkl 2D, `confidences` = pkl confidences, `camera_K/R/t`.
7. Outputs in **meters** (matching the current `_m` convention used by training configs).
8. Writes to a fresh directory `data/h36m_true_gt_v2/` so existing files are untouched.

## Verification

```bash
python scripts/diagnose_circular_labels.py data/h36m_true_gt_v2/s_01_act_02_multiview.npz
```

Result:

```
data\h36m_true_gt_v2\s_01_act_02_multiview.npz
  frames=2995, views=4, joints=17
  direct MJE (no root align): 14.5293 mm
  root-aligned MPJPE:       14.4353 mm
  max per-joint error:      33.6402 mm
  median per-joint error:   10.5238 mm
  mean per-joint error:     14.5293 mm
```

The direct MJE is non-zero and reasonable (~15 mm), reflecting detector/reprojection noise. It is neither 0 mm (circular) nor ~16 m (unit/coordinate mismatch).

## How to regenerate all subjects/actions after v85 finishes

Train subjects (S1, S5, S6, S7, S8), all actions 2-16, split `train`:

```bash
for s in 1 5 6 7 8; do
  python scripts/convert_h36m_true_gt_v2.py --subject $s --actions $(seq 2 16) --split train --out_dir data/h36m_true_gt_v2
done
```

Test subjects (S9, S11), all actions 2-16, split `test`:

```bash
for s in 9 11; do
  python scripts/convert_h36m_true_gt_v2.py --subject $s --actions $(seq 2 16) --split test --out_dir data/h36m_true_gt_v2
done
```

After generation, rename the resulting per-action files as needed (the converter currently emits `s_*_acts_*_multiview.npz` for both single and multiple actions).

## Files touched

- `scripts/convert_h36m_true_gt_v2.py` — new converter
- `data/h36m_true_gt_v2/s_01_act_02_multiview.npz` — test file (2995 frames, 4 views, 17 joints)
- `docs/h36m_true_gt_converter_fix_v2.md` — this note
