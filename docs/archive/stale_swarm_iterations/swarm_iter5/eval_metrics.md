# Evaluation metrics and protocol (swarm task)

## What changed

1. `motionflow_mv/eval/metrics.py`
   - Kept the existing scalar helpers (`mpjpe`, `pa_mpjpe`, `pck`) unchanged for backward compatibility.
   - Added batched and per-joint/per-view variants:
     - `mpjpe_batch(pred, gt)`
     - `per_joint_mpjpe(pred, gt)`
     - `per_view_mpjpe(pred, gt)`
     - `pa_mpjpe` now supports batched inputs
     - `pa_mpjpe_per_joint(pred, gt)`
     - `pck_batch(pred, gt, threshold)`
     - `pck_per_joint(pred, gt, threshold)`
     - `pck_auc(pred, gt, ...)` with optional per-joint AUC
     - `compute_all_metrics(pred, gt, ...)` returns a full report including MPJPE, PA-MPJPE, PCK@50/100/150mm, PCK-AUC, and per-joint breakdowns.
     - `summarize_metrics(report)` for compact printing.

2. `motionflow_mv/eval/__init__.py`
   - Exported the new metrics.

3. `experiments/eval_all_datasets.py`
   - Single entry point for evaluating DLT baseline (and optionally a learned model) on:
     - H36M multi-view `.npz` files
     - Synthetic generated sequence (fallback when no file exists)
     - Shelf / Campus (reprojection error only, because 3D GT is not available locally)
   - Reports MPJPE, PA-MPJPE, PCK@50/100/150mm, PCK-AUC and per-joint/per-view breakdowns where 3D GT exists.

## Verification

Run a quick synthetic-only test:

```bash
python experiments/eval_all_datasets.py --synthetic --max_frames 50
```

Run on a small H36M clip:

```bash
python experiments/eval_all_datasets.py \
    --h36m data/h36m_hf/s_01_act_02_multiview.npz \
    --max_frames 100
```

Run with a learned checkpoint:

```bash
python experiments/eval_all_datasets.py \
    --h36m data/h36m_hf/s_01_act_02_multiview.npz \
    --checkpoint outputs/ray_attention_v2_h36m_s1a2.pth \
    --model ray_attention_v2 --n_views 4 --joints 17
```

## Important findings

- DLT baseline on the small synthetic sequence with 0.5 px noise is near-perfect, as expected.
- H36M data uses millimeters internally, so all reported metric values are in mm.
- Shelf/Campus only provide 2D predictions and calibration locally; the script therefore reports reprojection error rather than 3D error. Per-view and per-joint reprojection breakdowns are provided.
- The protocol is intentionally simple: it loads `.npz` files directly, uses the existing `triangulate_dlt` baseline, and calls the new metrics. No training is started.
