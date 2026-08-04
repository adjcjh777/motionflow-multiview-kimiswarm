# Residual Correction Visualization – Swarm Iter 6

## Goal
Create per-joint visualisations of the residual refinement head: for each joint, show the raw DLT triangulated pose, the residual-corrected pose, and the ground-truth 3D pose over time.

## Files added / changed
- `motionflow_mv/fusion/ray_attention_temporal_residual_model_v3.py`  
  New subclass `RayAttentionFusionModelTemporalResidualV3` that extends `RayAttentionFusionModelTemporalResidual` and exposes the raw DLT pose via a `return_raw=True` flag.
- `experiments/visualize_residual_corrections_v1.py`  
  Stand-alone visualisation script. Loads the residual checkpoint, runs inference on a selected clip of MPI-INF-3DHP, and saves:
  - One figure per joint with a 3D trajectory plot and X/Y/Z coordinate time-series.
  - A summary bar chart of per-joint MPJPE for raw vs. residual-corrected poses.
- `docs/swarm_iter6/residual_visualization_report.md`  
  This report.

## How to reproduce
```bash
conda run -n mf python experiments/visualize_residual_corrections_v1.py \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
    --output_dir outputs/visualize_residual \
    --num_frames 100
```

Optional flags:
- `--num_frames` / `--start_frame`: select the clip to visualise.
- `--d`, `--n_temporal_layers`, `--residual_hidden`: model hyper-parameters (default to those of the v2 checkpoint).

## Results
Ran the script on the first 100 frames of MPI-INF-3DHP validation subject 2, sequence 1 (14 views, 28 joints).

```
Frames 0:100 | raw MPJPE = 29.66 mm | refined MPJPE = 12.05 mm
```
(The refined value can vary by ~2 mm across GPU runs due to transformer non-determinism; raw and corrected trajectories remain visually identical.)

Generated outputs under `outputs/visualize_residual/`:
- `summary_per_joint_mpjpe.png` – per-joint error comparison (raw DLT vs. residual-corrected).
- `joint_00.png` … `joint_27.png` – one figure per joint showing 3D trajectory + X/Y/Z coordinates over time for raw DLT, residual-corrected, and ground truth.

## Interpretation
The residual head consistently reduces the per-joint error, with the corrected trajectories lying much closer to ground truth than the raw DLT trajectories. The per-joint plots make it easy to identify which joints receive the largest corrections and whether the residual head under- or over-corrects at particular frames.

## Blockers
None. The script runs end-to-end and produces the requested visualisations.
