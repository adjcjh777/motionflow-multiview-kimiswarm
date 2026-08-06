# Temporal 3D pose visualization tool

**Task**: Implement a visualization tool that renders 3D pose predictions vs ground
truth across time for the temporal ray-attention model, saving results under
`outputs/visualize_temporal/`.

## Deliverables

- `experiments/visualize_temporal_v1.py`
- `docs/swarm_iter5/temporal_visualization.md` (this file)

## What the script does

`experiments/visualize_temporal_v1.py` loads a contiguous clip from an MPI-INF-3DHP
canonical `.npz` file, runs inference with a trained
`RayAttentionFusionModelTemporal` checkpoint, and writes the following outputs to
`--output_dir` (default `outputs/visualize_temporal`):

1. `frames/frame_{t:04d}.png` — per-frame 3D scatter plots of predicted (red)
   vs ground-truth (blue) poses.
2. `temporal_pose.gif` — animation assembled from the per-frame plots.
3. `joint_trajectories.png` — 3D trajectories of four representative joints over
   the clip (solid = ground truth, dashed = prediction).
4. `mpjpe_time.png` — per-frame mean per-joint position error (MPJPE) in mm.
5. `summary.npz` — small summary with MPJPE, per-frame errors, clip metadata,
   and the checkpoint/baseline used.

If `--checkpoint` points to a non-existent file, the script automatically falls
back to a confidence-weighted DLT baseline so the rendering pipeline can be
validated without a trained checkpoint.

## Usage

```bash
conda run -n mf python experiments/visualize_temporal_v1.py \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/ray_attention_temporal_smoke.pth \
    --start_frame 600 \
    --clip_len 30 \
    --output_dir outputs/visualize_temporal \
    --gif_fps 12
```

Key CLI arguments:

- `--dataset`: canonical `.npz` dataset.
- `--checkpoint`: path to a `RayAttentionFusionModelTemporal` `.pth` checkpoint.
- `--start_frame` / `--clip_len`: clip window.
- `--d` / `--n_temporal_layers`: model hyperparameters (default `64` / `2`).
- `--gif_fps`: frame rate for the output GIF.
- `--output_dir`: where all outputs are saved.

## Design choices

- **Temporal-focused**: unlike the per-frame `visualize_fusion.py`, this tool
  operates on a clip, runs the temporal model, and produces *temporal* outputs
  (GIF + trajectory + error-over-time plots).
- **Minimal and non-intrusive**: a single new file under `experiments/`; no
  existing working files were modified.
- **Fallback DLT baseline**: reuses the project's existing
  `triangulate_dlt_torch` from `motionflow_mv.fusion.triangulation` when no
  checkpoint is available.
- **Headless-safe**: uses `matplotlib.use("Agg")` so it runs on servers without a
  display.
- **Optional GIF dependency**: GIF assembly uses Pillow (already present in the
  `mf` conda environment); the script degrades gracefully if it were missing.
- **No hard-coded skeleton topology**: the MPI-INF-3DHP 28-joint parent map is
  not stored in the canonical `.npz`, so the visualizations show joint
  positions only (scatter points and trajectories) rather than guessing bone
  connections.

## Verification

The script was run end-to-end on the local RTX 4090 in the `mf` conda
environment:

```text
$ conda run -n mf python experiments/visualize_temporal_v1.py \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/ray_attention_temporal_smoke.pth \
    --start_frame 600 --clip_len 30 --output_dir outputs/visualize_temporal
Device: cuda
Loaded clip: 30 frames starting at frame 600
MPJPE = 25.1916 mm | per-frame range: [24.01, 26.55] mm
Per-view weight range: [0.0001, 0.1740]
Saved temporal GIF to outputs\visualize_temporal\temporal_pose.gif
Saved 3D trajectory plot to outputs\visualize_temporal\joint_trajectories.png
Saved MPJPE time-series plot to outputs\visualize_temporal\mpjpe_time.png
Saved summary to outputs\visualize_temporal\summary.npz
All outputs written to outputs\visualize_temporal
```

Generated files (representative):

```text
outputs/visualize_temporal/
├── frames/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
├── temporal_pose.gif
├── joint_trajectories.png
├── mpjpe_time.png
└── summary.npz
```

The 60-frame smoke run (`--start_frame 500 --clip_len 60`) produced a mean MPJPE
of `26.60 mm`, consistent with the model's reported validation performance.

## Notes / next steps

- The visualization currently renders joint *positions* only. If a reliable
  MPI-INF-3DHP 28-joint skeleton parent map is added to the project, the script
  can be extended to draw bones (reusing the `PARENTS` + `_draw_bones` pattern
  from `experiments/visualize_fusion.py`).
- Consider adding an option to overlay the 3D pose on real camera frames for
  qualitative video results.
- A `--joint_parent_json` flag could make the script reusable across datasets
  with different skeleton conventions.
