# Visualization toolkit for ray_attention_v3 fusion

**Task**: Add `experiments/visualize_fusion.py` that renders multi-view 2D
reprojections, 3D skeletons, and attention-weight heatmaps for the current
best fusion model (`motionflow_mv/fusion/ray_attention_v3_model.py`).

## Deliverable

- `experiments/visualize_fusion.py`
- `docs/swarm_iter5/visualize_fusion.md` (this file)

## What the script does

`experiments/visualize_fusion.py` loads a single frame from an H36M-style
`.npz` multi-view dataset and an optional `ray_attention_v3` checkpoint, then
produces three figures in `--output_dir`:

1. `frame_{idx:05d}_multi_view_2d.png` — per-view 2D overlay of input
   keypoints (blue) and the predicted 3D skeleton reprojected into each camera
   (red). Bones are drawn using a standard 17-joint H36M parent map.
2. `frame_{idx:05d}_skeleton_3d.png` — interactive-style 3D plot showing the
   predicted skeleton (red) and ground-truth skeleton (blue).
3. `frame_{idx:05d}_attention_heatmap.png` — per-view per-joint fusion weight
   heatmap from the model.

## Usage

```bash
python experiments/visualize_fusion.py \\
    --dataset data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz \\
    --checkpoint outputs/ray_attention_v3_h36m.pth \\
    --frame 0 \\
    --output_dir outputs/visualize_fusion
```

If `--checkpoint` is omitted, the script falls back to a confidence-weighted
DLT baseline and uses the input confidences as weights. This lets the
rendering pipeline be validated even before a trained checkpoint exists.

## Design choices

- **Minimal, self-contained**: all rendering helpers live in the same file.
- **Matches project style**: uses `Path`, `argparse`, `sys.path.insert`, and
  the existing `Camera` and `RayAttentionFusionModelV3` classes.
- **H36M 17-joint skeleton**: `JOINT_NAMES` and `PARENTS` are hard-coded to
  the karfly-preprocessed H36M joint order. A different skeleton can be
  visualized by editing these constants.
- **Non-interactive backend**: each plot function sets
  `matplotlib.use("Agg")` so figures can be saved on headless machines.
- **Torch-based DLT fallback**: the fallback triangulation uses
  `torch.linalg.svd` rather than `np.linalg.svd` because the Windows conda
  environment used for verification crashes during numpy matrix operations
  (exit 127, likely a BLAS/MKL packaging issue). The torch path is robust in
  that environment and produces equivalent results.

## Verification

- `python -c "import experiments.visualize_fusion"` succeeds.
- `python -m py_compile experiments/visualize_fusion.py` succeeds.
- Inference path verified: with a dummy `ray_attention_v3` checkpoint the
  script loads the H36M `.npz`, runs the model forward, and prints
  `MPJPE(pred, gt)` and the predicted weight range.
- The figures were not generated in this workspace because the active Python
  environment crashes inside `matplotlib.figure.Figure.savefig` (exit code
  127, likely a missing/ incompatible Windows DLL). All rendering logic was
  inspected by hand and uses only standard matplotlib/pyplot calls with the
  `Agg` backend.
- A full end-to-end run is expected to succeed in a clean Linux/WSL conda or
  pip environment with working numpy and matplotlib.

## Next steps / follow-up

- Run the script end-to-end once a trained `ray_attention_v3_h36m.pth`
  checkpoint is available.
- Extend the script to optionally overlay 2D keypoints on real camera frames
  by supplying a directory of per-view images.
- Add an optional `--joint_parent_json` flag so the same script can render
  other skeleton conventions (Shelf/Campus 14-joint, COCO, etc.).
