# Focused 5-Agent Synthesis: ScoreHMR Integration & Dataset Access

## WebBridge / Dataset Access

- **WebBridge is not a known multi-view human motion dataset.** It appears to be the Kimi browser-automation tool, not a data source. Do not plan around it.
- **Primary real 3D-GT targets:** Human3.6M (registration required), CMU Panoptic, 3DPW, AMASS.
- **Fastest unblock:** use synthetic multi-view generation and/or ScoreHMR pseudo-labels while waiting for Human3.6M approval.

## ScoreHMR Integration

- Repository: https://github.com/statho/ScoreHMR (MIT license)
- Output is **camera-relative SMPL params**, not world-metric. Must fuse multiple views to recover world coordinates.
- Best used as a **strong but slower per-view plugin** for offline pseudo-3D labeling; keep GVHMR as the fast default.
- RTX 4090 can run single-view inference; A800-D is better for batch processing.
- Main blockers: SMPL model download, PyTorch CUDA version matching, heavy runtime vs. GVHMR.

## AMASS Synthetic Multi-View Generation

- Load AMASS `.npz` clips, forward through `smplx`, project joints through virtual camera rigs, add noise/occlusion.
- Output matches `FusionModule.fuse(points_2d, confidences, cameras)` contract.
- Blockers: AMASS/SMPL registration, skeleton mismatch (24 vs. 17 joints), occlusion realism.
- Simpler fallback: generate random 3D skeletons with kinematic constraints (no AMASS dependency).

## Human3.6M

- Requires free academic registration at vision.imar.ro/human3.6m.
- Use `anibali/h36m-fetch` and `karfly/learnable-triangulation-pytorch` preprocessing scripts.
- ~200 GB disk usage; plan storage on A800-D `/mnt/nvme0n1` or local external SSD.
- Loader target: `(points_2d, confidences, proj_matrices, joints_3d_gt)` in meters.

## A800-D Read-Only Audit

- Likely assets: Shelf/Campus, Human3.6M, CMU Panoptic, 3DPW, AMASS, ScoreHMR data/checkpoints, MotionFlow Docker service.
- Safe audit commands provided in the raw report; require SSH access.
- Read-only manifest via `rsync -avn --list-only` is recommended before copying.

## Recommended Next Steps

1. Implement `motionflow_mv/ir/scorehmr_adapter.py` skeleton.
2. Implement a synthetic multi-view generator (AMASS-free) for immediate 3D-supervised training.
3. Apply for Human3.6M registration.
4. Run the A800-D read-only audit to discover existing data.
