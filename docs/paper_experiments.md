# Paper-Ready Experiments Roadmap

**Target venues**: ICRA / CVPR 2027  
**Goal**: demonstrate that MotionFlow-MultiView sets a new state of the art for multi-view 3D human pose estimation under realistic, variable-view conditions.

## 1. Datasets

| Dataset | Skeleton | Views | Use |
|---------|----------|-------|-----|
| MPI-INF-3DHP | 28 joints | 4–14 | Main benchmark (train/val/test) |
| H36M | 17 joints | 4 | In-the-lab generalisation |
| 3DPW | 24 joints | 2–4 | In-the-wild generalisation |
| AIST++ | 17 joints | 4–9 | Dance / motion-rich generalisation |

*WebBridge canonical `.npz` files are loaded via `configs/splits/webbridge_*.yaml`.

## 2. Main Architecture

- **Baseline**: single-view / dense multi-view pose network.
- **v2**: `OmniMultiViewFusionV2` – visibility gating, graph-joint attention, uncertainty-weighted triangulation, spatio-temporal transformer.
- **v3**: `OmniMultiViewFusionV3` – v2 + hierarchical multi-scale fusion + camera-conditioned epipolar-biased attention.

## 3. Metrics

- **MPJPE** (mm): mean per-joint position error.
- **PA-MPJPE** (mm): Procrustes-aligned MPJPE.
- **Per-joint error maps**: identify which joints / views benefit most.

## 4. Ablation Matrix

| Run | Model | Graph | Multi-scale | Camera cond | Epipolar bias | Dataset | Expected question |
|-----|-------|-------|-------------|-------------|---------------|---------|-------------------|
| A   | v2    | no    | no          | no          | no            | MPI  | Baseline |
| B   | v2    | yes   | no          | no          | no            | MPI  | Value of graph-joint attention |
| C   | v3    | yes   | yes         | yes         | yes           | MPI  | Full model |
| D   | v3    | yes   | no          | yes         | yes           | MPI  | Impact of multi-scale |
| E   | v3    | yes   | yes         | no          | yes           | MPI  | Impact of camera conditioning |
| F   | v3    | yes   | yes         | yes         | no            | MPI  | Impact of epipolar bias |
| G   | v2    | yes   | no          | no          | no            | H36M | Cross-dataset transfer |
| H   | v3    | yes   | yes         | yes         | yes           | H36M | Cross-dataset transfer |

## 5. Robustness Tests

- **Variable views**: 2, 3, ..., V visible views.
- **Camera perturbation**: rotation, translation, focal length, principal-point noise.
- **Missing/occluded views**: confidence dropout.
- **Cross-dataset**: train MPI, test H36M / 3DPW / AIST++.

## 6. Test-Set Protocol

- Train on official train split.
- Select best val checkpoint.
- Run inference on official test set.
- Report MPJPE / PA-MPJPE, per-sequence, per-joint, and per-view-count.

## 7. Current Status

- No-graph v2 ablation: 25.18 mm MPJPE / 23.99 mm PA-MPJPE on MPI-INF-3DHP val (full 14 views).
- Dense+graph v2 (freeze phase): val_MPJPE 25.13–25.35 mm.
- v3 prototype implemented; smoke tests pass.
- Multi-dataset loader and robustness scripts smoke-tested.

## 8. Next Steps

1. Finish dense+graph v2 end-to-end on 4090.
2. Launch v2 full run on A800.
3. Launch v3 ablation matrix on A800.
4. Collect test-set numbers and fill the ablation table.
