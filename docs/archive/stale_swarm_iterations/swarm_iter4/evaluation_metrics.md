# Evaluation Metrics (MPJPE, PA-MPJPE, Acceleration, Perceptual, AUC)

## TL;DR

`motionflow_mv/eval/metrics.py` already has raw MPJPE, PA-MPJPE, and PCK, and `train_ray_attention_real.py` logs a validation MPJPE. However, real-data evaluation still reports 2D reprojection error, and the repo lacks root-centering, AUC, temporal-consistency, and perceptual-plausibility metrics. This report proposes a compact, publication-grade metrics suite and the exact places to wire it in.

## 1. State of the codebase

- `motionflow_mv/eval/metrics.py`: `mpjpe`, `pa_mpjpe`, `pck` (rigid Procrustes, no scale).
- `experiments/train_ray_attention_real.py`: reports unaligned validation MPJPE in meters; no root-centering or alignment.
- `experiments/eval_all_plugins_shelf.py`: reports reprojection error (px) because 3D GT is not yet used for direct 3D evaluation.
- Missing: root-relative MPJPE, PCK-AUC, acceleration error, bone-length/symmetry error, and per-joint/per-sequence breakdowns.

## 2. Recommended metrics and concrete actions

### 2.1 Root-relative MPJPE and PA-MPJPE convention
Most benchmarks (Human3.6M, MPI-INF-3DHP) report MPJPE after subtracting the pelvis/root joint. The current `pa_mpjpe` is rigid (no scale), which is correct for CVPR-style pose comparisons, but the robot-downstream story also needs non-aligned, root-relative MPJPE to preserve metric scale.

**Action 1:** Add `root_relative_mpjpe(pred, gt, root_idx)` and a `scale=False` flag to `pa_mpjpe` in `motionflow_mv/eval/metrics.py`. Use dataset-specific root indices (H36M pelvis, COCO/Halpe pelvis proxy) and document the convention. Update `train_ray_attention_real.py` and `eval_ray_attention_robustness.py` to log root-relative MPJPE and the rigid PA-MPJPE side-by-side.

### 2.2 PCK and AUC
PCK is threshold-dependent; AUC aggregates thresholds. Human3.6M and 3DHP typically use 0–150 mm or 0–300 mm thresholds, sometimes normalized by pelvis-to-head length.

**Action 2:** Implement `compute_auc(pred, gt, thresholds=np.linspace(0, 150, 31))` in `metrics.py`. Report AUC on Shelf/Campus/H36M once 3D GT is wired into the evaluation scripts.

### 2.3 Acceleration / temporal consistency
Smoothness matters for the ICRA/robotics angle. Acceleration error is the L2 norm of the second derivative of the joint trajectory:

\[
a(t) = J(t+1) - 2J(t) + J(t-1)
\]

\[
\text{AccelErr} = \frac{1}{T-2}\sum_t \| a_{\text{pred}}(t) - a_{\text{gt}}(t) \|_2
\]

**Action 3:** Add `acceleration_error(pred_3d_seq, gt_3d_seq)` to `metrics.py` and log it alongside MPJPE. This directly validates the benefit of the `temporal_refiner` and any future smoothness loss.

### 2.4 Perceptual / biomechanical plausibility
Pure 3D pose metrics do not capture whether the skeleton looks human. A lightweight perceptual proxy combines:

- **Bone-length consistency:** mean absolute difference between predicted and GT bone lengths.
- **Symmetry error:** mean difference between left and right limb lengths.
- **Implausible-frame ratio:** fraction of frames with bone-length ratios outside a plausible range or with high jerk.

**Action 4:** Add `bone_length_error`, `symmetry_error`, and `implausible_frame_ratio` to `metrics.py`, and compare GVHMR single-view outputs against fused multi-view outputs to show that `ray_attention` reduces artifacts beyond MPJPE.

## 3. Risks

- **Unit confusion:** Shelf is in mm, synthetic data in meters, and the model receives scaled intrinsics. Metrics must assert units and convert consistently.
- **Skeleton mapping:** COCO/Halpe (17 joints), H36M (17/32 joints), and SMPL (24/6890) need dataset-specific root joints and bone-pair definitions.
- **Procrustes mismatch:** External papers sometimes use similarity alignment. We must report the same rigid, no-scale PA-MPJPE for fair comparison.
- **3D GT availability:** Shelf/Campus 3D annotations exist in `shelf_loader.py` but are not yet used in plugin evaluation. Without them we cannot validate MPJPE/PA-MPJPE on real data.

## 4. Fit into the paper plan

For ICRA/CVPR 2027, present two tables:

1. **3D accuracy table:** root-relative MPJPE, PA-MPJPE, and PCK-AUC on H36M, Panoptic, Shelf, and Campus. This is the main CVPR claim: `ray_attention` plus geometry beats DLT and prior learned fusion.
2. **Plausibility/downstream table:** acceleration error, bone-length error, symmetry error, and reprojection error. This supports the ICRA angle: fused motion is temporally consistent and safe for robot retargeting.

Reprojection error on Shelf should become a sanity check (single-digit pixels), not the headline result. The headline metrics must be 3D-supervised, which requires using existing Shelf/Campus 3D GT or acquiring Human3.6M and wiring these metrics into `eval_all_plugins_shelf.py`.

## 5. Concrete next steps

1. Extend `motionflow_mv/eval/metrics.py` with root-relative MPJPE, AUC, acceleration error, and bone-length/symmetry error.
2. Update `experiments/train_ray_attention_real.py` to log root-relative MPJPE and acceleration error on a held-out sequence.
3. Add a 3D-GT evaluation branch to `experiments/eval_all_plugins_shelf.py` and report MPJPE/PA-MPJPE/PCK-AUC per plugin.
4. Create `experiments/eval_all_plugins_h36m.py` once the H36M loader lands, reusing the same metric helpers.
5. Standardize on meters internally, document units in `HumanMotionIR` metadata, and freeze a `METRICS.md` protocol before any camera-ready submission.
