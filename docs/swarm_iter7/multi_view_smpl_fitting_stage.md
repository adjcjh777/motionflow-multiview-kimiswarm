# Multi-view SMPL fitting stage

I investigated the multi-view SMPL fitting stage but could not write the report to disk because this subagent does not have file-editing tools. Below is the full report content, which the parent agent can save to `docs/swarm_iter7/multi-view_smpl_fitting_stage.md`.

---

```markdown
# Multi-view SMPL fitting stage

## 1. Current state

A standalone offline SMPL fitter already exists:

- `experiments/fit_smpl_multiview.py` — PyTorch/Adam optimization of `global_orient`, `body_pose`, `transl`, and a shared `betas` vector given fused 3D joints and optional multi-view 2D keypoints + cameras.
- `motionflow_mv/ir/multiview_adapter.py:18-119` — fuses per-view `HumanMotionIR`s but only averages per-view `betas` (`_average_betas`, lines 143-156) and shifts root translation; it does not recover coherent SMPL parameters from multi-view evidence.
- `motionflow_mv/ir/human_motion_ir.py:25-39` — the IR already stores SMPL keys (`body_pose`, `global_orient`, `transl`, `betas`).
- `motionflow_mv/eval/metrics.py` — reports MPJPE/PA-MPJPE/PCK/AUC on 3D joints, but has no SMPL-parameter-level metrics (betas error, body_pose error, reprojection error).
- `experiments/generate_synthetic_multiview_dataset.py` — renders SMPL motion through calibrated rigs, but the saved `.npz` contains only `joints_3d`, `points_2d`, `confidences`, and camera arrays; ground-truth SMPL parameters are not stored.

The current best pose model, `RayAttentionFusionModelTemporalResidual` (`motionflow_mv/fusion/ray_attention_temporal_residual_model.py:38`), outputs 3D joints and per-view weights, not a parametric body.

## 2. Gap / opportunity

The project is still a **joint triangulation** pipeline. To reach the ICRA/CVPR 2027 claim of "recovering a coherent parametric body from multi-view video," we need a post-fusion **multi-view SMPL fitting stage** that:

1. Consumes the fused 3D joints *and* raw per-view 2D observations + ray-attention weights.
2. Optimizes a single sequence-level SMPL shape and per-frame pose under multi-view reprojection constraints.
3. Writes valid SMPL parameters back into a `HumanMotionIR`, instead of inheriting an arbitrary reference view's pose.
4. Provides quantitative SMPL-level evaluation on data where ground-truth SMPL parameters are known.

The opportunity is to close the loop from 2D keypoints → metric 3D joints → parametric body, which is a strong differentiator for both CVPR (geometry-aware learning) and ICRA (robot retargeting needs a valid, metric-scale body).

## 3. Concrete next step

**Add a synthetic SMPL-fitting benchmark that stores GT SMPL parameters and evaluates the fitter end-to-end.**

Specifically:

1. Extend `experiments/generate_synthetic_multiview_dataset.py` to save `gt_betas`, `gt_global_orient`, `gt_body_pose`, `gt_transl` for each sequence (and optionally per-frame). This is a one-line data-dict change.
2. Create a new experiment script `experiments/eval_smpl_fitting_synthetic.py` that:
   - Loads a synthetic `.npz`.
   - Runs the current `RayAttentionFusionModelTemporalResidual` (or DLT) to obtain fused 3D joints and ray-attention weights.
   - Calls `fit_smpl(..., reproj_weight=0.01, smooth_weight=0.01)` from `experiments/fit_smpl_multiview.py`.
   - Compares the fitted SMPL parameters to the stored GT parameters, and reports 3D MPJPE, reprojection error, and per-parameter error.
3. Optionally wrap the fitter into a `motionflow_mv/fusion/smpl_fitter.py` module with a function `fit_smpl_to_fusion_output(...)` so it can be invoked from the multiview adapter later.

This step intentionally does **not** modify the core `RayAttentionFusionModelTemporalResidual` or the multiview adapter yet; it validates the fitting stage on synthetic GT first.

## 4. Expected success metric

On the synthetic dataset with known SMPL GT:

- Fitted 3D joints MPJPE < 5 mm after fitting (the current offline fitter reaches ~32 cm on H36M; synthetic GT should be much tighter).
- Per-view reprojection error < 2 px.
- Shape parameter error `||fitted_betas - gt_betas||_2` < 0.5 (unit-normalized β).
- Body-pose axis-angle error < 5° per joint on average.

These numbers would confirm that the fitter recovers the true parametric body, not just a skeleton that interpolates the 3D joints.

## 5. Risks / blockers

- **No SMPL GT for real data.** Shelf/Campus/H36M provide only 3D joints; SMPL-parameter evaluation must rely on synthetic data or expensive pseudo-labeling (ScoreHMR) for now.
- **Windows NumPy BLAS instability.** The fitter uses `torch.linalg.svd` to avoid `np.linalg.svd` crashes; keep all matrix math in PyTorch.
- **Joint-set mismatch.** `fit_smpl_multiview.py` assumes the input joints are the first `J` SMPL body joints. If the fusion model outputs a different joint order (e.g., COCO/H36M 17 joints), a regressor or explicit mapping is required before fitting.
- **A800-D and Docker are read-only.** Do not modify anything there. WebBridge data may need download; do not commit large files.
- **Compute cost.** Per-sequence Adam optimization is offline-only. For real-time ICRA use, the fitting stage would later need to be amortized into a learned forward model.
```