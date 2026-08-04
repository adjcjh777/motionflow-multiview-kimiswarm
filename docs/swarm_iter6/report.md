# Swarm Iter 6: Iterative Residual Refinement

## Task
Implement iterative residual refinement for `RayAttentionFusionModelTemporalResidual`: at inference, apply the residual MLP head multiple times (`n_iter = 1, 2, 3`), feeding the updated 3D pose back into the head, and evaluate whether this further reduces MPJPE on MPI-INF-3DHP val S2 Seq1.

## Changes Made

### 1. Model: `motionflow_mv/fusion/ray_attention_temporal_residual_model.py`
- Added an `n_iter: int = 1` argument to `forward()`.
- Refactored the residual refinement block into a loop:
  ```python
  pred_3d = pred_3d_raw
  for _ in range(max(1, int(n_iter))):
      residual_input = torch.cat([feat_pooled, pred_3d], dim=-1)
      delta = self.residual_mlp(residual_input)
      pred_3d = pred_3d + delta
  ```
- Per-view attention weights and DLT triangulation are computed once; only the residual MLP is applied repeatedly.
- Updated the class docstring and `__main__` sanity check to cover `n_iter=3`.

### 2. Evaluation Script: `experiments/eval_ray_attention_temporal_residual_iterative.py`
- Loads `outputs/ray_attention_temporal_residual_v2.pth`.
- Evaluates `n_iter = 1, 2, 3` on a given `.npz` validation sequence.
- Reports MPJPE, PA-MPJPE, PCK@50/100/150 mm, and PCK AUC (0–150 mm).
- Writes a JSON summary.

## Smoke Evaluation (fast signal)

Dataset: `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz` (500 frames, 14 views, 28 joints).

| n_iter | MPJPE (mm) | PA-MPJPE (mm) | PCK@50mm | PCK@100mm | PCK@150mm | PCK AUC |
|--------|-----------|---------------|----------|-----------|-----------|---------|
| 1      | 10.76     | 11.26         | 1.0000   | 1.0000    | 1.0000    | 0.9283  |
| 2      | 42.35     | 44.79         | 0.6417   | 1.0000    | 1.0000    | 0.7177  |
| 3      | 75.81     | 79.10         | 0.3448   | 0.7038    | 0.9444    | 0.4951  |

**Result:** Naive iterative refinement strongly degrades performance. Each additional residual application over-corrects the pose because the residual MLP was trained to map the **raw** triangulated pose to the target, not to refine an already-refined estimate. MPJPE grows from ~10.8 mm (`n_iter=1`) to ~42.3 mm (`n_iter=2`) and ~75.8 mm (`n_iter=3`).

## Full MPI-INF-3DHP val S2 Seq1 Evaluation

Dataset: `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz` (6,502 frames, 14 views, 28 joints, 6,490 clips with `clip_len=13`).

| n_iter | MPJPE (mm) | PA-MPJPE (mm) | PCK@50mm | PCK@100mm | PCK@150mm | PCK AUC |
|--------|-----------|---------------|----------|-----------|-----------|---------|
| 1      | 13.12     | 10.86         | 0.9999   | 1.0000    | 1.0000    | 0.9125  |
| 2      | 40.99     | 40.68         | 0.7195   | 0.9982    | 1.0000    | 0.7267  |
| 3      | 71.15     | 71.49         | 0.3327   | 0.7730    | 0.9685    | 0.5271  |

The full-set results confirm the smoke trend. Single-pass refinement (`n_iter=1`) matches the expected strong baseline (~13.84 mm). Adding iterations severely harms accuracy: MPJPE more than triples at `n_iter=2` and grows to ~71 mm at `n_iter=3`.

## Conclusion
- **2–3 iterations of residual-head refinement at inference do NOT reduce MPJPE** for the current checkpoint; they cause rapid divergence on both the smoke subset and the full MPI-INF-3DHP val S2 Seq1 sequence.
- The residual MLP is trained to map the **raw** DLT triangulated pose to the target. When fed an already-refined pose, it over-corrects because the input distribution differs from training. A viable follow-up would be to train the model with the residual loop unrolled (i.e., predict and apply multiple residuals during training) or to add a learned damping factor / stopping criterion, but that is out of scope for this iteration.
- **Recommendation:** keep `n_iter=1` at inference for `ray_attention_temporal_residual_v2.pth`.

## Files Touched
- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py`
- `experiments/eval_ray_attention_temporal_residual_iterative.py`
- `docs/swarm_iter6/report.md`
- `docs/swarm_iter6/iterative_eval_smoke.json`
- `docs/swarm_iter6/eval_smoke_log.txt`
- `docs/swarm_iter6/iterative_eval.json`
- `docs/swarm_iter6/eval_full_log.txt`
