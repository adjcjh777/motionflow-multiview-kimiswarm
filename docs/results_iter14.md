# Iter14 Results Tracker

This document tracks the smoke/full results for the four concrete experiments launched from the 20-agent iter14 proposal swarm.

## Current anchor

- Model: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
- Clean MPJPE: **9.32 mm**
- Clean PA-MPJPE: **5.37 mm**

## Experiments

### 1. Robust Reprojection-Consistency Loss

- Files: `motionflow_mv/losses/reprojection_consistency.py`, `experiments/train_reprojection_consistency_pp_smoke_mpiinf3dhp.py`
- Smoke script: `scripts/run_reprojection_consistency_pp_smoke_wsl.sh`
- Output: `outputs/reprojection_consistency_pp_smoke.pth`
- Hypothesis: Direct 2-D reprojection supervision on raw + refined 3-D pose improves intrinsic robustness without adding parameters.
- Pass/fail:
  - Pass: val MPJPE ≤ 60 mm, no NaNs.
  - Pass: reprojection loss is finite and decreases.
- Results: **pending GPU**

### 2. Dynamic View-Selection Gate

- Files: `motionflow_mv/fusion/dynamic_view_selection_gate.py`, `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_dynamic_gate_model.py`
- Smoke script: `scripts/run_dynamic_view_gate_smoke_wsl.sh`
- Output: `outputs/dynamic_view_gate_smoke.pth`
- Hypothesis: A per-view/per-joint soft gate learns to drop noisy/occluded views per joint, improving dropout robustness at no clean cost.
- Pass/fail:
  - Pass: val MPJPE ≤ 11 mm (smoke) and gate mean ≤ 0.95.
  - Pass: gate produces non-uniform weights.
- Results: **pending GPU**

### 3. Skeleton-Graph Residual Refinement

- Files: `motionflow_mv/fusion/skeleton_graph_residual_refiner.py`, `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_graph_skeleton_residual_model.py`
- Smoke script: `scripts/run_graph_skeleton_residual_pp_smoke_wsl.sh`
- Output: `outputs/graph_skeleton_residual_pp_smoke.pth`
- Hypothesis: Propagating pose corrections along bone/symmetry edges enforces anatomical consistency and reduces distal-joint errors.
- Pass/fail:
  - Pass: val MPJPE ≤ 60 mm, no NaNs.
  - Pass: edge index builds for J=17 and J=28.
- Results: **pending GPU**

### 4. Epipolar-Biased Weight Head

- Files: `motionflow_mv/fusion/epipolar_attention_bias.py`, `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_model.py`
- Smoke script: `scripts/run_epipolar_pp_smoke_wsl.sh`
- Output: `outputs/epipolar_pp_smoke.pth`
- Hypothesis: Biasing per-view weight logits with epipolar-line distances focuses fusion on geometrically consistent view pairs.
- Pass/fail:
  - Pass: val MPJPE ≤ 60 mm, no NaNs.
  - Pass: epipolar bias is finite and non-zero for ≥90% of joints/views.
- Results: **pending GPU**

## Smoke evaluation summary

After the smoke queue finishes, run:

```bash
bash scripts/eval_iter14_smokes.sh
```

Results will appear in `outputs/iter14_smoke_eval/`.

## Integration plan

1. If any smoke reaches ≤ 9.8 mm clean MPJPE within 0.5 mm of the anchor, queue a full 20-epoch run.
2. Run the 6-axis robustness matrix on the best checkpoint.
3. Integrate the winning module into the factorized/visibility-v2 variants if it is orthogonal to them.
