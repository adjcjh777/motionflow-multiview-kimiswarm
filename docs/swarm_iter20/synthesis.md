# Swarm Iteration 20 — v4 Synthesis and Next Run Proposal

> Tracking issue: #76  
> Branch: `feat/swarm-iter20-v4`  
> Last updated: 2026-08-07

This document synthesises the iter20-v4 work packages (T01–T19) and proposes the highest-return next training run for ICRA/CVPR 2027.

## 1. Status at a glance

| Work package | Deliverable | Status | Main blocker / note |
|--------------|-------------|--------|---------------------|
| T01 | `motionflow_mv/fusion/omniview_fusion_v4.py` | planned | Needs v3 integration |
| T02 | `motionflow_mv/fusion/visibility_gated_fusion_v2.py` | planned | Depends on T01 |
| T03 | `motionflow_mv/fusion/adaptive_view_selector.py` | planned | None |
| T04 | `motionflow_mv/fusion/rotation_correction.py` | planned | None |
| T05 | `motionflow_mv/fusion/skeleton_graph_residual_refiner.py` | planned | None |
| T06 | `motionflow_mv/fusion/kinematic_chain_graph_refiner.py` | planned | None |
| T07 | `motionflow_mv/fusion/attention_entropy_loss.py` | planned | None |
| T08 | `experiments/train_omniview_fusion_v4_webbridge_multi.py` | planned | Needs T01–T07 |
| T09 | `experiments/eval_omniview_fusion_v4_mpiinf3dhp.py` | planned | Needs T01 |
| T10 | `tests/test_omniview_fusion_v4.py` | planned | Needs T01–T09 |
| T11 | `motionflow_mv/fusion/variable_view_inference.py` | planned | Critical path for k=2,3 |
| T12 | `docs/swarm_iter20/webbridge_data_audit.md` | planned | Read-only audit |
| T13 | `motionflow_mv/training/calibration_perturbation_curriculum.py` | planned | None |
| T14 | `scripts/run_omniview_fusion_v4_a800.sh` | planned | Needs T08 |
| T15 | `scripts/monitor_4090_v2_and_auto_eval.sh` | planned | None |
| T16 | `experiments/robustness_matrix_v4.py` | planned | Needs T09 |
| T17 | `docs/swarm_iter20/v2_v3_baseline_reproduction.md` | planned | Read-only / re-run evals |
| T18 | `docs/swarm_iter20/failure_analysis_2_3_views.md` + `scripts/visualize_variable_view_failure.py` | planned | Depends on T11/T17 |
| T19 | `docs/swarm_iter20/paper_story_v4.md` | **done** | Merged in this branch |
| T20 | `scripts/post_results_to_github.py` + `docs/swarm_iter20/synthesis.md` | **done** | This file |

## 2. Anchor numbers we are trying to beat

From the v2/v3 baseline (see `docs/results_h36m_v2_dense_graph_a800.md` and T17):

| Condition | v2/v3 anchor | v4 target |
|-----------|--------------|-----------|
| MPI-INF-3DHP S2/Seq1 clean MPJPE | **9.03 mm** single model / 8.35 mm ensemble | **< 8.6 mm** single model |
| H36M v2 2-view MPJPE | **~1990 mm** (catastrophic) | **< 50 mm** |
| H36M v2 3-view MPJPE | **~1620 mm** (catastrophic) | **< 30 mm** |
| 30 % view dropout | **18.15 mm** | **< 16.3 mm** |
| 30 % joint occlusion | **16.99 mm** | **< 16.0 mm** |
| Rotation 0.5° | **16.89 mm** | **< 14 mm** |
| Focal length 1 % | **19.13 mm** | **< 15 mm** |

The single biggest ROI is **variable-view inference (k=2,3)** because the current failure is catastrophic and is the main blocker for real-world capture rigs.

## 3. Highest-ROI directions (ranked)

### 1. Variable-view inference hardening (T11) — highest ROI
*Why:* 2-view ~1990 mm and 3-view ~1620 mm make the system unusable in practice.  
*What works:* explicit view padding/masking, confidence fallback to triangulation when active views < `min_views`, and adaptive view selection.  
*Success metric:* MPJPE@2 < 50 mm, MPJPE@3 < 30 mm on H36M v2.

### 2. View-mask-aware visibility gating v2 (T02) — high ROI
*Why:* Directly models occlusion and view dropout, the second-largest degradation.  
*What works:* per-joint context visibility across views + learned uncertainty.  
*Success metric:* 30 % view-dropout MPJPE < 16.3 mm, 30 % joint-occlusion MPJPE < 16.0 mm.

### 3. Adaptive view selector (T03) — high ROI
*Why:* Enables 2–3 view inference by learning a budgeted subset of views per joint.  
*What works:* Gumbel-softmax training, hard top-k inference, budget loss.  
*Success metric:* mean active views ≈ target k during training; exact k during inference.

### 4. Calibration perturbation curriculum + rotation correction (T04, T13) — medium-high ROI
*Why:* Rotation and focal drift are headline weaknesses for deployment.  
*What works:* bounded SO(3) residual head + progressive augmentation.  
*Success metric:* rot_0.5° < 14 mm, focal_1% < 15 mm.

### 5. Skeleton-graph residual refiner (T05) — medium ROI
*Why:* Replaces the dense MLP with anatomical graph propagation, reducing over-smoothing.  
*Success metric:* clean MPI MPJPE drops by ≥ 0.2 mm without hurting robustness.

### 6. Attention-entropy regularization (T07) — medium ROI
*Why:* Sharpens per-view weights and improves interpretability; cheap to add.  
*Success metric:* weight distributions become more one-hot; no accuracy regression.

### 7. Kinematic-chain final refiner (T06) — lower-medium ROI
*Why:* Targets distal-limb errors but risks over-smoothing.  
*Success metric:* reduced ankle/wrist error without global MPJPE regression.

## 4. Proposed next exact training run

**Run name:** `v4_baseline_vis_adaptive_rot`  
**Goal:** Close the variable-view gap first, then iterate on calibration robustness.

### Configuration

```yaml
model:
  omniview_fusion_v4:
    use_context_visibility: true
    use_skeleton_residual: true
    use_kinematic_refiner: false      # keep disabled until baseline is stable
    use_adaptive_view_selection: true
    use_rotation_correction: true
    use_entropy_regularization: true

training:
  datasets:
    - mpiinf3dhp_train
    - h36m_s1s5s6s7s8_train
    - webbridge_train
  epochs: 60
  batch_size: 32
  lr: 1e-4
  warm_start: outputs/omniview_fusion_v3_best.pth  # strict=False
  freeze_encoder_epochs: 5
  calibration_perturbation_curriculum:
    rot_deg: [0.0, 0.1, 0.25, 0.5]      # epoch milestones
    focal_pct: [0.0, 0.5, 1.0, 2.0]
    pp_px: [0.0, 2.0, 5.0, 10.0]
  view_dropout: 0.15                    # per-frame per-view
  joint_occlusion: 0.10                 # per-frame per-joint

hardware:
  host: a800-D
  gpus: [0, 1, 2, 3]                   # avoid 4, 5, 7
  tmux_session: iter20_v4_baseline
```

### Launch steps

1. Ensure T01–T07, T11, T13 and T10 are merged and `pytest tests/test_omniview_fusion_v4.py -q` passes on CPU.
2. Run CPU smoke: `python experiments/train_omniview_fusion_v4_webbridge_multi.py --smoke`.
3. On A800-D, start the persistent orchestration from `scripts/run_omniview_fusion_v4_a800.sh`.
4. After the first 10 epochs, run `experiments/eval_omniview_fusion_v4_mpiinf3dhp.py` to verify no catastrophic regression.
5. After convergence, run the robustness matrix (`experiments/robustness_matrix_v4.py`) and variable-view evaluation.
6. Post the JSON/CSV summary to issue #76 with `scripts/post_results_to_github.py --issue 76 --json <json> --csv <csv> --title "v4_baseline_vis_adaptive_rot results"`.

### Expected deliverables from this run

- `outputs/omniview_fusion_v4_baseline_vis_adaptive_rot/best_model.pth`
- `outputs/eval_jsons/v4_baseline_vis_adaptive_rot_mpiinf3dhp.json`
- `outputs/robustness_matrix_v4_baseline_vis_adaptive_rot.csv`
- Comment on #76 with clean + robustness + variable-view metrics.

## 5. Fallback plan if v4 training destabilises

1. Disable `use_kinematic_refiner` and `use_skeleton_residual` (keep residual MLP).
2. Reduce learning rate to `5e-5` and extend warm-start freezing to 10 epochs.
3. Disable `use_entropy_regularization` (lowest priority loss).
4. Re-run with only `use_context_visibility` + `use_adaptive_view_selection`.

## 6. Open questions before launch

1. Do we have a reproducible v3 checkpoint that loads with `strict=False` in v4?
2. Has T12 confirmed WebBridge train/val split sizes and 17-joint compatibility?
3. Is the variable-view eval harness (T11) passing CPU smoke before GPU launch?

## 7. Related files

- Action plan: `docs/swarm_iter20_action_plan.md`
- Paper story: `docs/swarm_iter20/paper_story_v4.md`
- Post-results script: `scripts/post_results_to_github.py`
- v4 model: `motionflow_mv/fusion/omniview_fusion_v4.py`
- v4 trainer: `experiments/train_omniview_fusion_v4_webbridge_multi.py`
- v4 eval: `experiments/eval_omniview_fusion_v4_mpiinf3dhp.py`
