# Next Iteration Plan — 20-Agent Swarm Design Review

**Date:** 2026-08-05
**Baseline:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
**Best result:** MPI-INF-3DHP clean **9.32 mm** / PA-MPJPE **5.37 mm**

## Purpose

Run a second self-evolution loop by having 20 independent planning agents review the codebase, the current best result, and every open design direction. Each agent produced a concrete, minimal next-step proposal. This document synthesizes those reports and ranks the highest-ROI directions for the next round of experiments.

## Current constraints

- GPU budget is limited to the local WSL RTX 4090 while one full training run is in progress.
- A800-D is read-only for this project; we cannot write data or start jobs there.
- No over-engineering: each next step must be the smallest change that can validate an idea.

## All 20 directions at a glance

| # | Direction | Core idea | Expected impact | Risk | Priority |
|---|---|---|---|---|---|
| 1 | variable_view_inference_optimization | Make the fixed-view best model work with 2–14 active views at inference; add training-time view dropout. | Variable-camera-count deployment; keep clean accuracy. | Dropout may slightly hurt full-view accuracy. | **High** |
| 2 | cross_dataset_h36m_mpi_fusion | Unify MPI-INF-3DHP (14V/28J) and H36M (4V/17J) in one cross-view PP model with padding/masking and per-dataset heads. | Strong cross-dataset generalization story. | Padding mask implementation; H36M may still fail to generalize. | **Medium-High** |
| 3 | aistpp_dataset_integration | Convert/integrate AIST++ canonical `.npz` into mixed training/eval. | Third real-world dataset for generalization. | Skeleton/rig mismatch with MPI/H36M. | **Medium** |
| 4 | shelf_campus_dataset_integration | Fix Campus scale issue and evaluate best cross-view PP model on Shelf/Campus. | Real small-camera-rig evidence. | Campus calibration/scale problems. | **Medium** |
| 5 | spatiotemporal_transformer_full | Add PP correction to the `(T×V×J)` spatio-temporal transformer and train full MPI. | Joint-joint attention may push clean below 9 mm. | Memory/compute `O((TVJ)²)`; small proven gain so far. | **Medium** |
| 6 | graph_joint_relation | Replace dense joint attention with `GraphJointRelation` in the best PP model. | Skeleton-aware reasoning; better occlusion handling. | Graph edges may constrain too much. | **Medium** |
| 7 | uncertainty_weighted_triangulation_v2 | Add learned per-view log-variance on top of best PP model. | Down-weight noisy views; slight clean/robust gain. | NLL loss can destabilize early training. | **Medium** |
| 8 | visibility_gated_fusion_v2 | Plug an explicit visibility head into the best PP model. | Handles real occlusion patterns. | Synthetic dropout labels are coarse. | **Medium** |
| 9 | adaptive_view_selection_efficient | Lightweight attention mask (not triangulation gate) on best PP model. | Faster inference; potential robustness gain. | Gating may hurt clean accuracy. | **Low-Medium** |
| 10 | camera_positional_encoding_v3 | Re-try CamPE on the best PP model at full capacity. | Geometry-aware embedding; variable rigs. | CamPE already failed twice. | **Low-Medium** |
| 11 | learned_triangulation_v2 | Combine Gauss-Newton learnable triangulation with best PP + uncertainty. | Further geometric refinement. | Complex, prone to instability. | **Low-Medium** |
| 12 | multi_task_shape_pose_v2 | Attach SMPL shape/pose head to best PP model. | Direct SMPL output for downstream. | No ground-truth SMPL; may hurt 3D accuracy. | **Low-Medium** |
| 13 | domain_adaptation_mpi_h36m | GRL+FiLM wrapper on best PP model for synthetic→real or cross-subject. | Better domain transfer. | GRL instability; limited data. | **Low-Medium** |
| 14 | robustness_to_occlusion_synthetic | Generate synthetic occlusion labels and train/evaluate explicit occlusion robustness. | Occlusion benchmark. | Synthetic vs real gap. | **Medium** |
| 15 | calibration_perturbation_augmentation | Curriculum of extrinsic perturbations on best PP model. | Reduce rot/trans robustness gap. | Too strong augmentation may hurt clean accuracy. | **High** |
| 16 | multiview_motionflow_pipeline_integration | Wrap best model as a `FusionModule` so the MotionFlow plugin can use it. | Production/pipeline integration. | Interface work only; no accuracy gain. | **High (non-GPU)** |
| 17 | real_time_inference_optimization | Replace `TransformerEncoderLayer` with SDPA/FlashAttention backend in the cross-view model. | 20–40% latency drop, equivalent accuracy. | Gains may vary by GPU. | **Low** |
| 18 | evaluation_webbridge_benchmark | Build a unified benchmark harness over all canonical WebBridge `.npz` datasets. | Paper-ready benchmark tables. | Preprocessing bugs (H36M S9/S11) may block. | **High (non-GPU)** |
| 19 | paper_writing_icra_cvpr | Update paper draft and figures with the new 9.32 mm results and robustness table. | Publishable narrative. | Must wait for final results. | **Medium (non-GPU)** |
| 20 | next_iteration_prioritization (adaptive PP) | Add `AdaptiveViewSelector` to best PP model. | Occlusion/perturbation robustness. | Selection training instability. | **Low-Medium** |

## Top 5 ranked next actions

### 1. Calibration perturbation curriculum (GPU, highest priority)
**Why:** The current robustness table shows rot_0.5° → 16.89 mm, rot_1.0° → 27.45 mm, focal_1pct → 19.13 mm. Clean accuracy is already excellent; the biggest paper-quality gain is to make the model robust to realistic calibration drift.

**Concrete step:**
- Modify `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` to support an extrinsic curriculum (rot 0.5°→1.5°, trans 5→15 mm) while keeping `pp_loss_weight=0.05`.
- Train for 16–20 epochs from scratch on full MPI S1+S3.
- Evaluate with the existing robustness matrix.

**Success criterion:** Clean MPJPE stays ≤ 9.6 mm, and rot_0.5° drops below 14 mm.

### 2. Variable-view inference benchmark and view-dropout training (GPU/CPU)
**Why:** Adds a strong practical/variable-view selling point without changing the model architecture.

**Concrete step:**
- Extend `experiments/eval_variable_views.py` to load the best PP checkpoint and support `crossview_residual_principal_point`.
- Run MPJPE@k for k = 2..14 on MPI-INF-3DHP S2.
- Add a `--view_dropout_rate` augmentation to the training script and re-train one model.

**Success criterion:** k=14 matches 9.32 mm baseline; k=4–10 shows graceful degradation.

### 3. Unified WebBridge evaluation harness (non-GPU)
**Why:** Needed for any paper-quality claim across datasets; also uncovers preprocessing gaps (H36M S9/S11).

**Concrete step:**
- Create `experiments/run_webbridge_benchmark.py` that reads a YAML manifest and runs `eval_full_metrics.py` over each `.npz`.
- Create `configs/benchmark_webbridge_mpi_smoke.yaml` listing the available MPI/H36M/AIST `.npz` files.
- Run on the best checkpoint and produce a CSV/JSON summary.

### 4. MotionFlow plugin integration for the best model (non-GPU)
**Why:** Closes the loop between research model and the MotionFlow single-view pipeline, directly serving the original idea (multi-view extraction → fusion → MotionFlow integration).

**Concrete step:**
- Add `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_module.py` as a `FusionModule`.
- Register it in `motionflow_mv/fusion/__init__.py`.
- Add a unit test in `tests/test_pipeline_multiview_plugin_best_model.py`.

### 5. Occlusion / visibility gating on the best model (GPU)
**Why:** Explicit occlusion handling is a natural follow-up after calibration robustness and a clear paper contribution.

**Concrete step:**
- Build `motionflow_mv/fusion/visibility_gated_fusion_v2.py` that subclasses the best PP model.
- Use dropout-generated visibility labels and a small BCE loss.
- Train and evaluate on clean + synthetic occlusion conditions.

**Success criterion:** Clean no regression; ≥10% improvement under .3 occlusion.

## Immediate action plan (next 48 hours)

1. **Wait for the running CamPE v2 full training to finish**, then evaluate it. If it is not better than the 9.32 mm baseline, document it as a negative result and free GPU for the next high-priority experiment.
2. **Non-GPU work in parallel:**
   - Implement the WebBridge benchmark harness.
   - Wrap the best model in a `FusionModule`.
   - Extend `eval_variable_views.py` for the best PP model.
3. **First GPU experiment after CamPE:** calibration perturbation curriculum.
4. **Update GitHub:** open an issue summarizing this plan and link the new doc; keep PR #17 current.

## Negative results to keep in mind

- Two-stage refined PP: 14.53 mm vs 10.34 mm (dropped).
- CamPE v2 small: 14.39 mm (negative; full run pending).
- Mixed-dataset PP small on H36M: 101 mm cross-dataset generalization failure.
- Adaptive view selection gates on triangulation hurt accuracy.

## How this fits the self-evolution loop

1. **Design:** 20-agent swarm produced the above plan.
2. **Implement:** Pick the smallest high-ROI item (WebBridge harness or plugin wrapper) first.
3. **Train:** Calibration curriculum and variable-view dropout once GPU is free.
4. **Evaluate:** Same clean + robustness protocol, plus the new variable-view and cross-dataset metrics.
5. **Critique:** Compare to 9.32 mm baseline; keep only improvements or clear ablations.
6. **Loop:** Feed results into another design swarm if needed.
