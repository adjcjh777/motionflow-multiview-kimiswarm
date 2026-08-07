# Swarm Iteration 20 — OmniMultiViewFusion v4 Scale-Up

**Goal:** push MotionFlow-MultiView toward ICRA/CVPR 2027 by closing the variable-view (2–3 view) robustness gap and scaling experiments on WebBridge data.  
**Method:** self-evolution loop (design → train → evaluate → feedback → next design), inspired by Qwen3-style iterative self-improvement.  
**Status:** plan  
**Date:** 2026-08-07  
**Tracking issue:** #76  

## Current bottlenecks (from iter19)

1. **Variable-view inference fails catastrophically for k<4.** H36M v2: 2-view ~1990 mm, 3-view ~1620 mm, 4-view ~15 mm. This is the single biggest blocker for real-world multi-camera capture where not all cameras see every joint.
2. **MPI clean accuracy has plateaued.** Best single-model ~9.03 mm; v2/v3 are near that but not yet reliably below 8.6 mm.
3. **Calibration robustness is still a headline weakness.** Rotation 0.5° → ~16.9 mm, principal-point ±10 px is catastrophic.
4. **Training on A800 needs persistent, reproducible orchestration.** We have ad-hoc tmux sessions; we need standardized scripts and lock files.

## v4 scope (no over-design)

Focus on **view-mask-aware graph attention + adaptive view selection / view-dropout training** as the headline change.  Everything else is optional, toggleable, and warm-startable from v2/v3 checkpoints.

## 20 parallel work packages

Each package is independent and writes to a dedicated file/directory to avoid merge conflicts.  Implementers must run CPU smoke tests before touching GPU.  All GPU work targets the **A800-D** (`a800-D` SSH host) unless noted; the 4090 WSL run is monitored but not restarted.

### T01 — v4 main model integration
*Owner:* coder  
*File:* `motionflow_mv/fusion/omniview_fusion_v4.py`  
*Task:* Subclass `OmniMultiViewFusionV3`.  Add `use_context_visibility`, `use_skeleton_residual`, `use_kinematic_refiner`, `use_adaptive_view_selection`, `use_rotation_correction`, `use_entropy_regularization` toggles.  Keep v2/v3 parameter names so `strict=False` warm-start works.  Provide `__main__` CPU smoke test.  
*Success:* `python motionflow_mv/fusion/omniview_fusion_v4.py` runs on CPU with B=2,T=9,V=4,J=17 and produces `(pred_3d, weights, visibility, covariance, epipolar_loss)`.

### T02 — view-mask-aware visibility head v2
*Owner:* coder  
*File:* `motionflow_mv/fusion/visibility_gated_fusion_v2.py` (extend existing)  
*Task:* Extend `VisibilityGatedFusionV2` to support per-joint context across views and an learned uncertainty channel that scales the soft visibility mask.  Ensure fallback guard still works when all views are masked out.  
*Success:* `tests/test_omniview_fusion_v4.py` (from T10) passes with `use_context_visibility=True` and `use_context_visibility=False`.

### T03 — adaptive view selector
*Owner:* coder  
*File:* `motionflow_mv/fusion/adaptive_view_selector.py`  
*Task:* Implement `AdaptiveViewSelector`: per `(view, joint)` Gumbel-softmax sampling during training, hard top-k during inference, budget loss.  Make it optional and bypassable.  
*Success:* CPU smoke test with V=4, J=17; training path samples a mask with mean k≈target; inference path selects exactly k views.

### T04 — rotation correction head
*Owner:* coder  
*File:* `motionflow_mv/fusion/rotation_correction.py`  
*Task:* Predict a bounded SO(3) residual per view from pooled per-view features; apply to `R` before triangulation.  Bound with `tanh` so init is near identity.  
*Success:* CPU smoke test shows a small rotation delta applied to a 4-view rig without NaNs.

### T05 — skeleton-graph residual refiner integration
*Owner:* coder  
*File:* `motionflow_mv/fusion/skeleton_graph_residual_refiner.py` (extend existing)  
*Task:* Wrap `SkeletonGraphResidualRefiner` so it can replace `residual_mlp` in v4.  Add smoke test.  
*Success:* Input `(B*T, J, d+3)` → output `(B*T, J, 3)` with correct shapes; gradients flow on CPU.

### T06 — kinematic-chain final refiner integration
*Owner:* coder  
*File:* `motionflow_mv/fusion/kinematic_chain_graph_refiner.py` (extend existing)  
*Task:* Add a tiny temporal wrapper `KinematicChainGraphRefinerTemporal` that operates on `(B, T, J, 3)` and is togglable in v4.  
*Success:* CPU smoke test on H36M skeleton.

### T07 — attention-entropy regularization module
*Owner:* coder  
*File:* `motionflow_mv/fusion/attention_entropy_loss.py`  
*Task:* Implement per-view triangulation-weight entropy loss as described in `docs/v4_architecture_design_proposal.md`.  Make it optional with weight flag.  
*Success:* Loss is non-negative, zero when weights are one-hot, differentiable.

### T08 — v4 WebBridge multi-dataset trainer
*Owner:* coder  
*File:* `experiments/train_omniview_fusion_v4_webbridge_multi.py`  
*Task:* Adapt `train_omniview_fusion_v2_webbridge_multi.py` for v4.  Support multi-dataset manifest, warm-start from v2/v3 checkpoint, calibration curriculum, view-dropout augmentation, entropy/budget/aux losses.  Add `--smoke` CPU mode.  
*Success:* `--smoke` runs end-to-end on CPU; full script launches on A800 with no import errors.

### T09 — v4 evaluation script
*Owner:* coder  
*File:* `experiments/eval_omniview_fusion_v4_mpiinf3dhp.py`  
*Task:* Adapt `eval_omniview_fusion_v2_mpiinf3dhp.py` for v4.  Report clean/PA-MPJPE, robustness matrix, variable-view MPJPE@k (k=2..V), per-joint error, visibility accuracy.  
*Success:* CPU smoke test completes; on A800 produces JSON/CSV matching v2/v3 format.

### T10 — CPU smoke / pytest suite for v4
*Owner:* coder  
*File:* `tests/test_omniview_fusion_v4.py`  
*Task:* Write pytest covering: shape/arity, gradient flow, warm-start from a tiny synthetic v2 checkpoint, toggles on/off, variable-view wrapper, entropy loss.  
*Success:* `pytest tests/test_omniview_fusion_v4.py -q` passes locally.

### T11 — variable-view inference hardening
*Owner:* coder  
*File:* `motionflow_mv/fusion/variable_view_inference.py` (extend existing)  
*Task:* Fix the 2/3-view catastrophic failure.  Add explicit view padding/masking, ensure graph-joint attention and visibility head handle missing views, and add a confidence-based fallback to triangulation when active views < min_views.  
*Success:* Running variable-view eval on v4 (or patched v2/v3) yields MPJPE@2 < 50 mm (target) instead of ~2000 mm.

### T12 — WebBridge data audit and expansion
*Owner:* explore  
*File:* `docs/swarm_iter20/webbridge_data_audit.md`  
*Task:* Audit `data/webbridge/` and `data/h36m_hf/` / `data/h36m_mirror/` availability.  List all subjects/sequences, sizes, 2D/3D shapes, camera counts, and any missing conversions.  Propose a full manifest for large-scale v4 training (all MPI subjects + H36M S1/S5/S6/S7/S8 train, S9/S11 val/test).  Read-only; do not modify data.  
*Success:* Manifest table with ≥10 rows and a clear list of missing files.

### T13 — calibration perturbation curriculum
*Owner:* coder  
*File:* `motionflow_mv/training/calibration_perturbation_curriculum.py`  
*Task:* Implement a curriculum that increases `rot_deg`, `focal_pct`, `pp_px` perturbations over epochs.  Hook into v4 trainer.  
*Success:* Unit test shows perturbation magnitude grows with epoch; no NaNs in triangulation after perturbation.

### T14 — A800-D persistent training orchestration
*Owner:* coder  
*File:* `scripts/run_omniview_fusion_v4_a800.sh` + `scripts/tmux_omniview_fusion_v4_a800.sh`  
*Task:* Produce a tmux-based script that: picks a free GPU on A800-D, activates the project venv, runs v4 trainer under `nohup`, restarts on non-zero exit up to 3 times, and writes to `outputs/omniview_fusion_v4_*.log`.  Use a lock file to prevent duplicate sessions for the same experiment.  
*Success:* Dry-run (`--help` or `--smoke`) works; can be launched by the orchestrator on A800-D.

### T15 — 4090 WSL monitoring and auto-eval
*Owner:* coder  
*File:* `scripts/monitor_4090_v2_and_auto_eval.sh` (improve existing)  
*Task:* Ensure the existing watchdog script monitors the 4090 training, reports GPU memory/temp, and triggers auto-eval only when a checkpoint is new and no eval is running.  Avoid cron conflicts with the A800-D evals.  
*Success:* Local dry-run shows lock-file logic; no duplicate eval processes.

### T16 — robustness matrix harness for v4
*Owner:* coder  
*File:* `experiments/robustness_matrix_v4.py`  
*Task:* Generalize the existing robustness matrix to support v4, variable views, and per-joint metrics.  Output a CSV compatible with `docs/tables/icra2027/robustness_matrix.md`.  
*Success:* Produces a CSV on CPU smoke data with all expected columns.

### T17 — v2/v3 baseline checkpoint reproduction and table
*Owner:* explore  
*File:* `docs/swarm_iter20/v2_v3_baseline_reproduction.md`  
*Task:* Re-run (or collect logs from) v2 and v3 MPI-INF-3DHP clean/robustness/variable-view evals.  Fill a single results table with MPJPE/PA-MPJPE and identify the strongest baseline to beat.  Do not modify model code.  
*Success:* Baseline table with ≥3 conditions and clear winner.

### T18 — 2/3-view failure analysis and visualization
*Owner:* explore  
*File:* `docs/swarm_iter20/failure_analysis_2_3_views.md` + `scripts/visualize_variable_view_failure.py`  
*Task:* Diagnose why 2/3 views fail: plot per-joint errors, view weights, visibility predictions, and triangulation residuals for k=2,3,4.  Identify if it is an attention collapse, triangulation degeneracy, or units/scale issue.  
*Success:* A visualization script and a markdown report with ≥3 concrete failure hypotheses.

### T19 — paper story and novelty positioning update
*Owner:* explore  
*File:* `docs/swarm_iter20/paper_story_v4.md`  
*Task:* Update `docs/paper_story_system_v2.md` / `docs/icra_cvpr_2027_paper_story.md` with the v4 narrative: "View-mask-aware adaptive multi-view fusion for robust monocular-to-multiview human pose estimation".  Map each module to a paper section and list 3–5 strongest quantitative claims we can defend once v4 trains complete.  
*Success:* A 1-page narrative plus an ablation table outline.

### T20 — GitHub issue/PR automation and synthesis
*Owner:* coder  
*File:* `docs/swarm_iter20/synthesis.md` + `scripts/post_results_to_github.py`  
*Task:* Create a helper script that posts a comment to issue #76 summarizing a given result JSON/CSV.  After all T01–T19 deliverables land, write `docs/swarm_iter20/synthesis.md` ranking the highest-ROI directions and propose the next exact training run.  
*Success:* Script can post to #76 (dry-run mode supported); synthesis doc is committed.

## Cross-package dependencies (read-only references)

- `motionflow_mv/fusion/omniview_fusion_v3.py` — v4 base class.
- `docs/v4_architecture_design_proposal.md` — design rationale.
- `docs/results_h36m_v2_dense_graph_a800.md` — the 2/3-view failure evidence.
- `experiments/train_omniview_fusion_v2_mpiinf3dhp.py` — trainer template.
- `experiments/eval_omniview_fusion_v2_mpiinf3dhp.py` — eval template.

## Global constraints

1. **Do not touch A800-D `/mnt/nvme0n1/zhangzy/projects` or the running Docker service — read-only.**
2. **GPU 4, 5, 7 on A800-D are currently busy; use GPU 0–3 or 6 for new training, and GPU 7 only after the current evals finish.**
3. **Commit atomically per task; do not batch half-finished work.**
4. **Run CPU smoke tests before GPU launch.**
5. **Update this doc and issue #76 when a task is done.**

## Definition of done for iter20

- [ ] All 20 tasks have a committed deliverable on branch `feat/swarm-iter20-v4`.
- [ ] `tests/test_omniview_fusion_v4.py` passes on CPU.
- [ ] At least one v4 training run is queued or running on A800-D (GPU 0–3/6) under tmux.
- [ ] A results table comparing v4 (any ablation) to the best v2/v3 baseline is posted to #76.
- [ ] A PR from `feat/swarm-iter20-v4` to `main` is opened, reviewed, and merged.
