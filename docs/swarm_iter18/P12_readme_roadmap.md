# P12: README.md Roadmap Update — Swarm-Iter18 (OmniMultiViewFusion)

**Branch:** `feat/swarm-iter18-omniview`
**Date:** 2026-08-07
**Goal:** Refresh the top-level README roadmap so it reflects the current 8.35 mm ensemble result and the swarm-iter18 plan, instead of the old pre-8.75 mm iteration.

---

## What changed and why

The existing `README.md` `## Roadmap (next iteration)` section still points to `docs/iter_next_swarm_plan.md`, describes an anchor run that has already completed, and lists parallel tracks from the pre-8.75 mm iteration. Since the project has now reached an **8.35 mm** MPJPE ensemble on MPI-INF-3DHP S2/Seq1 and is moving into the OmniMultiViewFusion unification phase, the roadmap needs to:

1. Acknowledge the current best result (8.35 mm ensemble / 5.29 mm PA-MPJPE).
2. State the new single-model target (≤ 8.0 mm) and ensemble target (≤ 7.8 mm).
3. Align the parallel tracks with the swarm-iter18 priorities: unified architecture, visibility gating, graph-joint attention, uncertainty-weighted triangulation, calibration robustness, and paper packaging.
4. Add explicit success gates (clean accuracy, robustness, repeated seeds, variable-view curves) before any new model can replace the anchor.

---

## Proposed new `## Roadmap (next iteration)` section

Replace the entire `## Roadmap (next iteration)` block in `README.md` with the following.

```markdown
## Roadmap (next iteration)

**Goal:** push MotionFlow-MultiView to ICRA/CVPR 2027 publishable quality on MPI-INF-3DHP. Current best: **8.35 mm** MPJPE / **5.29 mm** PA-MPJPE on S2/Seq1 (ensemble of `bayesian_tri_v2` variants). Next target: a single model at **≤ 8.0 mm** and a reproducible ensemble at **≤ 7.8 mm**, with robustness under ≥30% view dropout and ≥0.5° camera rotation perturbation.

**Plan docs:** `docs/swarm_iter18/P02_omniview_arch.md` (architecture), `docs/swarm_iter18/P11_paper_story.md` (paper narrative), and `docs/iter_next_action_plan.md` (action plan).

### Current anchor

- Ensemble: `bayesian_tri_v2_stabilized` + `bayesian_tri_v2_aug` (d=128)
- Checkpoints: `outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth`, `outputs/bayesian_tri_v2_aug_mpiinf3dhp.pth`
- MPI-INF-3DHP S2/Seq1 clean: **8.35 mm** MPJPE, **5.29 mm** PA-MPJPE, **0.9444** PCK-AUC
- Eval script: `scripts/eval_bayesian_tri_v2_large_scale_wsl.sh`

### Parallel tracks (swarm-iter18 "OmniMultiViewFusion")

1. **Unified architecture**
   Integrate visibility gating, skeleton-graph joint attention, uncertainty-weighted triangulation, and spatiotemporal view attention into a single `OmniMultiViewFusion` module. See `docs/swarm_iter18/P02_omniview_arch.md`.
2. **Visibility & occlusion robustness**
   Learned per-view/per-joint soft visibility, synthetic joint occlusion augmentation, and a fallback guard for <2 visible views.
3. **Graph-joint & skeleton constraints**
   Skeleton-graph attention and kinematic-chain refiner to improve limb joints without trunk regression.
4. **Uncertainty & calibration**
   Anisotropic covariance/precision head, adaptive Gauss-Newton refinement, and an intrinsic self-calibration head; extend the camera-perturbation curriculum.
5. **Robustness evaluation**
   Run the extended robustness matrix on every candidate; produce MPJPE@k variable-view curves for k=2..14.
6. **Reproducibility & test-set**
   Repeated-seed runs (≥3 seeds), `manifest.json` per run, and evaluation on MPI-INF-3DHP official test subjects TS1–TS6.
7. **Cross-dataset & paper package**
   WebBridge benchmark on H36M, MPI-INF-3DHP, and AIST; ablation CSV template + plotting; figures; runtime benchmark on RTX 4090; update `docs/icra_cvpr_2027_paper_story.md`.

### Success gates (all must pass to replace the anchor)

| Gate | Threshold |
|------|-----------|
| Clean MPJPE (S2/Seq1) | < current best; target single-model ≤ 8.0 mm |
| Robustness — view_dropout_30 | ≤ 12 mm |
| Robustness — rot_0.5° | ≤ 12 mm |
| Robustness — joint_occlusion_30 | ≤ 12 mm |
| Repeated seeds | n ≥ 3, mean ≤ 8.0 mm, no seed >10% worse |
| Variable views | k=14 within 0.5 mm of full views; k=4 < 20 mm |
```

---

## Diff to apply

```diff
--- README.md (old Roadmap section)
+++ README.md (new Roadmap section)
@@
-## Roadmap (next iteration)
-
-Goal: push the multi-view pose model to ICRA/CVPR 2027 publishable quality on MPI-INF-3DHP, with the near-term target of **MPJPE < 8.75 mm** on the validation set.
-
-Full plan: `docs/iter_next_swarm_plan.md`.
-
-### Anchor run (in progress — no new GPU training until it finishes)
-
-- Script: `scripts/run_bayesian_tri_v2_large_scale_wsl.sh`
-- Config: `bayesian_tri_v2_pp` (d=128, residual_hidden=256, n_st_layers=3, 50 epochs, 2000 samples)
-- Hardware: RTX 4090 (WSL)
-- Log: `outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.log`
-- Eval: `scripts/eval_bayesian_tri_v2_large_scale_wsl.sh`
-
-### Parallel tracks (code, CPU smoke tests, and docs only for now)
-
-1. **Data & Augmentation**
-   - Audit WebBridge MPI-INF-3DHP data availability and create a manifest.
-   - Synchronized multiview 2D augmentation.
-   - Synthetic joint occlusion augmentation.
-   - Confidence-aware random view dropout with per-joint resampling.
-2. **Calibration**
-   - Extend camera perturbation ranges and intrinsics curriculum.
-3. **Trainer**
-   - Cosine LR schedule, warmup, gradient clipping, and AMP.
-   - EMA checkpoint save/load support.
-4. **Architecture**
-   - Prototype deeper ST attention model as a new model class.
-   - Prototype cross-view graph attention fusion module.
-   - Prototype Bayesian tri v3 with learned per-joint precision and refinement.
-5. **Loss & Inference**
-   - Temporal velocity + acceleration consistency loss.
-   - Ensemble inference across multiple checkpoints.
-6. **Evaluation, HPO & Documentation**
-   - Robustness matrix across noise, occlusion, and view dropout.
-   - Hyperparameter search script for large runs.
-   - Ablation study CSV template and plotting script.
-   - ICRA/CVPR 2027 paper story draft.
-   - README roadmap and GitHub issue #25 status update.
-   - Synthesize swarm outputs into a next-iteration action plan.
+## Roadmap (next iteration)
+
+**Goal:** push MotionFlow-MultiView to ICRA/CVPR 2027 publishable quality on MPI-INF-3DHP. Current best: **8.35 mm** MPJPE / **5.29 mm** PA-MPJPE on S2/Seq1 (ensemble of `bayesian_tri_v2` variants). Next target: a single model at **≤ 8.0 mm** and a reproducible ensemble at **≤ 7.8 mm**, with robustness under ≥30% view dropout and ≥0.5° camera rotation perturbation.
+
+**Plan docs:** `docs/swarm_iter18/P02_omniview_arch.md` (architecture), `docs/swarm_iter18/P11_paper_story.md` (paper narrative), and `docs/iter_next_action_plan.md` (action plan).
+
+### Current anchor
+
+- Ensemble: `bayesian_tri_v2_stabilized` + `bayesian_tri_v2_aug` (d=128)
+- Checkpoints: `outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth`, `outputs/bayesian_tri_v2_aug_mpiinf3dhp.pth`
+- MPI-INF-3DHP S2/Seq1 clean: **8.35 mm** MPJPE, **5.29 mm** PA-MPJPE, **0.9444** PCK-AUC
+- Eval script: `scripts/eval_bayesian_tri_v2_large_scale_wsl.sh`
+
+### Parallel tracks (swarm-iter18 "OmniMultiViewFusion")
+
+1. **Unified architecture**
+   Integrate visibility gating, skeleton-graph joint attention, uncertainty-weighted triangulation, and spatiotemporal view attention into a single `OmniMultiViewFusion` module. See `docs/swarm_iter18/P02_omniview_arch.md`.
+2. **Visibility & occlusion robustness**
+   Learned per-view/per-joint soft visibility, synthetic joint occlusion augmentation, and a fallback guard for <2 visible views.
+3. **Graph-joint & skeleton constraints**
+   Skeleton-graph attention and kinematic-chain refiner to improve limb joints without trunk regression.
+4. **Uncertainty & calibration**
+   Anisotropic covariance/precision head, adaptive Gauss-Newton refinement, and an intrinsic self-calibration head; extend the camera-perturbation curriculum.
+5. **Robustness evaluation**
+   Run the extended robustness matrix on every candidate; produce MPJPE@k variable-view curves for k=2..14.
+6. **Reproducibility & test-set**
+   Repeated-seed runs (≥3 seeds), `manifest.json` per run, and evaluation on MPI-INF-3DHP official test subjects TS1–TS6.
+7. **Cross-dataset & paper package**
+   WebBridge benchmark on H36M, MPI-INF-3DHP, and AIST; ablation CSV template + plotting; figures; runtime benchmark on RTX 4090; update `docs/icra_cvpr_2027_paper_story.md`.
+
+### Success gates (all must pass to replace the anchor)
+
+| Gate | Threshold |
+|------|-----------|
+| Clean MPJPE (S2/Seq1) | < current best; target single-model ≤ 8.0 mm |
+| Robustness — view_dropout_30 | ≤ 12 mm |
+| Robustness — rot_0.5° | ≤ 12 mm |
+| Robustness — joint_occlusion_30 | ≤ 12 mm |
+| Repeated seeds | n ≥ 3, mean ≤ 8.0 mm, no seed >10% worse |
+| Variable views | k=14 within 0.5 mm of full views; k=4 < 20 mm |
```

---

## Verification

After applying the diff, run:

```bash
git diff --check README.md
git diff --exit-code -- README.md || echo "ready to commit"
```

No code was added, so no CPU smoke test is required. The change is documentation-only.
