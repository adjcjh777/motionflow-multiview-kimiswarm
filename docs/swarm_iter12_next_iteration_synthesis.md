# Swarm Iter12 — Next-Iteration Synthesis (20-agent)

**Date:** 2026-08-06  
**Baseline:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** / PA-MPJPE **5.37 mm**  
**Running GPU job:** `train_crossview_residual_visibility_v2_mpiinf3dhp.py` on RTX 4090  
**A800-D:** SSH reachable, 8× A800-SXM4-80GB; read-only access only for `/mnt/nvme0n1/zhangzy/projects` and Docker.

This document synthesizes the outputs of a 20-agent read-only/design swarm that explored the next iteration of the MotionFlow-MultiView pipeline. The goal is to identify the highest-ROI next experiments for ICRA/CVPR 2027 without over-engineering.

---

## 1. Current State

- The PP-corrected cross-view residual model is the empirical anchor.
- `CrossviewResidualVisibilityV2` is training and will auto-evaluate on completion.
- Several advanced model files exist but are not validated at convergence (`uncertainty_residual_learned_tri_v1`, `spatiotemporal_principal_point`, factorized variants).
- Paper figures/tables are stale (still show 11.17 mm instead of 9.32 mm).
- No unified SOTA comparison harness exists.
- A800-D is available but memory-heavy jobs are running on GPUs 4–7.

---

## 2. Key Findings by Theme

### Architecture (Agents 1–5, 9, 11)

| Direction | Current State | Main Gap | Proposed Next Step |
|---|---|---|---|
| Factorized ST attention | `ray_attention_temporal_crossview_factorized_residual_model.py` exists but lacks PP correction and validation. | Not integrated with the 9.32 mm PP baseline. | Add `PrincipalPointCorrection` to a new `...FactorizedPP` model; smoke-train and compare latency/accuracy. |
| Robust Gauss-Newton | `uncertainty_residual_learned_tri_v1_model.py` has GN but fixed damping + squared loss. | Not robust to outliers; under-trained. | Add learned per-joint damping + Huber/Geman-McClure kernel; smoke-run. |
| Skeleton-aware residual | `graph_joint_relation.py` exists; not used in residual head. | Residual MLP is per-joint. | Insert a lightweight GAT over skeleton adjacency inside the residual head. |
| AdamW+EMA+AMP | Not used in current scripts. | Training regime can be upgraded. | Switch one trainer to AdamW + cosine decay + EMA + AMP + grad clip. |
| Uncertainty + learned tri v2 | Uncertainty head exists; not converged. | No ablation of uncertainty loss weight. | Run 10-epoch smoke sweeping `uncertainty_loss_weight` ∈ {0,0.05,0.1,0.2}. |
| Multi-scale temporal | Not implemented. | Single temporal scale. | Add hierarchical temporal conv on top of current temporal module. |

### Data & Datasets (Agents 5–8)

| Dataset | State | Next Step |
|---|---|---|
| WebBridge H36M | `data/webbridge/h36m` exists; conversion scripts present. | Audit `.npz` quality, fix S9/S11, add loader integration. |
| AIST++ | `convert_aistpp_v1.py` exists; multi-view conversion partial. | Finish multi-view canonicalization, add loader. |
| Panoptic | `convert_panoptic_v1.py` exists; unused. | Add loader + evaluation smoke. |
| 3DPW | `convert_3dpw_multiview.py` exists; unused. | Convert test split, evaluate zero-shot. |

### Training & Optimization (Agents 4, 9, 12)

- **Training regime upgrade:** AdamW, cosine decay, EMA, AMP, gradient clipping is a near-zero-cost change that should be tested on the next GPU run.
- **SSL pre-training:** `ssl_dataset.py` exists; run only after visibility v2 validates.
- **Domain adaptation:** `domain_adaptation_wrapper.py` exists; try a small cross-dataset smoke.

### Evaluation (Agents 14–16, 19)

- **Cross-dataset benchmark protocol:** `motionflow_mv/eval/benchmark_protocol.py` exists but is not used by the main eval scripts.
- **Robustness matrix:** rotation is still the biggest failure; need a unified 6-axis matrix (rot, trans, focal, PP, distortion, occlusion).
- **Real-time benchmark:** no latency/FLOPs numbers for the 9.32 mm PP model.
- **SOTA comparison:** no apples-to-apples script vs. DLT/robust triangulation/Iskakov baseline.

### Paper & Integration (Agents 17–18, 20)

- **Paper draft:** `docs/paper_draft_icra_cvpr_2027.md` still advertises 10.46 mm and contains fabricated/missing citations.
- **Figures/tables:** `docs/tables/icra2027/main_results.md` is stale (11.17 mm).
- **MotionFlow plugin:** best model is not yet wrapped as a reusable `FusionModule` plugin.

---

## 3. Top-5 Prioritized Next Experiments

1. **Finish visibility v2 training & auto-evaluate** (GPU, in progress).
   - Goal: clean ≤ 9.6 mm, ≥ 10% relative gain at 30% occlusion.
   - After finish, run `scripts/eval_crossview_residual_visibility_v2_wsl.sh`.

2. **Update paper tables/figures to the 9.32 mm baseline** (CPU).
   - `experiments/convert_eval_json_to_paper_schema.py`
   - Regenerate `docs/tables/icra2027/main_results.md` and `docs/figures/icra2027/`.
   - Fix stale paper draft numbers/citations.

3. **Implement unified SOTA comparison harness** (CPU/A800-D).
   - `experiments/compare_sota_baselines.py`
   - Compare DLT, robust IRLS, Iskakov-style learned triangulation, and the PP baseline on the same split and corruptions.

4. **Add factorized ST attention + PP correction smoke** (GPU after visibility v2).
   - New model: `ray_attention_temporal_crossview_factorized_residual_principal_point_model.py`
   - Target: ≤ 0.5 mm accuracy loss, 30–50% latency reduction.

5. **Unified 6-axis robustness matrix evaluator** (CPU).
   - `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py`
   - Covers rot/trans/focal/PP/distortion/occlusion; outputs `outputs/robustness_matrix_pp_full.json`.

---

## 4. GPU Queue Recommendations

The RTX 4090 is the bottleneck. A800-D GPUs 0–3 appear lightly loaded but require coordination with the owner before writing/running jobs there. Until then, keep the 4090 queue lean:

1. Visibility v2 (running).
2. Factorized ST + PP smoke.
3. SSL pre-training on H36M (only if visibility v2 meets targets).
4. Spatiotemporal PP full run (only after factorized smoke validates).

CPU/A800-read-only tasks can run in parallel: paper figure regeneration, robustness matrix, SOTA comparison, failure analysis.

---

## 5. Immediate Action Items

- [ ] Monitor visibility v2 training to completion.
- [ ] Run auto-evaluation and update issue #21 / PR #17.
- [ ] Regenerate paper assets from the 9.32 mm baseline.
- [ ] Implement `compare_sota_baselines.py` and `eval_robustness_matrix_pp_mpiinf3dhp.py`.
- [ ] Smoke-test factorized ST + PP model once GPU is free.
- [ ] Confirm A800-D GPU availability and policy with owner for read-only data + GPU experiments.
