# WebBridge Scaling Plan Proposal — MotionFlow-MultiView

**Date:** 2026-08-07  
**Target:** ICRA / CVPR 2027  
**Scope:** Concrete, code-grounded next steps for scaling OmniMultiViewFusion v2/v3 experiments across the WebBridge multi-view pose corpus.

---

## 1. Current State (Ground Truth from Repo)

- **Best anchor:** Bayesian Tri v2 ensemble at **8.35 mm MPJPE / 5.29 mm PA-MPJPE** on MPI-INF-3DHP S2/Seq1 (`docs/results_icra_cvpr_2027.md`).
- **Best single model:** Bayesian Tri v2 single-model **~9.03 mm** (`docs/swarm_iter19/20_next_iteration_synthesis.md`).
- **Running experiments:**
  - RTX 4090 WSL: `OmniMultiViewFusionV2` no-graph ablation (`scripts/run_omniview_fusion_v2_full_wsl.sh`).
  - A800-D: `OmniMultiViewFusionV2` and `OmniMultiViewFusionV3` restarted on GPUs 4/5 (`docs/swarm_iter19_status.md`).
- **Data readiness:** WebBridge canonical `.npz` files exist for MPI-INF-3DHP, H36M (meter units), AIST++, 3DPW, Shelf/Campus; split manifests are under `configs/splits/` (`docs/swarm_iter18/P07_webbridge_manifest.md`).
- **Known blockers:**
  - `configs/benchmark_webbridge_h36m_test_smoke.yaml` is missing `source_n_views: 14`, so MPI-trained 14-view checkpoints load incorrectly on 4-view H36M.
  - `OmniMultiViewFusionV2` is instantiated once per run and rejects mixed `(n_views, n_joints)` manifests (`experiments/train_omniview_fusion_v2_webbridge_multi.py:503–552`).
  - Variable-view inference is not mask-aware in graph attention (`docs/swarm_iter19/01_omniv2_architecture_review.md` §Key findings).  
  - No multi-GPU / distributed training harness in `TrainerV2` (`motionflow_mv/training/trainer_v2.py`).

---

## 2. Guiding Principles

1. **Evidence before capacity.** Earlier 1.06 M-parameter models underperformed 243 k-parameter models; do not scale width/depth blindly (`docs/swarm_iter19/14_model_scaling_plan.md`).
2. **One variable at a time.** Each GPU experiment must isolate a single change and be compared against the anchor.
3. **Fail fast on CPU.** New modules must pass CPU smoke tests before GPU training.
4. **Reproducibility gates.** Any new anchor must pass: clean MPJPE, robustness matrix, variable-view curve, and ≥3 repeated seeds.

---

## 3. Immediate Action Items (3–5)

1. **Finish and evaluate the running no-graph ablation, then queue the dense+graph v2 full run.**
   - Code: `motionflow_mv/fusion/omniview_fusion_v2.py`, `scripts/run_omniview_fusion_v2_dense_graph_v2_full_wsl.sh`.
   - Decision gate: if no-graph clean MPJPE < 9 mm, run graph ablation; otherwise fall back to Bayesian Tri v2 improvements.

2. **Fix the H36M cross-dataset benchmark config and run the first MPI→H36M transfer baseline.**
   - File: `configs/benchmark_webbridge_h36m_test_smoke.yaml`.
   - Add `source_n_views: 14` to `model_config`, then run `scripts/run_webbridge_h36m_test_smoke.sh`.

3. **Make graph attention / ST transformer view-mask aware so variable-view inference is reliable.**
   - Files: `motionflow_mv/fusion/graph_joint_attention_v2.py`, `motionflow_mv/fusion/omniview_fusion_v2.py`.
   - Pass the active-view mask through `_apply_graph_joint_attention` and the ST transformer and zero out messages from dropped views.

4. **Enable true WebBridge mixed-dataset training on a common 17-joint skeleton.**
   - Files: `motionflow_mv/data/webbridge_mixed_dataset.py`, `experiments/train_omniview_fusion_v2_webbridge_multi.py`.
   - Either make `OmniMultiViewFusionV2` accept padded variable-view input (`MAX_VIEWS=14`, actual views via mask) or add a per-skeleton wrapper so H36M/AIST++/Shelf/Campus can train jointly.

5. **Add multi-GPU / distributed training and a runtime budget guard to the v2/v3 trainers.**
   - File: `motionflow_mv/training/trainer_v2.py`.
   - Wrap model with `DistributedDataParallel`, aggregate metrics across ranks, and report params/FLOPs/wall-time per epoch (reuse `experiments/prototypes/swarm_iter18/profile_bayesian_tri_v2_latency.py`).

---

## 4. Detailed Scaling Plan

### 4.1 Model Scaling: v2 → v3

**Current architecture** (read from `motionflow_mv/fusion/omniview_fusion_v2.py` and `omniview_fusion_v3.py`):

- v2: visibility gating + graph-joint attention + anisotropic covariance + adaptive Gauss-Newton + residual MLP.
- v3: v2 + hierarchical multi-scale fusion + camera conditioning + epipolar-biased ST attention.  Each component is independently toggled by boolean flags.

**Recommended sequence:**

| Step | Experiment | Code entry | Validation |
|------|-----------|------------|------------|
| 1 | Complete no-graph v2 ablation | `scripts/run_omniview_fusion_v2_full_wsl.sh --graph_num_layers 0` | Full S2/Seq1 MPJPE + robustness + variable-view |
| 2 | Dense+graph v2 full run | `scripts/run_omniview_fusion_v2_dense_graph_v2_full_wsl.sh` (graph_num_layers=1, n_joint_layers=1) | Same as above; compare to no-graph |
| 3 | v3 ablation grid (multiscale / camera / epipolar) | `experiments/train_omniview_fusion_v3_mpiinf3dhp.py --use_multiscale_fusion --use_camera_conditioning --use_epipolar_bias` | Per-component MPJPE; warm-start from best v2 checkpoint with `strict=False` |
| 4 | Capacity sweep only if steps 1–3 justify it | sweep `d{96,160}`, `residual_hidden∈{128,256}`, `n_st_layers∈{2,3}` | Each run ≤1.3× wall-time of d=128 baseline |

**Concrete code changes needed:**

- In `motionflow_mv/fusion/omniview_fusion_v2.py`:
  - Add `view_mask` argument to `_apply_graph_joint_attention` and zero-out edges from inactive views.
  - Forward the same mask to the ST transformer so attention scores from dropped views are set to `-inf`.
- In `motionflow_mv/fusion/omniview_fusion_v3.py`:
  - Apply the same view-mask logic inside `_HierarchicalMultiscaleFusion` (scale branches should not pool from dropped views).
  - Verify `EpipolarBiasedTransformerEncoderLayer` handles variable `V` when `use_epipolar_bias=True`.

### 4.2 Data Scaling: WebBridge Mixed-Dataset

**Current mixed loader** (`motionflow_mv/data/webbridge_mixed_dataset.py`):

- Already maps H36M/AIST++/Shelf/Campus/MPI to a common 17-joint skeleton.
- Pads views to `MAX_VIEWS=14` with identity/zero placeholders.
- Returns `(x, y, K, R, t, dataset_id)`.

**Blocker in the trainer** (`experiments/train_omniview_fusion_v2_webbridge_multi.py`):

- `_validate_dataset_consistency` forbids mixed `(n_views, n_joints)` and builds a model for one `(V, J)` pair only.

**Implementable options:**

A. **Variable-view model (preferred for true scaling).**
   - Modify `OmniMultiViewFusionV2`/`V3` to always accept `MAX_VIEWS=14` and accept a `view_mask` (or infer it from zero confidences).
   - All attention / fusion blocks respect the mask.
   - Triangulation only uses active views.
   - This is the clean long-term path but touches more code.

B. **Per-skeleton training wrapper (faster to ship).**
   - Keep the existing single-model-per-run constraint but allow manifests to be split by `(V, J)`.
   - Train one model per skeleton (MPI 28-joint/14-view, H36M 17-joint/4-view, AIST 17-joint/9-view) and report per-dataset plus macro-average metrics.
   - This matches the current `configs/splits/webbridge_all_train.yaml` note and unblocks the cross-dataset benchmark quickly.

**Recommended next step:** implement option B first (minimal code), then option C (common 17-joint mixed loader) in parallel with the v3 ablations.

### 4.3 Evaluation Scaling

The project already has the pieces; scaling means making them routine for every anchor candidate:

| Deliverable | Existing artifact | Gap | Concrete next step |
|-------------|-------------------|-----|---------------------|
| Robustness matrix | `experiments/eval_omniview_fusion_v2_camera_perturbation.py`, `eval_full_metrics.py` | Not automated for every checkpoint | Add `--robustness` default in `scripts/eval_omniview_fusion_v2_wsl.sh` |
| Variable views | `experiments/eval_omniview_fusion_v2_variable_views.py` | Needs view-mask fix first | Run MPJPE@k (k=2..14) after graph/ST masking is fixed |
| Repeated seeds | `experiments/run_repeated_seeds_benchmark.py` | Not wired to Omni v2/v3 | Create `scripts/run_omniview_fusion_v2_repeated_seeds.sh` |
| Cross-dataset benchmark | `experiments/run_webbridge_benchmark.py` | H36M config broken | Fix `source_n_views: 14`, run, then extend to AIST++/Shelf/Campus |
| Test-set inference | `experiments/infer_mpiinf3dhp_test_set_omniview_v2.py` | Needs v3 variant | Add `--model_type v3` flag and checkpoint loading |

### 4.4 Training Infrastructure Scaling

- **Distributed training:** add DDP to `TrainerV2` or wrap the trainer in a small launcher.  The A800 has headroom and multi-GPU scaling will matter for v3 and mixed-dataset runs.
- **Runtime budget guard:** before any larger model is committed, run `experiments/prototypes/swarm_iter18/profile_bayesian_tri_v2_latency.py` (or a v2/v3 equivalent) and reject configs that exceed 1.3× the d=128 baseline wall-time.
- **Checkpoint manifest:** every run should write a JSON with model hparams, seed, data split, and final metrics so `experiments/run_repeated_seeds_benchmark.py` can consume it.

### 4.5 Test Coverage Scaling

The tests review (`tests/`) identified gaps. Add:

1. `tests/test_webbridge_mixed_dataset.py` — verify 28→17 joint re-indexing, view padding, collate with `dataset_id`.
2. `tests/test_omniview_fusion_v2_variable_views.py` — smoke test with `V=14` input and active views < 14.
3. `tests/test_graph_joint_attention_view_mask.py` — assert dropped-view nodes do not propagate.
4. `tests/test_omniview_fusion_v3_components.py` — toggle multiscale/camera/epipolar flags and verify forward pass and shape.
5. `tests/test_trainer_v2_ddp.py` — minimal 2-rank smoke (CPU) for distributed wrapper.

---

## 5. Risk Register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| No-graph v2 does not beat 9 mm | Medium | Fall back to Bayesian Tri v2 + calibration curriculum / visibility gating per `docs/iter_next_action_plan.md` P0.2–P0.3. |
| v3 multi-scale/epipolar bias adds instability | Medium | Warm-start from best v2 checkpoint; toggle one flag at a time; use `--smoke` first. |
| H36M cross-dataset remains broken | Low (known fix) | Add `source_n_views: 14` and validate before any large mixed run. |
| Graph attention breaks variable-view | Medium | Add explicit view-mask tests before declaring variable-view results. |
| Mixed-dataset unit mismatch | Medium | Enforce `_m.npz` (meter) canonical files; reject millimeter-only datasets in mixed loader. |
| DDP memory / deadlock | Low | Start with 2-GPU smoke; use `find_unused_parameters=False` unless warm-start mismatch requires it. |

---

## 6. Definition of Done for This Plan

- [ ] `docs/webbridge_scaling_plan_proposal.md` committed (this file).
- [ ] `configs/benchmark_webbridge_h36m_test_smoke.yaml` updated with `source_n_views: 14`.
- [ ] CPU smoke test added for WebBridge mixed 17-joint loader.
- [ ] No-graph ablation evaluated and decision recorded.
- [ ] Dense+graph v2 full run queued or deprioritized based on no-graph result.
- [ ] View-mask support added to graph/ST attention with a passing test.
- [ ] DDP launcher sketched or integrated for A800 multi-GPU runs.

---

## 7. References

- `motionflow_mv/fusion/omniview_fusion_v2.py` — v2 implementation.
- `motionflow_mv/fusion/omniview_fusion_v3.py` — v3 implementation.
- `motionflow_mv/data/webbridge_mixed_dataset.py` — 17-joint mixed loader.
- `experiments/train_omniview_fusion_v2_webbridge_multi.py` — multi-dataset v2 trainer.
- `experiments/train_omniview_fusion_v3_mpiinf3dhp.py` — v3 trainer.
- `docs/iter_next_action_plan.md` — ranked P0/P1/P2 actions.
- `docs/swarm_iter19/14_model_scaling_plan.md` — evidence-based scaling rules.
- `docs/swarm_iter19/20_next_iteration_synthesis.md` — synthesis of iter19 swarm outputs.
- `docs/swarm_iter19/01_omniv2_architecture_review.md` — v2 architecture review and known issues.
- `docs/omniview_fusion_v3_design.md` — v3 design rationale and ablation table.
