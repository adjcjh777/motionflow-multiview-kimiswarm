# GitHub Issue #70 — Swarm Iteration 18 / OmniMultiViewFusion Roadmap

> **Labels:** `enhancement`, `research`, `swarm-iter18`, `omniview`
> **Assignees:** motionflow-research-lead
> **Milestone:** ICRA/CVPR 2027 submission
> **Branch:** `feat/swarm-iter18-omniview`

---

## 1. Update on issue #70 (P17 tracking issue)

This issue replaces/updates the previous P17 ("Action semantics / category prior") placeholder with a higher-priority, unified architecture direction. After the iter17 ensemble run pushed the MPI-INF-3DHP S2/Seq1 benchmark to **8.35 mm MPJPE**, the remaining single-model gap to a true ICRA/CVPR 2027 publishable story is no longer raw accuracy but **unified robustness**: a single model that simultaneously handles variable views, occlusion, calibration drift, and cross-dataset transfer. The working name for this direction is **OmniMultiViewFusion**.

### Why P17 was repurposed

- The original P17 (action-semantics category prior) was deprioritised in the iter17 direction review because the action-aware PP model had no runnable best-model training path and marginal expected gain versus the current 8.35 mm ensemble.
- The current best ensemble already combines `bayesian_tri_v2_stabilized` + `bayesian_tri_v2_aug` (d=128, h=256). Pushing a single model below the ensemble while adding robustness requires fusing the strongest isolated modules into one trainable architecture.
- Therefore, P17 is re-scoped to **OmniMultiViewFusion**: a single multi-view fusion backbone that subsumes principal-point/focal correction, visibility gating, graph-joint refinement, uncertainty-weighted triangulation, and cross-view spatio-temporal attention.

---

## 2. Current best result (baseline for this issue)

| Dataset / Split | Model | MPJPE (mm) | PA-MPJPE (mm) | Params | Checkpoint |
|---|---|---:|---:|---:|---|
| MPI-INF-3DHP S2/Seq1 | Bayesian Tri v2 ensemble (stabilized + aug, d=128) | **8.35** | 5.29 | ~1.06 M | `outputs/bayesian_tri_v2_ensemble_best.pth` |
| MPI-INF-3DHP S2/Seq1 | Bayesian Tri v2 stabilized (single, d=128) | 9.03 | 5.69 | ~1.06 M | `outputs/bayesian_tri_v2_stabilized.pth` |
| Human3.6M S5/Act2 | Cross-view residual + PP (d=64, h=128) | 5.24 | 4.84 | 243 k | `outputs/ray_attention_temporal_crossview_residual_pp_h36m.pth` |

Source: `docs/results_icra_cvpr_2027.md` and `docs/icra_cvpr_2027_paper_story.md`.

---

## 3. OmniMultiViewFusion — new issue content

### 3.1 Motivation

Current modules are strong but isolated:

- `PrincipalPointCorrection` fixes small calibration drift.
- `VisibilityGatedFusion` handles occluded views.
- `CrossviewResidualUncertaintyModel` predicts per-view log-variance.
- `GraphJointRelation` / `SkeletonGraphResidualRefiner` enforces skeleton-aware reasoning.
- `RayAttentionSpatiotemporalModel` models `(time, view)` interactions.

No single model combines all of them. The hypothesis is that a unified architecture can:

1. Match or beat the 8.35 mm ensemble with a **single model**.
2. Push clean MPJPE below **8.0 mm**.
3. Improve robustness under view dropout / occlusion / calibration error to publishable levels.
4. Provide a single `MultiViewFusionPlugin` entry point instead of multiple partial solutions.

### 3.2 Proposed architecture: `OmniMultiViewFusion`

```
Input: (B, T, V, J, 3)  -> 2D + confidence
        |
        v
PrincipalPointCorrection(K) + optional focal-scale head  ->  corrected intrinsics
        |
        v
Ray/camera embedding encoder (per-frame, per-view, per-joint)
        |
        v
Visibility head  ->  per-view, per-joint visibility multiplier m_vj in [0,1]
        |
        v
Factorised (T × V × J) transformer block
   - temporal layers  (attention over T)
   - view layers        (attention over V, masked by visibility)
   - joint layers       (graph-constrained attention over J)
        |
        v
Weight head  ->  per-view, per-joint fusion weight w_vj
        |
        v
Uncertainty head  ->  per-view, per-joint log-variance λ_vj
        |
        v
Bayesian precision-weighted triangulation  ->  DLT / Gauss-Newton
        |
        v
Graph-joint residual refinement  ->  final 3D pose
```

### 3.3 Key design decisions

| Decision | Rationale | Reference file |
|---|---|---|
| **Visibility before the transformer** | Multiplies occluded views out before view attention; provides auxiliary BCE loss. | `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_visibility_model.py` |
| **Graph constraint in joint layers** | Replaces dense self-attention with anatomical-edge propagation. | `docs/swarm_iter_next/design_graph_joint_relation/` |
| **Bayesian precision weighting** | Anisotropic covariance → precision → DLT weights. | `docs/icra_cvpr_2027_paper_story.md` §4.1 |
| **Adaptive Gauss-Newton refinement** | 1–2 differentiable GN steps with learned per-joint damping. | `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py` |
| **Principal-point + focal correction head** | Self-corrects calibration drift before triangulation. | `motionflow_mv/fusion/principal_point_correction.py` |

### 3.4 Training recipe (draft)

```bash
python experiments/train_omnimultiview_mpiinf3dhp.py \
    --train_subj 1 3 \
    --val_subj 2 --val_seq 1 \
    --d_model 128 --residual_hidden 256 \
    --warm_start outputs/bayesian_tri_v2_stabilized.pth \
    --freeze_encoder_epochs 5 \
    --epochs 30 \
    --view_dropout_rate 0.3 --min_views 2 \
    --visibility_loss_weight 0.1 \
    --uncertainty_loss_weight 0.05 \
    --graph_num_layers 2 \
    --output outputs/omnimultiview_mpiinf3dhp.pth
```

Losses:
- `L_3d_mse` (main)
- `L_visibility_bce` (aux)
- `L_uncertainty_nll` (aux)
- `L_bone_length` (aux)
- `L_velocity_smoothness` (aux)

### 3.5 Evaluation plan

1. **Clean accuracy:** MPI-INF-3DHP S2/Seq1 MPJPE / PA-MPJPE.
2. **Variable-view:** MPJPE@k for k = 2..14.
3. **Robustness matrix:** rotation ±0.5°/±1.0°, translation ±5 mm/±10 mm, focal ±1%/±2%, principal point ±3 px/±5 px/±10 px, joint occlusion 10/20/30%, view dropout 10/30/50%.
4. **Cross-dataset zero-shot:** Human3.6M S9/S11 and WebBridge s2/v14, s3/v14, s1/v4.
5. **Runtime/latency:** RTX 4090 single-batch latency and throughput.

### 3.6 Target metrics vs. current best

| Metric | Current best | OmniMultiViewFusion target |
|---|---|---:|
| MPI-INF-3DHP S2/Seq1 MPJPE | 8.35 mm (ensemble) | ≤ 8.0 mm (single model) |
| MPI-INF-3DHP S2/Seq1 PA-MPJPE | 5.29 mm | ≤ 5.0 mm |
| View dropout 30% | 18.15 mm | ≤ 14.0 mm |
| Joint occlusion 30% | 16.99 mm | ≤ 13.0 mm |
| Principal point ±5 px | 13.87 mm (PP full 20 ep) | ≤ 12.0 mm |
| Variable-view k=4 | 73.90 mm | ≤ 60.0 mm |

---

## 4. Deliverables for this issue

### 4.1 Design / docs

- `docs/swarm_iter18/P17_github_issue_draft.md` — this issue draft.
- `docs/design_omniview_fusion.md` — original OmniMultiViewFusion design sketch (updated).
- `docs/swarm_iter18/omnimultiview_fusion_plan.md` — detailed implementation plan (to be written).

### 4.2 Code / prototypes

- `experiments/prototypes/swarm_iter18/omni_multiview_fusion.py` — skeleton implementation of the unified model.
- `experiments/train_omnimultiview_mpiinf3dhp.py` — full training script.
- `experiments/eval_omnimultiview_robustness.py` — robustness-matrix evaluation.
- `motionflow_mv/fusion/omni_multiview_fusion.py` — production module (after prototype is validated).

### 4.3 Smoke tests / validators

- `docs/swarm_iter18/validate_p17_issue_draft.py` — checks this draft for required sections and referenced files.

---

## 5. Known blockers and risks

| Blocker / Risk | Impact | Mitigation |
|---|---|---|
| Single RTX 4090 GPU queue is full | Full d=128 30-epoch run may take >1 week | Start with d=64 smoke; use gradient checkpointing; queue overnight |
| Memory blow-up from full (T × V × J) attention | O(T·V·J) attention is expensive | Factorise attention; use FlashAttention/SDPA; start with T=9, V=14, J=28 |
| Negative clean-MPJPE interaction when combining modules | Ensemble may still beat single model | Warm-start + freeze encoder phase; staged unfreezing |
| Uncertainty head diverges | NaN/instability | Clamp log-variance to [-5, 5]; use uncertainty NLL auxiliary loss |
| Graph attention incompatible with variable views | Edge index breaks when views drop | Rebuild edge index for active subset; use visibility mask instead of hard removal |

---

## 6. Next steps

- [ ] Finalise `OmniMultiViewFusion` model skeleton and run CPU forward-pass smoke test.
- [ ] Wire warm-start from `bayesian_tri_v2_stabilized.pth` and freeze encoder for first 5 epochs.
- [ ] Run d=64 smoke training on MPI-INF-3DHP S1/S3 (5 epochs) and validate no NaNs.
- [ ] Run d=128 full training (30 epochs) and evaluate clean + robustness matrix.
- [ ] Generate variable-view MPJPE@k curve and paper-ready robustness figures.
- [ ] Draft PR with the new module, training/eval scripts, and updated docs.
- [ ] Close this issue when the single-model MPJPE ≤ 8.0 mm target is reached or the direction is deprioritised.

---

## 7. Related files and references

- `docs/design_omniview_fusion.md`
- `docs/results_icra_cvpr_2027.md`
- `docs/icra_cvpr_2027_paper_story.md`
- `docs/swarm_iter_next/synthesis_2026_08_07.md`
- `docs/swarm_iter_next/20_agent_direction_review.md`
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`
- `motionflow_mv/fusion/principal_point_correction.py`
- `experiments/prototypes/swarm_iter18/omni_multiview_fusion.py`
- `experiments/prototypes/swarm_iter18/validate_p17_issue_draft.py`

---

*Last updated: 2026-08-07 on branch `feat/swarm-iter18-omniview`.*
