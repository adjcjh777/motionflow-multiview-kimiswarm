# Proposal: Semantic Action-Conditional Fusion

**Author:** Iter15 design swarm — agent task "semantic-action-conditional-fusion"  
**Date:** 2026-08-06  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  
**Related prior work in repo:** additive action embedding in `motionflow_mv/fusion/action_aware_principal_point_model.py`, action-aware dataset in `motionflow_mv/data/action_aware_dataset.py`.

---

## 1. Problem

The current anchor treats every video clip as a generic multi-view pose problem.  Human motion, however, is highly structured by action class: a "walking" clip and a "sitting" clip have very different 3-D skeleton priors, limb orientations, and likely occlusions.  The existing `ActionAwareRayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` injects an action embedding only as an simple additive bias, which is too weak to adapt the calibration, fusion, or residual refinement stages to the action semantics.

## 2. Hypothesis

Conditioning the spatio-temporal transformer and residual refinement on a discrete action label via an additive action embedding, per-layer FiLM affine modulation, and an action-aware residual head will improve multi-view 3-D skeleton fusion, physical calibration alignment, and cross-view robustness beyond the 9.32 mm anchor while adding only a small number of parameters.

## 3. Method

### 3.1 Architecture changes

Create a new model file:

- **New:** `motionflow_mv/fusion/semantic_action_conditional_fusion_model.py`
  - Subclass `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
  - Add `action_embed` table, `action_to_feat` projection, per-ST-layer FiLM generators, and an action-aware residual MLP.
  - `forward` signature extends the parent with `action_id: Optional[torch.Tensor] = None`; `None` falls back to the reserved "unknown" index, preserving the anchor behavior.
  - Inject action information at three points:
    1. **Additive embedding:** project the action vector into the feature dimension and add it to the `(B, T, V, J, d)` tokens before the spatio-temporal transformer.
    2. **FiLM modulation:** for each `st_transformer` layer, predict per-sample `(gamma, beta)` from the action embedding and apply `feat = (1 + gamma) * feat + beta` to the `(B*J, T*V, d)` tokens.
    3. **Action-conditional residual refinement:** concatenate the pooled spatio-temporal feature, the raw triangulated 3-D pose, and a projected action embedding, then feed a residual MLP with output `3`.

Class signature:

```python
class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSemanticActionConditional(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    def __init__(
        self,
        num_actions: int = 16,
        action_embed_dim: Optional[int] = None,
        **kwargs,  # forwarded to the PP anchor
    ):
        ...
```

### 3.2 Loss / data changes

No loss change is required for the smoke; keep the standard MPJPE objective.  The action label is already produced by `motionflow_mv/data/action_aware_dataset.py` (`collate_fn` returns the `action` tensor).  For the full run, optionally add a small action-classification auxiliary loss on a per-clip action head if action labels are noisy; this is left for a follow-up ablation.

### 3.3 Files to create or modify

- **Create:** `motionflow_mv/fusion/semantic_action_conditional_fusion_model.py`
- **Create (post-smoke, optional):** `motionflow_mv/fusion/semantic_action_conditional_fusion_module.py` — a thin `FusionModule` wrapper so the model can be loaded by the standard harness.
- **Create (post-smoke, optional):** `experiments/train_semantic_action_conditional_pp_smoke_mpiinf3dhp.py` — copy of the PP anchor smoke script, passing `action` to the model.

No existing files need modification for the skeleton to run.

## 4. Smoke-Test Plan

Run a -epoch CPU/GPU smoke on a small action-labelled MPI-INF-3DHP or H36M split, reusing the existing action-aware dataloader.

| Setting | Value |
|---|---|
| Dataset | `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz` (action labels parsed as generic / known if available) |
| Samples | 500 random clips |
| Clip length | 9 or 13 |
| Batch size | 4 |
| Model dims | `d=32`, `residual_hidden=64`, `n_st_layers=2`, `action_embed_dim=32` |
| Optimizer | Adam, lr=1e-3 |
| Loss | MPJPE only |
| Epochs | 5 |

**Pass/fail criteria:**

- **Pass:** training completes with no NaNs / crashes and val MPJPE ≤ 12 mm.
- **Pass:** forward and backward pass both produce finite gradients for all trainable parameters.
- **Pass:** model output shapes are correct for both 17-joint and 28-joint skeletons.
- **Fail:** val MPJPE > 15 mm, any NaN/Inf, or action embedding causes training instability.

## 5. Evaluation Plan

If the smoke passes, evaluate with the standard harness:

1. **Clean accuracy on MPI-INF-3DHP S2/Seq1:**
   - `python experiments/eval_full_metrics.py --model semantic_action_conditional_pp --checkpoint <smoke_checkpoint> ...`
   - Report MPJPE, PA-MPJPE, PCK@50/100/150, AUC.
   - Target: clean MPJPE ≤ 9.5 mm (within 0.2 mm of the 9.32 mm anchor).

2. **Robustness matrix:**
   - Run the existing robustness script with `view_dropout_0.2` and `joint_dropout_0.2`.
   - Target: degradation relative to clean is not more than 10 percentage points worse than the anchor, especially for action classes with heavy self-occlusion.

3. **Ablation:**
   - Compare against the simpler additive-only action-aware model and against the unconditioned anchor on the same split.
   - Pass if FiLM + action-aware residual improves over additive-only by ≥ 0.2 mm.

## 6. Estimated GPU/CPU Cost on RTX 4090

| Phase | Hardware | Time |
|---|---|---|
| Smoke (5 epochs, 500 samples) | RTX 4090 | ~20–30 min |
| CPU sanity (`python -m motionflow_mv.fusion.semantic_action_conditional_fusion_model`) | CPU | < 1 min |
| Full training (20–50 epochs, full split) | RTX 4090 / A800-D | ~4–8 h on RTX 4090 |
| Clean eval + robustness matrix | RTX 4090 or CPU | ~10–20 min |

The new layers add only the action embedding table (~1k parameters), three small linear projections, and the extra residual input dimension, so throughput and memory are essentially identical to the PP anchor.

## 7. Risks & Fallback

| Risk | Likelihood | Mitigation / Fallback |
|---|---|---|
| Action labels are unavailable at inference (in-the-wild data). | Medium | The model uses a reserved "unknown" action index at test time, which falls back to anchor-like behaviour.  If accuracy drops, freeze the action-conditional layers and use the learned average embedding. |
| FiLM overfits the small smoke dataset. | Medium | Add action embedding dropout (e.g. `p=0.1`) or reduce `action_embed_dim` to `d//2`; if no improvement, keep only the additive embedding path. |
| Action-conditional layers destabilize the residual head. | Low | The action embedding is concatenated, not added, to the residual input to avoid scale coupling; if NaNs appear, clip gradients or lower lr to 3e-4. |
| No improvement over the 9.32 mm anchor. | Medium-High | The change is modular: drop FiLM and keep only the additive action embedding, or revert to the unconditioned anchor entirely. |

---

## Summary

Add an action-conditional variant of the PP anchor that injects a discrete action label into the spatio-temporal transformer through an additive embedding, per-layer FiLM, and an action-aware residual head.  The implementation is a single new model file, requires no data changes, and is validated by a 5-epoch smoke before any full run.
