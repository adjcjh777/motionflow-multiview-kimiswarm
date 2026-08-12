# OmniMultiViewFusion v4 — Architecture Design Proposal

**Target venues**: ICRA / CVPR 2027  
**Status**: Design proposal — read-only, no running training touched  
**Date**: 2026-08-07  

## 1. Where we are

The project currently has two converging architecture lines:

* **OmniMultiViewFusion v2** (`motionflow_mv/fusion/omniview_fusion_v2.py`) unites visibility gating, graph-joint attention, anisotropic covariance / uncertainty-weighted DLT, adaptive Gauss-Newton refinement, and a spatio-temporal transformer.
* **OmniMultiViewFusion v3** (`motionflow_mv/fusion/omniview_fusion_v3.py`) extends v2 with hierarchical multi-scale temporal/cross-view fusion, camera-parameter conditioning, and epipolar-biased cross-view attention (see `docs/omniview_fusion_v3_design.md`).

Latest project status (`docs/swarm_iter_next/synthesis_2026_08_07.md`):

| Item | Value |
|------|-------|
| Best single model | **9.03 mm** MPJPE (`bayesian_tri_v2_stabilized`) |
| Ensemble result | **8.61 mm** MPJPE |
| Largest clean gaps | view dropout 30 % → 18.15 mm; joint occlusion 30 % → 16.99 mm |
| Calibration gaps | rot 0.5° → ~16.9 mm; focal 1 % → ~19.1 mm; pp ±10 px catastrophic |

**Goal of v4**: preserve the clean-accuracy path (target single-model **< 8.6 mm**) while closing the occlusion, calibration-robustness, and cross-dataset generalisation gaps through a single, modular, end-to-end architecture.

## 2. Guiding principles for v4

1. **No speculative stacking.** Every new module must be individually togglable and warm-startable from a v2/v3 checkpoint.
2. **Reuse proven components.** v4 should be the consolidation of the highest-ROI directions already prototyped in the repo, not a ground-up redesign.
3. **Fail fast on CPU.** Each new block must ship with a `__main__` smoke test and a pytest file before being queued on GPU.
4. **Geometry-aware throughout.** Calibration should be treated as a learnable, robust signal, not a hard assumption.

## 3. Proposed v4 architecture

### 3.1 High-level block diagram

```
Input (B, T, V, J, 3) + cameras / K, R, t
        │
        ▼
┌─────────────────────────────────────────────┐
│ Principal-point + focal correction (v2/v3)  │  ← motionflow_mv/fusion/principal_point_correction.py
│ Optional rotation correction head (new)       │
└─────────────────────────────────────────────┘
        │
        ▼
Per-frame ray-aware feature extraction (v2/v3)
        │
        ▼
Graph-joint attention v2 (v2/v3)
        │
        
Camera conditioning + epipolar bias (v3)
        │
        ▼
Hierarchical multi-scale fusion (v3)
        │
        ▼
Spatio-temporal (time × view) transformer (v2/v3)
        │
        ▼
┌─────────────────────────────────────────────┐
│ Context-aware visibility gating v2          │  ← replaces simple sigmoid head
│ + adaptive view selection (hard top-k)      │  ← optional, Gumbel-softmax
│ + per-view log-variance / covariance        │  ← existing v2 heads
└─────────────────────────────────────────────┘
        │
        ▼
Uncertainty-weighted DLT + adaptive Gauss-Newton (v2/v3)
        │
        ▼
Skeleton-graph residual refinement (new default)
        │
        ▼
Kinematic-chain graph refiner (optional final pass)
        │
        
Output: pred_3d, weights, visibility, covariance, epi_loss, entropy_loss
```

### 3.2 Concrete code additions

| New / changed file | Purpose | Grounded in |
|--------------------|---------|-------------|
| `motionflow_mv/fusion/omniview_fusion_v4.py` | Main v4 model, subclasses `OmniMultiViewFusionV3` | `motionflow_mv/fusion/omniview_fusion_v3.py` |
| `motionflow_mv/fusion/visibility_gated_fusion_v2.py` (reused) | Context-aware visibility head with uncertainty | already exists; replace v2's inline `_visibility_multiplier` (`omniview_fusion_v2.py:227`) |
| `motionflow_mv/fusion/skeleton_graph_residual_refiner.py` (reused) | Drop-in replacement for the dense `residual_mlp` | already exists; replace `self.residual_mlp` in v2/v3 |
| `motionflow_mv/fusion/kinematic_chain_graph_refiner.py` (reused) | Optional final skeleton-aware pose refiner | already exists; apply after `pred_3d` |
| `motionflow_mv/fusion/adaptive_view_selector.py` | Hard budgeted view selection via Gumbel-softmax | design in `docs/swarm_iter_next/design_adaptive_view_selection/report.md` |
| `motionflow_mv/fusion/rotation_correction.py` | Learned per-view SO(3) residual before triangulation | extends `principal_point_correction.py` pattern |
| `experiments/train_omniview_fusion_v4_webbridge_multi.py` | Multi-dataset v4 trainer | mirrors `experiments/train_omniview_fusion_v2_webbridge_multi.py` |
| `experiments/eval_omniview_fusion_v4_mpiinf3dhp.py` | Robustness + variable-view eval | mirrors `experiments/eval_omniview_fusion_v2_mpiinf3dhp.py` |
| `tests/test_omniview_fusion_v4.py` | CPU smoke / shape / gradient tests | follows `tests/test_train_omniview_fusion_v2_smoke.py` |

## 4. Detailed component design

### 4.1 Visibility-gated fusion v2 (high priority)

**Problem.** The current v2 head (`omniview_fusion_v2.py:131`) predicts visibility from a single per-view token and a fallback guard. It is already effective, but the standalone `VisibilityGatedFusionV2` (`visibility_gated_fusion_v2.py`) additionally:

* conditions each view on the per-joint mean-pooled context across views,
* optionally predicts an uncertainty channel that scales the soft mask,
* and keeps the same fallback guard.

**v4 change.** Replace `self.visibility_head` in v2/v3 with `VisibilityGatedFusionV2(d, n_views, use_context=True, use_uncertainty=True)`. The forward signature stays compatible: it still returns `(N, V, J)` visibility and accepts `(N, V, J, d)` features plus confidences.

**Expected impact.** Directly addresses the 30 % view-dropout gap (18.15 mm) and the 30 % joint-occlusion gap (16.99 mm) by letting the model explicitly learn when a view is occluded.

### 4.2 Skeleton-graph residual refiner (high priority)

**Problem.** The residual refinement head in v2/v3 is a dense MLP: `delta = residual_mlp([feat_pooled, pred_3d_gn])`. It has no anatomical structure, so it can introduce bone-length violations.

**v4 change.** Replace the dense `residual_mlp` with `SkeletonGraphResidualRefiner` (`skeleton_graph_residual_refiner.py:22`). It:

* projects the concatenated `[feat_pooled, pred_3d_gn]` to a hidden dim,
* runs `GraphJointRelation` over bone/symmetry/self-loop edges,
* projects back to a 3-D residual.

This is a drop-in replacement: the input is still `(B*T, J, d+3)` and the output is `(B*T, J, 3)`.

### 4.3 Kinematic-chain final refiner (medium priority)

**Problem.** Even after residual refinement, per-joint errors can remain on distal limbs.

**v4 change.** Add an optional `KinematicChainGraphRefinerTemporal` after the residual head (`kinematic_chain_graph_refiner.py:140`). It operates purely on the output 3-D skeleton `(B, T, J, 3)` and is therefore compatible with any upstream model. The module is tiny and can be toggled with `use_kinematic_refiner=True`.

### 4.4 Adaptive view selection (medium priority)

**Problem.** v2/v3 always fuse all `V` views. At high occlusion rates, the model would benefit from explicitly selecting a small, reliable subset.

**v4 change.** Introduce an `AdaptiveViewSelector` after the ST transformer. Following the existing design (`docs/swarm_iter_next/design_adaptive_view_selection/report.md`):

* Per `(view, joint)` token, predict a selection score.
* During training, sample a soft mask with Gumbel-softmax + straight-through.
* During inference, keep the top-`k` views per joint.
* Multiply the existing triangulation weights by the selection mask.

Add a `budget_loss = (sum(mask) - k)^2` to train the selection. Default `k=V` (no regression), ablate with `k ∈ {2, 3, 4}`.

### 4.5 Calibration robustness

The current `PrincipalPointCorrection` (`principal_point_correction.py`) already supports small principal-point and focal corrections. v4 extends this with:

1. **Enable focal correction.** In v2/v3 `focal_max_scale` is passed through but often left at `0.0`. v4 default: `focal_max_scale=0.05`.
2. **Rotation correction head (new).** Predict a bounded `so(3)` residual `ΔR_v` per view from pooled per-view features and apply it to `R` before triangulation. Bound with `tanh` to stay near identity at init. This directly targets the rot-0.5° robustness regression.
3. **Curriculum augmentation.** The trainer should apply progressive rotation/focal/PP perturbations during training (already planned in `docs/iter_next_action_plan.md` P0.2). v4 trainer exposes flags: `--perturb_rot_deg`, `--perturb_focal_pct`, `--perturb_pp_px`, and a curriculum schedule.

### 4.6 Attention-entropy regularization

**Problem.** The per-view weight distribution can be too diffuse, hurting interpretability and robustness.

**v4 change.** Add an optional entropy loss on the normalised per-view triangulation weights, as prototyped in `RayAttentionFusionModelHierarchicalAttentionEntropyReg` (`ray_attention_hierarchical_attention_entropy_reg_model.py`). The loss is:

```python
p = weights / (weights.sum(dim=-3, keepdim=True) + 1e-8)
entropy = -(p * torch.log(p + 1e-8)).sum(dim=-3).mean()
loss = loss + attention_entropy_weight * entropy
```

Default `attention_entropy_weight=0.0`; ablate at `0.01`.

## 5. v4 class signature (proposal)

```python
class OmniMultiViewFusionV4(OmniMultiViewFusionV3):
    def __init__(
        self,
        # v2/v3 inherited args
        j=17, d=64, n_views=4, n_heads=4,
        n_joint_layers=0, n_st_layers=2,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
        # v4 toggles
        use_context_visibility=True,      # VisibilityGatedFusionV2
        use_skeleton_residual=True,         # SkeletonGraphResidualRefiner
        use_kinematic_refiner=True,         # KinematicChainGraphRefinerTemporal
        use_adaptive_view_selection=False,  # AdaptiveViewSelector
        use_rotation_correction=False,        # RotationCorrectionHead
        use_entropy_regularization=False,   # entropy loss
        # component hyperparameters
        visibility_use_uncertainty=True,
        kc_hidden_dim=64,
        kc_num_layers=2,
        adaptive_view_k=None,
        rotation_max_deg=2.0,
        attention_entropy_weight=0.0,
        **kwargs,
    ):
        ...
```

All v2/v3 parameter names are preserved so a v4 model can load v2/v3 checkpoints with `strict=False`, exactly as v3 loads v2 today (`docs/omniview_fusion_v3_design.md`, Section 6).

## 6. Training and evaluation plan

### 6.1 Trainer

Create `experiments/train_omniview_fusion_v4_webbridge_multi.py`, mirroring `train_omniview_fusion_v2_webbridge_multi.py` and `train_omniview_fusion_v3_mpiinf3dhp.py`:

* Use `OmniMultiViewFusionV4`.
* Loss mix: 3D MPJPE + visibility BCE + uncertainty NLL + temporal velocity + bone length + epipolar consistency + optional entropy/budget losses.
* Warm-start from the best v2/v3 checkpoint with `strict=False`.
* Multi-dataset manifest loading (already in `train_omniview_fusion_v2_webbridge_multi.py`).
* Calibration curriculum: progressive `rot/focal/pp` augmentation.
* View-dropout augmentation: `view_dropout_rate > 0` (already in `augment_clip`).

### 6.2 Evaluation

Create `experiments/eval_omniview_fusion_v4_mpiinf3dhp.py`, mirroring `eval_omniview_fusion_v2_mpiinf3dhp.py`:

* Clean MPJPE / PA-MPJPE.
* Robustness matrix: rot, trans, focal, principal-point perturbations.
* Variable-view MPJPE@k curve (k=2..V) using `VariableViewInferenceWrapper` (`variable_view_inference.py`).
* Per-joint error maps and visibility accuracy.

### 6.3 Suggested ablation matrix

| Run | Vis. v2 | Skel. residual | KC refiner | Adaptive views | Rot. corr. | Entropy | Goal |
|-----|---------|----------------|------------|----------------|------------|---------|------|
| A (v4 full) | ✓ | ✓ | ✓ | ✗ |  | ✗ | Final v4 |
| B | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | Value of visibility v2 |
| C | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | Value of skeleton residual |
| D | ✓ |  | ✗ | ✗ | ✓ | ✗ | Value of KC refiner |
| E | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | Value of adaptive view selection |
| F | ✓ |  | ✓ | ✗ | ✗ | ✗ | Value of rotation correction |
| G (v3 baseline) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Reproduce v3 |

### 6.4 Success gates

A v4 candidate can replace the current 9.03 mm anchor if:

1. Clean MPI-INF-3DHP S2/Seq1 MPJPE < 9.03 mm (target < 8.6 mm).
2. rot_0.5° MPJPE < 14 mm and focal_1% MPJPE < 15 mm.
3. view_dropout_30 MPJPE < 16.3 mm (matching the swarm target).
4. Variable-view k=14 result is within 0.5 mm of full-view.
5. Repeated seeds (n ≥ 3) pass with mean MPJPE lower than the anchor.

## 7. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Too many new modules destabilise training | All modules are optional and can be warm-started from v2/v3; stage freezing used in v2/v3 trainers already (`freeze_old_params`) |
| Visibility head collapses to all-occluded | Fallback guard + BCE weight ≤ 0.1 + warm-start from anchor |
| Skeleton residual over-smooths detail | Keep dense residual path as a residual branch; graph refiner only adds correction |
| Rotation correction breaks triangulation init | `tanh` bound near identity; initialised to zero |
| Adaptive view selection is non-differentiable | Gumbel-softmax + straight-through; inference uses hard top-k |
| Longer runtime with more modules | Components are small; adaptive view selection can reduce compute at inference |

## 8. Implementation checklist

- [ ] `motionflow_mv/fusion/omniview_fusion_v4.py` with v4 class and `__main__` smoke test.
- [ ] `motionflow_mv/fusion/adaptive_view_selector.py`.
- [ ] `motionflow_mv/fusion/rotation_correction.py`.
- [ ] `experiments/train_omniview_fusion_v4_webbridge_multi.py`.
- [ ] `experiments/eval_omniview_fusion_v4_mpiinf3dhp.py`.
- [ ] `tests/test_omniview_fusion_v4.py`.
- [ ] CPU smoke tests pass for every optional module.
- [ ] Warm-start from current best 9.03 mm checkpoint succeeds with `strict=False`.
- [ ] Run one short GPU epoch to confirm loss/gradient stability before full training.

## 9. What to do next (this week)

1. **Visibility v2 + skeleton residual**: create `omniview_fusion_v4.py` containing only these two changes and run the CPU smoke test. This is the lowest-risk, highest-ROI first cut.
2. **Add rotation correction + focal correction**: build on the v2/v3 calibration-robustness curriculum and evaluate the robustness matrix.
3. **Add kinematic-chain refiner + entropy regularisation**: final refinement and interpretability losses.
4. **Adaptive view selection**: implement and ablate as the last module once the full v4 baseline is stable.
5. **Run the ablation matrix on A800** using `train_omniview_fusion_v4_webbridge_multi.py`, starting from the 9.03 mm checkpoint, and fill the v4 ablation table.
