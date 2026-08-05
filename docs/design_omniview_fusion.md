# OmniMultiView Fusion — Design Sketch for ICRA/CVPR 2027

**Goal:** Build a single, more complex multi-view fusion backbone that combines
the strongest inductive biases already explored in this repo, while staying
incrementally trainable and benchmarkable on the existing RTX 4090 queue.

## 1. Motivation

Current best baseline:
- `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
- MPI-INF-3DHP clean **9.32 mm** / PA-MPJPE **5.37 mm**

The repo already has strong but isolated modules:
- Principal-point / focal correction
- Visibility-gated fusion v2
- Spatio-temporal (T×V×J) factorized attention
- Uncertainty-weighted triangulation
- Graph-joint attention

None of these modules have been combined into a single model. The hypothesis is
that a unified architecture can push clean MPJPE below **8.5 mm** and improve
robustness under occlusion/calibration error.

## 2. Proposed architecture: `OmniMultiViewFusion`

```
Input: (B, T, V, J, 3)  -> 2D + confidence
        |
        v
PrincipalPointCorrection(K)  ->  corrected intrinsics
        |
        v
Per-frame encoder (obs + ray + camera embed)
        |
        v
Visibility head  ->  per-view, per-joint visibility multiplier m_vj in [0,1]
        |
        v
Factorised (T × V × J) transformer block
   - temporal layers  (attention over T)
   - view layers        (attention over V)
   - joint layers       (attention over J, optionally graph-constrained)
        |
        v
Weight head  ->  per-view, per-joint fusion weight w_vj
        |
        v
Uncertainty head  ->  per-view log-variance λ_vj
        |
        v
Triangulation  ->  Gauss-Newton / DLT with visibility-masked, uncertainty-weighted rays
        |
        v
Residual refinement MLP  ->  final 3D pose
```

## 3. Key design decisions

### 3.1 Visibility before spatio-temporal block

Visibility multipliers are computed early and used to:
- Mask out occluded views in the view-attention layers (set attention bias to `-∞`).
- Zero out the contribution of occluded views to the fused triangulation.
- Provide a BCE auxiliary loss that regularizes the model under view dropout.

### 3.2 Graph constraint in joint attention

Replace dense self-attention in `joint_layers` with a graph attention module that
only propagates messages along anatomical edges (parent-child, symmetry). This
replaces the separate `GraphJointRelation` experiment and enforces skeleton-aware
reasoning inside the unified model.

### 3.3 Uncertainty-weighted triangulation

The uncertainty head predicts per-view log-variance. The final triangulation is:

```
argmin_X  Σ_v  m_v * exp(-λ_v) * ||π_v(X) - x_v||^2
```

This is differentiable and reduces to standard DLT when λ is constant.

### 3.4 Warm-start strategy

To avoid destabilizing the strong 9.32 mm baseline:

1. Start from `ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth`.
2. Initialize new heads (visibility, graph, uncertainty) randomly.
3. Freeze the per-frame encoder for the first 5 epochs.
4. Unfreeze all layers and train end-to-end for another 15 epochs.

## 4. Training recipe

```bash
python experiments/train_omnimultiview_mpiinf3dhp.py \
    --train ... \
    --val ... \
    --warm_start outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth \
    --epochs 20 \
    --view_dropout_rate 0.3 --min_views 2 \
    --visibility_loss_weight 0.1 \
    --uncertainty_loss_weight 0.05 \
    --graph_num_layers 2 \
    --output outputs/omnimultiview_mpiinf3dhp.pth
```

Losses:
- 3D MPJPE (main)
- Visibility BCE (aux)
- Uncertainty NLL (aux)
- Bone-length consistency (aux)
- Velocity smoothness (aux)

## 5. Evaluation plan

1. Clean MPJPE / PA-MPJPE on MPI-INF-3DHP S6–S8.
2. Variable-view MPJPE@k (k = 2..14) using `eval_variable_views.py`.
3. Robustness matrix: rot_0.5°, trans_5mm, focal_1%, pp_5/10px, occlusion 20/40%.
4. Cross-dataset zero-shot on H36M S9/S11.
5. Runtime/latency profile on RTX 4090.

## 6. Expected impact

| Metric | Target | vs baseline (9.32 mm) |
|--------|--------|----------------------|
| Clean MPJPE | ≤ 8.5 mm | -0.8 mm |
| PA-MPJPE | ≤ 5.0 mm | -0.4 mm |
| k=4 MPJPE | ≤ 25 mm | from ~40 mm |
| Occlusion 30% | ≤ 12 mm | strong robustness story |

## 7. Risks & mitigation

| Risk | Mitigation |
|------|------------|
| Too complex for single RTX 4090 | Use factorized attention, gradient checkpointing, d=48 variant |
| Negative clean-MPJPE interaction | Warm-start + freeze encoder phase |
| Uncertainty head diverges | Clamp log-variance to [-5, 5] |
| Graph attention incompatible with variable views | Rebuild edge index for active subset |

## 8. Next step

After the current GPU queue (variable-view → visibility v2 → SSL → spatiotemporal),
queue the first `OmniMultiViewFusion` smoke run on MPI-INF-3DHP with:
- d=48
- graph_num_layers=1
- uncertainty + visibility only
- 10 epochs

This smoke will validate that the unified architecture trains without NaNs and
reaches within 5% of the baseline before committing to a full 20-30 epoch run.
