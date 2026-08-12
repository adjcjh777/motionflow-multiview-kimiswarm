# v55 Skeleton-Graph Uncertainty Gating (SGUG)

**Tracking issue:** #208 (placeholder)  
**Base branch:** `v55-sgug`  
**Depends on:** v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT, v53-PSC, v54-PSC-v2

## 1. Module name and one-line purpose

**Module:** `SkeletonGraphUncertaintyGatingV55` → `motionflow_mv/fusion/skeleton_graph_uncertainty_gating_v55.py`

**One-line purpose:** Refine the v54 physically-calibrated 3D pose by propagating corrections through a kinematic skeleton graph, where each edge and node is gated by per-joint uncertainty so reliable joints anchor the propagation and uncertain joints receive anatomy-guided corrections while preserving the v54 baseline at initialization.

## 2. Where it sits in the OmniMultiViewFusionV5 forward pass

Placed **after** `PhysicalSpaceCalibrationV2V54` and **before** the final residual MLP / v47/v49 temporal / v50 SEFH heads.

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss
    ↓
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, floor_height, bone_scale, psc_loss
    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_loss
    ↓
v55 SkeletonGraphUncertaintyGatingV55
    (consumes pred_3d_psc2, uwt_weights, points_2d, K, R, t, view_mask, domain_id)
    → pred_3d_sgug, sgug_loss
    
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

## 3. Inputs, outputs, and shapes

### Inputs

| Tensor / value | Shape | Description |
|---|---|---|
| `pred_3d_in` | `(B, T, J, 3)` | Output of v54 PSC-v2, the physically calibrated 3D pose. |
| `uwt_weights` | `(B, T, V, J)` or `(B, T, V, J, 1)` | v52 per-view-per-joint triangulation weights used as a robustness proxy. |
| `points_2d` | `(B, T, V, J, 2)` | Original 2D keypoints. |
| `confidences` | `(B, T, V, J)` | Detection confidences. |
| `K`, `R`, `t` | `(B, V, 3, 3)`, `(B, V, 3, 3)`, `(B, V, 3)` | Intrinsic / extrinsic camera parameters. |
| `view_mask` | `(B, T, V)` or `(B, T, V, 1)` | Boolean/0-1 mask of valid views. |
| `domain_id` | `(B,)` or `(B, 1)` | Domain indices for per-domain uncertainty statistics (optional, default 0). |
| `skeleton_parent` | `(J,)` | Static parent-index lookup defining the kinematic tree. |
| `skeleton_left/right` | optional | Left/right symmetry indices for symmetry-aware edge gating. |

### Outputs

| Tensor / value | Shape | Description |
|---|---|---|
| `pred_3d_sgug` | `(B, T, J, 3)` | Uncertainty-gated skeleton-graph refined pose. |
| `sgug_loss` | scalar | Auxiliary loss encouraging graph-consistent, low-uncertainty corrections. |
| `per_joint_gate` | `(B, T, J)` | Diagnostic: per-joint correction magnitude (clamped, identity ≈ 0). |

## 4. Architecture

### High-level design

SGUG builds a small skeleton graph on the `J` body joints. Each node is initialized from the v54 pose and augmented with an uncertainty-aware feature. Messages are passed only along kinematic edges, and the strength of each message is modulated by an **edge uncertainty gate** derived from both the source and target joint uncertainties. A **residual gate** with a strong negative logit initially blocks the correction, preserving the v54 checkpoint exactly.

### Per-joint uncertainty feature

For each joint `j`, an uncertainty scalar `u_j` is computed from the v52 UWT weights:

```
u_j = -log( max_v(w_jv) + eps )      # high max-weight → low uncertainty
u_j = u_j * (1 + σ(reproj_error_j))   # re-weight by reprojection residual
```

Optionally, a small per-joint MLP maps `(u_j, confidence_j, bone_residual_j)` to a node embedding `h_j ∈ R^d` where `d = v55_sgug_hidden`.

### Skeleton-graph message passing

For each kinematic edge `(parent(j), j)` and symmetry edge `(j, j_sym)`:

```
m_{j←p} = φ_e(h_p, h_j, u_p, u_j) · g_e(u_p, u_j)
```

where:

- `φ_e` is a small MLP / edge network.
- `g_e(u_p, u_j) = sigmoid( MLP([u_p, u_j]) )` is the **edge uncertainty gate**. At init, `g_e` is biased toward 1 (messages pass freely) by initializing the MLP output bias to `+2.0` and using a soft identity-at-init design; the overall correction is still killed by the output gate below.
- The gate can optionally down-weight messages when **either** endpoint is uncertain, so reliable joints become anchors.

Node update (one layer, `v55_sgug_gnn_layers=1`):

```
h_j^{(l+1)} = h_j^{(l)} + (1 / |N(j)|) Σ_{p∈N(j)} m_{j←p}^{(l)}
```

### Correction head and identity-at-init mechanism

A final MLP maps each updated node feature to a 3D correction `Δ_j ∈ R^3`. The correction is added to the v54 pose through a scalar residual gate:

```
pred_3d_sgug = pred_3d_psc2 + σ(g_gate) · Δ
```

**Identity-at-init is enforced by:**

1. Zero-initializing the final output layer of the correction MLP so that `Δ_j = 0` at initialization.
2. Initializing the residual gate logit `g_gate = v55_sgug_residual_gate_init` (default `−6.0`), so `σ(g_gate) ≈ 0.0025` and the pose is unchanged.
3. Keeping edge-MLP message passers with small random weights but final output zero so that even if messages pass, the aggregate is projected to zero.
4. Initializing per-domain uncertainty statistics (mean, log-variance) to neutral values.

### Auxiliary losses

`sgug_loss` combines:

| Loss | Symbol | Description | Weight |
|---|---|---|---|
| Graph consistency loss | `L_graph` | Encourage adjacent joints to have similar correction magnitudes when both are low-uncertainty; uses a smooth L1 on `||Δ_j - Δ_p||` weighted by `1 - |u_j - u_p|`. | `v55_sgug_graph_weight` (default `0.05`) |
| Uncertainty-weighted reprojection loss | `L_reproj` | Standard reprojection error weighted by the updated per-joint uncertainty. | `v55_sgug_reproj_weight` (default `0.1`) |
| Bone-length preservation loss | `L_bone` | Penalize bone-length changes after correction, masked by visibility. | `v55_sgug_bone_weight` (default `0.05`) |
| Smoothness loss | `L_smooth` | Temporal smoothness on `Δ_j` along `T` weighted by joint uncertainty. | `v55_sgug_smooth_weight` (default `0.01`) |

The total auxiliary loss is:

```
sgug_loss = v55_sgug_loss_weight * (L_reproj + L_graph + L_bone + L_smooth)
```

A warmup guard can zero out `sgug_loss` for the first `v55_sgug_warmup_epochs` epochs.

## 5. Expected MPJPE impact and main risks

### Expected impact

| View setting | Expected change |
|---|---|
| Full views | Identity `< 0.1 mm`; smoke `−0.3 to −1.0 mm`; full `−0.5 to −1.5 mm` |
| Sparse `@3` | `−0.5 to −1.5 mm` |
| Sparse `@2` | `−1.0 to −3.0 mm` |
| 3DPW actual | `−1.0 to −2.5 mm` (uncertainty gating should especially help cross-domain joints) |

Rationale: v54 already calibrates the pose locally. The remaining error is often concentrated in joints with sparse or noisy views (wrists, ankles). By propagating corrections from reliable to uncertain joints along anatomically valid skeleton edges, SGUG can clean up these end-effector jitters without disturbing well-estimated joints.

### Main risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| GNN over-smooths fine motion detail | Wrist/ankle detail collapses; MPJPE rises. | Shallow graph (`gnn_layers=1`), small hidden dim (`64`), and clamp correction magnitude in the first epoch. |
| Uncertainty gating suppresses all updates | No improvement over v54. | Edge gate initialized near 1; only the output residual gate is closed at init. Loss weights are warmed up. |
| Edge gates become hard binary switches | Gradients vanish; sparse-view joints not helped. | Use soft sigmoid gates; add a small floor `v55_sgug_min_edge_gate` (default `0.1`) to keep gradient flow. |
| Conflicts with v54 physical losses | Bone/floor losses double-count. | Initialize bone/floor weights lower than v54 defaults and mask losses by visibility; ablate `v55_sgug_use_bone_loss`. |
| Identity-at-init regression | v54 checkpoint changes by `>0.1 mm` with SGUG enabled. | Zero-initialized correction head, `g_gate = −6.0`, and a unit test asserting `||pred_sgug - pred_psc2||_∞ < 1e-4`. |

## 6. Smoke acceptance criteria

On the local RTX 4090:

1. **Identity-at-init:** loading the best v54 checkpoint with `use_v55_skeleton_graph_uncertainty_gating=True` and taking **no training step** changes `val_MPJPE@full` by `< 0.1 mm`.
2. **Baseline proximity:** after one full smoke epoch, `val_MPJPE@full` is within `1 mm` of the v54 baseline.
3. **Sparse-view non-regression:** `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.
4. **Numerical stability:** no NaN, Inf, or OOM through at least one full epoch.
5. **Gate sanity:** edge gate values stay in `[0.1, 0.9]` for ≥ 95% of joints; output residual gate `σ(g_gate)` stays `< 0.1` for the first 500 steps.
6. **Bone-length sanity:** ratio of corrected bone lengths to v54 bone lengths stays in `[0.8, 1.2]` for ≥ 95% of bones.

## 7. Required new files and files to modify

### New files

| Path | Purpose |
|---|---|
| `motionflow_mv/fusion/skeleton_graph_uncertainty_gating_v55.py` | `SkeletonGraphUncertaintyGatingV55` module implementation. |
| `tests/test_skeleton_graph_uncertainty_gating_v55.py` | Unit tests for identity-at-init, graph construction, edge gating, and gradient flow. |
| `configs/benchmark_v55_sgug_smoke.yaml` | Smoke config copied from `configs/benchmark_v54_psc_v2_smoke.yaml` with v55 flags enabled. |
| `scripts/run_v55_sgug_smoke_local_4090.sh` | Smoke launch script that warm-starts from the best available v54 checkpoint. |

### Files to modify

| Path | Change |
|---|---|
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Add v55 constructor flag, instantiate `SkeletonGraphUncertaintyGatingV55` when enabled, insert call after v54 PSC-v2 block, and add `sgug_loss` to the existing loss dictionary with key `v55_sgug`. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Forward `domain_id` and aggregate `loss_dict["v55_sgug"]` with `v55_sgug_loss_weight`, honoring `v55_sgug_warmup_epochs`. |
| `scripts/launch_v33_a800_queue.py` | Add A800 full-run entry `v55_skeleton_graph_uncertainty_gating_on_v54` after v54 PSC-v2 results are available. |

### Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_skeleton_graph_uncertainty_gating` | bool | `False` | Master toggle |
| `v55_sgug_hidden` | int | `64` | Node/edge embedding dimension |
| `v55_sgug_n_layers` | int | `2` | Correction MLP depth |
| `v55_sgug_gnn_layers` | int | `1` | Number of skeleton-graph message-passing layers |
| `v55_sgug_identity_init` | bool | `True` | Zero-initialize final correction layer and gate |
| `v55_sgug_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init |
| `v55_sgug_min_edge_gate` | float | `0.1` | Floor on edge uncertainty gate for gradient flow |
| `v55_sgug_use_edge_gating` | bool | `True` | Modulate messages by joint uncertainty |
| `v55_sgug_use_node_gating` | bool | `True` | Gate per-node correction by local uncertainty |
| `v55_sgug_use_bone_loss` | bool | `True` | Enable bone-length preservation loss |
| `v55_sgug_loss_weight` | float | `1.0` | Multiplier on total `L_sgug` |
| `v55_sgug_reproj_weight` | float | `0.1` | Weight of `L_reproj` |
| `v55_sgug_graph_weight` | float | `0.05` | Weight of `L_graph` |
| `v55_sgug_bone_weight` | float | `0.05` | Weight of `L_bone` |
| `v55_sgug_smooth_weight` | float | `0.01` | Weight of `L_smooth` |
| `v55_sgug_warmup_epochs` | int | `0` | Epochs before `sgug_loss` contributes to total loss |
