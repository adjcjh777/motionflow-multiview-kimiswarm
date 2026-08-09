# v55: Implicit Neural Geometry Refinement

**Status:** Proposal (design-only) | **Labels:** `experiment`, `P1-next` | **Tracking issue:** #207 (depends on v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT, v53-PSC, v54-PSC-v2)

## 1. Module name and purpose

**Module:** `ImplicitNeuralGeometryV55` → `motionflow_mv/fusion/implicit_neural_geometry_v55.py`

**One-line purpose:** Refine the v54 physically-calibrated 3-D pose with a lightweight, per-joint implicit neural field that learns smooth, bone-structured corrections without changing the baseline at initialization.

## 2. Where it sits in the OmniMultiViewFusionV5 forward pass

Placed **after** `PhysicalSpaceCalibrationV2V54` (v54 PSC-v2) and **before** the final residual MLP / v47/v49 temporal / v50 SEFH heads.

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss
    
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, floor_height, bone_scale, psc_loss
    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_loss
    ↓
v55 ImplicitNeuralGeometryV55
    (consumes pred_3d_psc2, uwt_weights, points_2d, K, R, t, view_mask, domain_id,
             floor_height_v2, bone_scale_v2)
    → pred_3d_ing, ing_loss
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

## 3. Inputs, outputs, and shapes

**Inputs:**

| Tensor | Shape | Description |
|---|---|---|
| `pred_3d` | `(B, T, J, 3)` | v54-calibrated 3-D pose |
| `uwt_weights` | `(B, T, V, J)` | v52 uncertainty-weighted triangulation weights |
| `points_2d` | `(B, T, V, J, 2)` | Input 2-D keypoints |
| `K` | `(B, T, V, 3, 3)` | Intrinsics |
| `R` | `(B, T, V, 3, 3)` | Camera rotation |
| `t` | `(B, T, V, 3)` | Camera translation |
| `view_mask` | `(B, T, V)` | Valid-view mask |
| `domain_id` | `(B,)` or int | Domain index for per-domain conditioning |
| `floor_height_v2` | `(B, T)` or scalar | v54 estimated floor height (optional) |
| `bone_scale_v2` | `(B, T, num_bones)` | v54 per-bone scale (optional) |

**Outputs:**

| Tensor | Shape | Description |
|---|---|---|
| `pred_3d_ing` | `(B, T, J, 3)` | ING-refined 3-D pose |
| `ing_loss` | scalar | Auxiliary loss term |
| `per_joint_energy` | `(B, T, J)` | Implicit surface energy per joint (for logging) |

## 4. Architecture

### 4.1 Per-joint feature encoding

For each `(time, joint)` token, build a per-view ray embedding and aggregate across views using the v52 UWT weights as a soft attention mechanism.

```text
o_v = -R_v^T t_v                                    # camera center
x_vj = K_v^{-1} [u_vj, 1]^T                         # image ray in camera space
d_vj = normalize(R_v^T x_vj)                         # world-space ray direction

z_vj = MLP_spatial( concat(
    f_vj,                                            # view/joint feature from v52/v54
    d_vj,                                            # ray direction
    o_v,                                             # camera center
    w_vj,                                            # uwt weight
    log(w_vj + ε)                                    # log-uncertainty
) )                                                  # (B,T,V,J,d_hidden)

α_vj = MLP_α(z_vj)                                   # view attention logit
w̃_vj = softmax_v( α_vj / τ )                         # view attention (τ=1 default)
g_tj = Σ_v w_vj · z_vj                              # (B,T,J,d_hidden)
```

### 4.2 Implicit neural field head

The ING head interprets the per-joint feature as a query into a learned implicit field. It predicts:

1. `s_j` — signed-distance-like energy per joint (encourages the pose to lie near a learned human-shape manifold).
2. `Δp_j` — per-joint 3-D residual correction.

```text
h_tj = MLP_field( concat(g_tj, c_tj, floor_hint_tj, bone_hint_tj) )
s_j  = MLP_s(h_tj)                                   # (B,T,J)
Δp_j = MLP_Δ(h_tj)                                   # (B,T,J,3)

gate = sigmoid( v55_ing_residual_gate_init )           # ≈ 0.0025 at init
pred_3d_ing = pred_3d + gate · Δp_j                    # identity at init
```

`c_tj` is the joint position relative to its parent in the kinematic chain (local bone-frame coordinate). `floor_hint_tj` and `bone_hint_tj` are optional scalar/vector hints from v54 PSC-v2; when v54 outputs are unavailable, they are set to zero so the module remains self-contained.

### 4.3 Layers and dimensions

```text
MLP_spatial: Linear(d_input, d_hidden)
               → LayerNorm
               → ReLU
               → Linear(d_hidden, d_hidden)
               → (repeat n_layers times)

MLP_α:        Linear(d_hidden, 1)                    # view attention score

MLP_field:    Linear(d_hidden + d_hints, d_hidden)
               → LayerNorm
               → ReLU
               → (repeat n_layers times)

MLP_s:        Linear(d_hidden, 1)                    # zero-initialized final layer
MLP_Δ:        Linear(d_hidden, 3)                     # zero-initialized final layer
```

Default `d_hidden = 64`, `n_layers = 2`.

### 4.4 Identity-at-init mechanism

- The **final layers** of `MLP_s` and `MLP_Δ` are zero-initialized.
- The **residual gate** is initialized with logit `−6.0`, so `gate ≈ 0.0025`.
- The **view attention logits** `MLP_α` are zero-initialized, producing uniform attention at init (views are weighted by UWT only).
- `floor_hint` and `bone_hint` are zero when v54 outputs are unavailable, so no new bias is introduced.

Result: loading a v54 checkpoint with v55 enabled leaves `pred_3d_ing` within `1e-4` of `pred_3d`.

### 4.5 Losses

```text
L_surface  = (1/BTJ) Σ s_j^2
L_reproj   = (1/Σmask) Σ mask_vj · w_vj · ||π_v(p_j) - u_vj||_2
L_bone     = (1/BT·Bones) Σ_b (||p_child - p_parent||_2 / exp(s_b) - l_b^canonical)^2
L_floor    = (1/BT) Σ max(0, floor_height - min_{foot joints} y_j)^2
L_temporal = (1/B(T-1)J) Σ ||Δp_tj - Δp_{t+1,j}||_2^2

L_ing = w_surface·L_surface + w_reproj·L_reproj + w_bone·L_bone
      + w_floor·L_floor + w_temporal·L_temporal
```

Loss weights are clamped during warmup so the module does not disturb the v54 baseline before it has learned anything meaningful.

## 5. Expected MPJPE impact and main risks

| Scenario | Expected delta vs v54 |
|---|---|
| Full views (H36M/MPI) | −0.4 to −1.0 mm |
| Sparse views (v46, k=2/3) | −1.0 to −2.5 mm |
| 3DPW actual / cross-domain | −1.5 to −3.5 mm |

**Main risks:**

| Risk | Symptom | Mitigation |
|---|---|---|
| Surface energy collapses or over-smooths pose | Wrists/ankles drift toward mean pose; MPJPE rises | Clamp `s_j` magnitude; keep field shallow (`n_layers=2`, `d=64`); zero-init final layers |
| Double-counting v54 physical losses | Floor/bone losses pull pose in conflicting directions; training stalls | Use v54 outputs only as *hints*, not hard constraints; loss weights 5× smaller than v54 defaults |
| Ray attention overfits to training cameras | Validation MPJPE rises while train MPJPE falls | Tie `MLP_α` to UWT weights via residual only: `α_vj = α_base + log(w_vj)` |
| Temporal smoothness over-smoothes fast motion | High-velocity frames show lag | Velocity-gate `L_temporal`; use causal window only at inference |
| Identity-at-init fails | v54 checkpoint changes by `>0.1 mm` when v55 enabled | Unit test asserting `||pred_3d_ing - pred_3d||_∞ < 1e-4`; gate logit `−6.0` |

## 6. Smoke acceptance criteria

- **Identity-at-init:** loading a v54 checkpoint with v55 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
- **No NaN/Inf/OOM** through at least one full epoch on RTX 4090.
- **Full-view stability:** `val_MPJPE@full` is within `1 mm` of the v54-PSC-v2 baseline on the same smoke config.
- **Surface sanity:** per-joint energy `s_j` is finite and 90% of values lie in `[-1, 1]` after the first epoch.
- **Bone sanity:** bone-length ratios computed from `pred_3d_ing` stay within `[0.5, 2.0]` of v54 for ≥95% of bones.
- **Sparse-view improvement:** `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.

## 7. Required new files and files to modify

**New files:**

- `motionflow_mv/fusion/implicit_neural_geometry_v55.py` — main `ImplicitNeuralGeometryV55` module.
- `tests/test_implicit_neural_geometry_v55.py` — unit tests for identity-at-init, shape correctness, surface energy sanity, and gradient flow.
- `configs/benchmark_v55_implicit_neural_geometry_smoke.yaml` — smoke config copied from v54 with v55 flags enabled.
- `scripts/run_v55_implicit_neural_geometry_smoke_local_4090.sh` — smoke launch script that warm-starts from the best v54 checkpoint.

**Files to modify:**

- `motionflow_mv/fusion/omniview_fusion_v5.py` — add v55 constructor flag, instantiate module when enabled, insert call after v54 PSC-v2, add `ing_loss` to `epi_loss`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — forward `domain_id` and aggregate `ing_loss` with `v55_ing_loss_weight` and warmup guard.
- `scripts/launch_v33_a800_queue.py` — add A800 full-run entry `v55_implicit_neural_geometry_on_v54`.

### Config flags

```python
use_v55_implicit_neural_geometry: bool = False
v55_ing_hidden: int = 64
v55_ing_n_layers: int = 2
v55_ing_num_domains: int = 8
v55_ing_identity_init: bool = True
v55_ing_residual_gate_init: float = -6.0
v55_ing_use_view_attention: bool = True
v55_ing_use_temporal_field: bool = True
v55_ing_use_physical_hints: bool = True
v55_ing_loss_weight: float = 1.0
v55_ing_surface_weight: float = 0.01
v55_ing_reproj_weight: float = 0.1
v55_ing_bone_weight: float = 0.05
v55_ing_floor_weight: float = 0.01
v55_ing_temporal_weight: float = 0.01
v55_ing_min_visible_views: int = 2
v55_ing_warmup_epochs: int = 0
v55_ing_temperature: float = 1.0
```

## Notes

- Keep v55 optional: `OmniMultiViewFusionV5` must load and run when `use_v55_implicit_neural_geometry=False`.
- Do not remove or replace v54; v55 refines it.
- If smoke shows conflict with v54 losses, disable the overlapping v55 loss terms (`surface_weight`, `bone_weight`, `floor_weight`) and keep only the ING residual correction with `reproj_weight` and `temporal_weight`.
