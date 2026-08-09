# v54 Physical-Space Calibration v2 (PSC-v2)

**Module name:** `physical_space_calibration_v2_v54`  
**Base:** `OmniMultiViewFusionV5` with `use_v52_uncertainty_weighted_triangulation` and `use_v53_physical_space_calibration` enabled.  
**Tracking issue:** TBD (swarm-iter28 / v54)

---

## 1. Motivation

v53 Physical-Space Calibration (PSC) already enforces floor and bone-length constraints on the v52 triangulated pose, but its correction is mostly global: a single residual MLP and scalar gate adjust all joints uniformly. In practice, different body parts violate physical priors in different ways—feet penetrate the floor, forearms over-stretch, wrists jitter between frames—and a single global residual cannot fix these local errors without over-smoothing the whole pose.

v54 Physical-Space Calibration v2 therefore sits **on top of v53** and applies a **skeleton-graph, joint-level physical refiner**. It uses the v52 uncertainty weights as a per-view robustness signal, builds per-joint physical hints (floor distance, bone-length residual, reprojection error, temporal velocity), and propagates corrections along the skeleton graph. Like v53, the module is warm-startable/identity-at-init, so it can be enabled on a trained v53 checkpoint without perturbing the baseline.

This aligns with the paper narrative: **multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized MotionFlow pipeline**. PSC-v2 tightens the physical-space alignment stage before the final output.

## 2. Architecture

PSC-v2 is placed **after** `PhysicalSpaceCalibrationV53` and **before** the final residual MLP in `OmniMultiViewFusionV5`.

```
v52 UWT output ──► v53 PSC ──► v54 PSC-v2 ──► final residual MLP
```

Internally it has three heads:

1. **Floor & contact head** — estimates a per-batch/time floor height from UWT-weighted foot joints and applies a soft contact correction.
2. **Bone-length calibration head** — learns per-domain canonical bone log-scales and computes per-bone length residuals.
3. **Skeleton-graph physical refiner** — a lightweight GNN over the kinematic chain that predicts a per-joint 3-D residual from concatenated physical hints.

### Tensor shapes

- Input pose: `X ∈ R^{B×T×J×3}`
- v52 UWT weights: `w ∈ [0,1]^{B×T×V×J}`
- 2-D keypoints: `x ∈ R^{B×T×V×J×2}`
- Cameras: `K ∈ R^{B×T×V×3×3}`, `R ∈ R^{B×T×V×3×3}`, `t ∈ R^{B×T×V×3}`
- View mask: `M ∈ {0,1}^{B×T×V}`
- Output: refined pose `X' ∈ R^{B×T×J×3}` and auxiliary loss `L_psc2`

## 3. Equations

Let `g = (0, -1, 0)` be the world up direction. Denote parent/child bone pairs by `b = (parent, child)` and the kinematic parent map by `π(j)`.

### Floor height & loss

```
c_{t,j} = max_v w_{t,v,j}                         # per-joint UWT confidence
h_floor^{t} = weighted_min_f { X_{t,f} · g }      # over foot/ankle joints f
L_floor = (1/|F|) Σ_t Σ_f [ clamp( h_floor^{t} - X_{t,f}·g, 0) ]^2
```

At initialization the floor estimate equals the current lowest foot joint, so `L_floor = 0`.

### Bone-length calibration

```
l_{t,b} = || X_{t,child(b)} - X_{t,parent(b)} ||_2
s_b ~ learned per-domain log-scale,   init s_b = 0
l*_{t,b} = l_{t,b} · exp(s_b)
L_bone = (1/B) Σ_{t,b} (l_{t,b} - l*_{t,b})^2
```

Because `s_b` is initialized to zero, `l* = l` and the bone loss is zero at start.

### Skeleton-graph residual refiner

For each joint `j` build a physical-hint feature:

```
d_j^floor = X_j · g - h_floor                       # floor distance
v_j       = X_j^{(t)} - X_j^{(t-1)}                 # temporal velocity
r_j       = Σ_v w_{v,j} · || π_v(X) - x_v ||_2      # weighted reprojection residual
h_j       = concat[ X_j, d_j^floor, v_j, r_j ]
```

Run one graph-convolution step over the skeleton edges:

```
m_j = Σ_{k ∈ N(j)} α_{jk} · MLP_edge( concat[h_j, h_k, e_{jk}] )
H_j = h_j + m_j
```

where `α_{jk}` are learned edge attention weights and `e_{jk}` encodes bone direction/length. The final layer is zero-initialized:

```
ΔX_j = MLP_out(H_j) ∈ R^3,    init ΔX_j = 0
X'_j = X_j + σ(γ) · ΔX_j
```

The gate logit `γ` is initialized to `v54_psc2_residual_gate_init = -6.0`, so `σ(γ) ≈ 0.0025` and `X' ≈ X` at start.

### Contact & temporal losses

```
L_contact = Σ_{t,f} 1_{||v_f|| < τ_v} · [ clamp( h_floor^{t} - X_{t,f}·g, 0) ]^2
L_temporal = Σ_{t>1} || ΔX^{(t)} - ΔX^{(t-1)} ||_2^2
```

`L_temporal` penalizes high-frequency jitter of the correction; it is zero at identity init.

### Total auxiliary loss

```
L_psc2 = λ_floor L_floor + λ_bone L_bone + λ_contact L_contact
       + λ_temporal L_temporal + λ_reproj L_reproj
```

## 4. Inputs / Outputs

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d_psc` | `(B, T, J, 3)` | Input pose from v53 |
| `uwt_weights` | `(B, T, V, J)` | v52 triangulation weights |
| `points_2d` | `(B, T, V, J, 2)` | 2-D keypoints |
| `K`, `R`, `t` | see above | Camera parameters |
| `view_mask` | `(B, T, V)` | Valid-view mask |
| `domain_id` | `(B,)` | Optional domain labels |
| **Output** `pred_3d_psc2` | `(B, T, J, 3)` | Calibrated pose |
| **Output** `psc2_loss` | scalar | Auxiliary loss |
| **Output** `floor_height` | `(B, T)` | Estimated floor height |
| **Output** `bone_scale` | `(B, T, n_bones)` | Per-bone scale logging |

## 5. Config Flags

```yaml
use_v54_physical_space_calibration_v2: false
v54_psc2_hidden: 64
v54_psc2_n_layers: 2
v54_psc2_num_domains: 8
v54_psc2_use_floor: true
v54_psc2_use_bone_scale: true
v54_psc2_use_contact: true
v54_psc2_use_temporal_smoothness: true
v54_psc2_use_gnn: true
v54_psc2_gnn_layers: 1
v54_psc2_identity_init: true
v54_psc2_residual_gate_init: -6.0
v54_psc2_floor_weight: 0.01
v54_psc2_bone_weight: 0.05
v54_psc2_contact_weight: 0.01
v54_psc2_temporal_weight: 0.01
v54_psc2_reproj_weight: 0.1
v54_psc2_contact_velocity_thresh: 0.3   # m/s
v54_psc2_warmup_epochs: 0
```

## 6. Expected MPJPE Impact

- **Identity check:** enabling PSC-v2 on a trained v53 checkpoint should change `val_MPJPE` by `< 0.1 mm`.
- **Smoke (2 epochs / 500 samples):** neutral to `-1.5 mm`; the GNN residual needs a few hundred steps to overcome identity init.
- **Full A800:** target an extra `-0.8 mm` to `-2.0 mm` on H36M/MPI, with larger relative gains on sparse views (`MPJPE@2/3`) where local bone/contact corrections are most valuable.
- **3DPW / cross-domain:** expect reduced foot-floor penetration and more stable root height, giving an additional `-1 mm` to `-3 mm`.

## 7. Risks

See the companion risk report: `docs/swarm_iter28/reports/agent_physical_space_calibration_v2_v54_risks.md`.

## 8. 5-step Implementation Plan

1. **Implement `PhysicalSpaceCalibrationV2V54`** in `motionflow_mv/fusion/physical_space_calibration_v2_v54.py` with the floor/contact head, bone-length calibration head, and skeleton-graph residual refiner; enforce identity-at-init via zero-initialized final MLP/GNN layers and a gate initialized to `-6.0`.
2. **Wire into `OmniMultiViewFusionV5`** after the v53 PSC block: instantiate when `use_v54_physical_space_calibration_v2=True`, call with the v53 output and v52 UWT weights, and store the auxiliary loss `_v54_psc2_loss`.
3. **Add loss and warm-up gating** in `forward` / `get_loss`: add `v54_psc2_loss_weight * psc2_loss` to `epi_loss`, respecting `v54_psc2_warmup_epochs`.
4. **Create smoke assets:** add `configs/benchmark_v54_psc2_smoke.yaml` and `scripts/run_v54_psc2_smoke_local_4090.sh`; run on RTX 4090 and verify identity-at-init (`ΔMPJPE < 0.1 mm`) plus stable training for one epoch.
5. **Queue and ablate:** add an entry `v54_physical_space_calibration_v2_on_v53` to `scripts/launch_v33_a800_queue.py`; compare `v53` vs `v53+v54` on full data and report `MPJPE@full`, `MPJPE@2/3/4`, and physical metrics (foot-floor distance, bone-length variance).
