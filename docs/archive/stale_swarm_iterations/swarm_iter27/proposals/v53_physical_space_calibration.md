# v53 Physical-Space Calibration (PSC)

## Motivation

v52 uncertainty-weighted triangulation (UWT) already produces per-view/joint precision weights, but it triangulates in a purely geometric space. v53 Physical-Space Calibration (PSC) therefore sits **on top of v52 UWT** and calibrates the fused 3-D pose against physical invariants—the ground plane and the subject’s skeleton—using those weights as a robustness signal. The module is warm-startable/identity-at-init, so enabling it does not perturb a trained v52 checkpoint.

## Architecture

PSC has three sub-heads: (1) a floor calibration head that estimates the ground-plane height; (2) a bone-length calibration head that compares observed bone lengths to a learned canonical skeleton; and (3) a gated physical residual refiner that fuses these cues into a bounded 3-D correction. The v52 triangulation weights `w_{vtj}` down-weight noisy views/joints.

### Tensor shapes

- Input pose from v52: `X ∈ R^{B×T×J×3}`
- v52 triangulation weights: `w ∈ [0,1]^{B×T×V×J}`
- 2-D keypoints: `x ∈ R^{B×T×V×J×2}`
- Camera intrinsics/extrinsics: `K ∈ R^{B×T×V×3×3}, R ∈ R^{B×T×V×3×3}, t ∈ R^{B×T×V×3}`
- View mask: `M ∈ {0,1}^{B×T×V}`
- Output: refined pose `X' ∈ R^{B×T×J×3}` and auxiliary calibration loss `L_psc`

## Equations

Let `g = (0, -1, 0)` be the gravity direction (fixed world up). For each frame `t` and sample `b`:

1. **Uncertainty-weighted foot height**

   ```
   h_{t,f} = X_{t,f} · g,   f ∈ F (foot/ankle joints)
   f_t = weighted_min_f { h_{t,f} },
   ```

   where the weighted minimum is taken over foot joints using the per-joint UWT confidence `c_{t,j} = max_v w_{vtj}`.

2. **Floor residual and loss**

   ```
   r^{floor}_{t,f} = clamp(f_t - X_{t,f} · g, 0)
   L_floor = (1/|F|) Σ_{t,f} (r^{floor}_{t,f})^2
   ```

3. **Bone-length calibration**

   For each bone `b` with parent `p` and child `c`:

   ```
   l_{t,b} = || X_{t,c} - X_{t,p} ||_2
   l^{canon}_b = learned canonical length for bone b
   Δl_{t,b} = l_{t,b} - l^{canon}_b
   L_bone = (1/B_{bones}) Σ_{t,b} (Δl_{t,b})^2 · softmax(-|Δl|/τ)
   ```

   The soft weighting de-emphasizes bones with large residuals.

4. **Gated physical residual refiner**

   ```
   feat = concat[ X; g; f_t; Δl_{t,joint} ]
   r = MLP(feat)                        # (B,T,J,3)
   X' = X + σ(γ) · tanh(r)              # γ initialized so σ(γ) ≈ 0 at init
   ```

   `γ` is a learned scalar gate (or per-joint gate). Because `MLP` final layer is zero-initialized and `γ` is initialized to `-6.0`, `X' = X` at the start of training.

5. **Calibration consistency loss**

   ```
   L_reproj = (1/|V|) Σ_v w_v · || π_v(X') - x_v ||_2
   L_psc = λ_floor L_floor + λ_bone L_bone + λ_reproj L_reproj
   ```

   where `π_v` is the projection for view `v` and `w_v` are the per-view UWT weights averaged over joints.

## Inputs / Outputs

### Inputs

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d_uwt` | `(B, T, J, 3)` | v52 UWT output pose |
| `uwt_weights` | `(B, T, V, J)` | v52 triangulation weights |
| `points_2d` | `(B, T, V, J, 2)` | 2-D keypoints |
| `K`, `R`, `t` | cameras | `(B,T,V,3,3)` / `(B,T,V,3)` |
| `view_mask` | `(B, T, V)` | Valid-view mask |
| `domain_id` | `(B,)` optional | Domain label |

### Outputs

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d_psc` | `(B, T, J, 3)` | Calibrated 3-D pose |
| `psc_loss` | scalar | Auxiliary loss |
| `floor_height` | `(B, T)` | Estimated floor height |
| `bone_scale` | `(B, T, N_bones)` | Per-bone scale (logging) |

## Config flags

- `use_v53_physical_space_calibration` – master toggle
- `v53_psc_hidden` (default `64`) – residual MLP hidden dim
- `v53_psc_n_layers` (default `2`)
- `v53_psc_identity_init` (default `True`) – zero-init final layers and gate
- `v53_psc_residual_gate_init` (default `-6.0`) – gate logit
- `v53_psc_use_uwt_weights`, `v53_psc_use_floor`, `v53_psc_use_bone_scale` – head toggles
- `v53_psc_loss_weight` (default `1.0`) – multiplier on `L_psc`
- `v53_psc_floor_weight`, `v53_psc_bone_weight`, `v53_psc_reproj_weight` – per-term weights
- `v53_psc_warmup_epochs` (default `0`), `v53_psc_min_visible_views` (default `2`)

## Expected MPJPE impact

- **Identity check:** enabling PSC on a trained v52 checkpoint should change `val_MPJPE` by `< 0.1 mm`.
- **Smoke:** expect a `2–5 mm` drop versus v52 on the mixed smoke set, mainly from reduced foot-floor penetration and stabler bone lengths.
- **Full A800:** target an extra `1–3% MPJPE` gain on H36M/MPI, larger for sparse views (`MPJPE@2/3`).

## Risks and mitigations

See the companion risk report: `docs/swarm_iter27/reports/agent_physical_space_calibration_risks.md`.

## 5-step implementation plan

1. **Implement `PhysicalSpaceCalibrationV53`** in `motionflow_mv/fusion/physical_space_calibration_v53.py` with floor/bone heads and a gated residual refiner; enforce identity-at-init via zero-initialized final layers and gate initialized to `-6.0`.
2. **Wire into `OmniMultiViewFusionV5`** right after the v52 UWT block: instantiate when `use_v53_physical_space_calibration=True`, call with `pred_3d_uwt`, `uwt_weights`, cameras, 2-D points, and add `v53_psc_loss_weight * psc_loss` to `epi_loss` with warmup guard.
3. **Add config flags** in the model `__init__` and create `configs/benchmark_v53_physical_space_calibration_smoke.yaml` mirroring v46/v47/v48 smoke footprints.
4. **Run smoke validation** on the local RTX 4090: confirm identity-at-init (`val_MPJPE` delta `< 0.1 mm` vs v52), check for NaN/Inf/OOM, and compare epoch-1 `val_MPJPE`.
5. **Queue full A800 run** by adding an entry to `scripts/launch_v33_a800_queue.py` (e.g., `v53_physical_space_calibration_on_v52`) and update `AGENTS.md` once results are in.
