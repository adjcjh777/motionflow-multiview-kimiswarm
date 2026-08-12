# v54 SMPL Human Model Bridge (SHMB)

**Task identifier:** `design_v54_smpl_human_model_bridge`  
**Status:** Proposal (no code yet)  
**Depends on:** v45 (`adaptive_geometry_fusion`), v46 (`sparse_view_generalization_v46`), v47 (`temporal_aggregation_v47`), v49-Lite, v50 (`SelfEvolutionFeedbackHeadV50`), v51 (`CrossDomainSparseViewReliabilityV51`), v52 (`UncertaintyWeightedTriangulationV52`), v53 (`PhysicalSpaceCalibrationV53`)

## 1. Motivation

The MotionFlow pipeline now reaches a calibrated 3-D pose via v53 Physical-Space Calibration (PSC), but the output is still a collection of unconstrained 3-D joints. The next logical step in the paper story—multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized motionflow pipeline—is to connect the calibrated skeleton to an explicit human body model. v54 SMPL Human Model Bridge (SHMB) therefore sits **on top of v53 PSC** and regularizes the 3-D pose through a lightweight, differentiable SMPL-style body parameterization. Because the final correction is a gated residual, the module is warm-startable/identity-at-init and does not perturb a trained v53 checkpoint when first enabled.

## 2. Architecture

SHMB predicts low-dimensional SMPL body parameters from the v53-calibrated 3-D joints, maps them back to the same joint space through a differentiable body model (or a learned surrogate when SMPL is unavailable), and applies a gated residual correction. The body model acts as a soft anatomical prior rather than a hard projection.

### 2.1 Inputs / outputs

**Forward signature**

```python
pred_3d_ref, shmb_dict = smpl_bridge(
    pred_3d_psc,  # (B, T, J, 3)
    uwt_weights,  # (B, T, V, J) optional
    points_2d,    # (B, T, V, J, 2)
    K, R, t,      # cameras
    view_mask,    # (B, T, V)
    domain_id,    # (B,) optional
)
```

**Outputs**

- `pred_3d_ref`: `(B, T, J, 3)` — SMPL-regularized 3-D pose.
- `theta`: `(B, T, P)` — predicted pose parameters (`P=48` for a 16-joint rotation compact representation, or `P=72` for full SMPL pose).
- `beta`: `(B, T, S)` — predicted shape parameters (`S=10`).
- `shmb_loss`: scalar — SMPL reprojection, shape regularization, and pose prior losses.

### 2.2 Tensor shapes

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d_psc` | `(B, T, J, 3)` | v53 PSC output pose |
| `theta` | `(B, T, P)` | Predicted pose parameters (axis-angle or 6-D rotation) |
| `beta` | `(B, T, S)` | Predicted shape parameters |
| `J_smpl` | `(B, T, J, 3)` | Body-model joints recovered from `(theta, beta)` |
| `gate` | scalar | Logit initialized to `-6.0`, so `σ(gate) ≈ 0.0025` at init |
| `delta` | `(B, T, J, 3)` | Zero-initialized residual from final MLP |

### 2.3 Equations

1. **Parameter prediction.** A small MLP pools per-joint v53 positions and predicts pose and shape parameters:

   ```
   h = mean_j MLP_theta_input(pred_3d_psc)              # (B, T, hidden)
   theta = MLP_theta(h)                                  # (B, T, P)
   beta  = MLP_beta(h)                                   # (B, T, S)
   ```

2. **Differentiable body model.** When `v54_shmb_use_pretrained_smpl=True`, a pre-trained SMPL model maps `(theta, beta)` to 3-D joints:

   ```
   J_smpl = SMPL(theta, beta)                            # (B, T, J, 3)
   ```

   When SMPL is unavailable, a learned low-rank surrogate regressor with a fixed joint regressor is used. The surrogate is initialized so that `J_smpl ≈ pred_3d_psc` at the start of training.

3. **Gated residual correction.** The final output blends the v53 pose with the body-model pose:

   ```
   delta = MLP_delta(concat[pred_3d_psc, J_smpl])       # (B, T, J, 3)
   X_out = pred_3d_psc + σ(gate) * tanh(delta + J_smpl - pred_3d_psc)
   ```

   The final `MLP_delta` layer is zero-initialized and the gate logit is initialized to `-6.0`, so `X_out = pred_3d_psc` at init.

4. **Auxiliary losses.**

   ```
   L_reproj = (1/|V|) Σ_v ||π_v(J_smpl) - x_v||_2 · mask
   L_shape  = ||beta||_2^2
   L_pose   = Mahalanobis distance of theta from a zero-mean pose prior
   L_smpl   = λ_reproj L_reproj + λ_shape L_shape + λ_pose L_pose
   ```

   `L_reproj` encourages the body model to explain the original 2-D keypoints, while `L_shape` and `L_pose` prevent the network from drifting into extreme body configurations.

## 3. Config flags

```python
use_v54_smpl_human_model_bridge: bool = False
v54_shmb_hidden: int = 64
v54_shmb_n_layers: int = 2
v54_shmb_num_shape_params: int = 10
v54_shmb_num_pose_params: int = 48
v54_shmb_identity_init: bool = True
v54_shmb_residual_gate_init: float = -6.0
v54_shmb_use_pretrained_smpl: bool = True
v54_shmb_smpl_model_path: Optional[str] = "data/smpl/SMPL_NEUTRAL.pkl"
v54_shmb_use_shape_prior: bool = True
v54_shmb_use_pose_prior: bool = True
v54_shmb_loss_weight: float = 0.01
v54_shmb_reproj_weight: float = 0.1
v54_shmb_shape_reg_weight: float = 0.001
v54_shmb_pose_prior_weight: float = 0.001
v54_shmb_warmup_epochs: int = 0
```

## 4. Expected MPJPE impact

- **Identity check:** Enabling SHMB on a trained v53 checkpoint should change `val_MPJPE` by `< 0.1 mm`.
- **Smoke (RTX 4090):** Expect a `1–3 mm` drop versus v53 on the mixed smoke set, mainly from reduced joint drift and more plausible bone configurations.
- **Full A800:** Target an extra `1–2% MPJPE` gain on H36M/MPI, with larger relative improvements on sparse views (`MPJPE@2/3`) where the body prior suppresses outliers.

## 5. Risks and mitigations

See the companion risk report: `docs/swarm_iter28/reports/agent_smpl_human_model_bridge_v54_risks.md`.

## 6. 5-step implementation plan

1. **Implement `SMPLHumanModelBridgeV54`** in `motionflow_mv/fusion/smpl_human_model_bridge_v54.py`. Include pose/shape parameter MLPs, a differentiable body-model wrapper (SMPL or learned surrogate), and a gated residual refiner. Enforce identity-at-init via zero-initialized final layers and a gate initialized to `-6.0`.
2. **Wire into `OmniMultiViewFusionV5`** right after the v53 PSC block: instantiate when `use_v54_smpl_human_model_bridge=True`, call with `pred_3d_psc`, `uwt_weights`, cameras, 2-D points, and add `v54_shmb_loss_weight * shmb_loss` to `epi_loss` with a warmup guard.
3. **Add config flags** in the model `__init__` and create `configs/benchmark_v54_smpl_human_model_bridge_smoke.yaml` mirroring the v52/v53 smoke footprint.
4. **Run smoke validation** on the local RTX 4090: confirm identity-at-init (`val_MPJPE` delta `< 0.1 mm` vs v53), check for NaN/Inf/OOM, and compare epoch-1 `val_MPJPE`.
5. **Queue full A800 run** by adding an entry to `scripts/launch_v33_a800_queue.py` (e.g., `v54_smpl_human_model_bridge_on_v53`) and update `AGENTS.md` once results are in.
