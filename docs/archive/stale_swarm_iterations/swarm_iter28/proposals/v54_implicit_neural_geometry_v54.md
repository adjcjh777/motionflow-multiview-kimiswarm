# v54: Spatio-Temporal Implicit Neural Geometry Refinement

**Status:** Proposal (design-only) | **Labels:** `experiment`, `P1-next` | **Tracking issue:** #207 (depends on v52 UWT, v53 PSC, v47/v49 temporal)

## Motivation

v52 learns per-view, per-joint triangulation weights and v53 enforces floor/bone physical constraints, but both still represent the body as independent 3-D points. v54 introduces a *spatio-temporal implicit neural geometry* (ST-ING) module that treats the skeleton as a continuous 4-D field and refines the v53-calibrated pose with ray-aware, temporally-consistent residuals. It is inserted **after** v53 PSC in `OmniMultiViewFusionV5` and supports the paper pipeline: multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized MotionFlow.

## Architecture

```text
Input:  pred_3d_v53 (B,T,J,3), features (B,T,V,J,d), points_2d (B,T,V,J,2)
        K,R,t (B,T,V,3,3/3), uwt_weights (B,T,V,J), view_mask (B,T,V)
        optional: floor_height, bone_scale from v53 PSC
              │
[Ray Builder]  o_v = -R_v^T t_v ;  d_vj = R_v^T K_v^{-1} [u_vj,1]
              │
[Spatial MLP]  z_vj = MLP( concat(f_vj, d_vj, o_v, w_vj, log(w_vj+ε)) )
              │
[View Attn]    α_vj = MLP_α(z_vj), w_vj = softmax_v(α_vj/τ)
               g_tj = Σ_v w_vj·z_vj
              │
[Temporal Ctx] e_tj = mean_{s∈window(t)} MLP_temp(g_sj)
              │
[ING Head]     h_tj = MLP_field( concat(g_tj, e_tj, c_tj) )
               s_j = MLP_s(h_tj)        (signed-distance energy)
               Δp_j = MLP_Δ(h_tj)       (residual)
              │
[Refinement]   pred_3d_v54 = pred_3d_v53 + λ·Δp_j
```

`c_tj` is the joint in its local bone frame. The temporal window is symmetric at training and causal at inference. Final `MLP_α`, `MLP_s`, and `MLP_Δ` layers are zero-initialized and `λ = sigmoid(-6.0) ≈ 0`, so v54 is identity at init and a v53 checkpoint loads unchanged.

## Inputs / Outputs

Module: `motionflow_mv/fusion/implicit_neural_geometry_v54.py`

```python
class ImplicitNeuralGeometryV54(nn.Module):
    def forward(
        self,
        pred_3d: torch.Tensor,        # (B, T, J, 3)
        features: torch.Tensor,       # (B, T, V, J, d)
        points_2d: torch.Tensor,      # (B, T, V, J, 2)
        K: torch.Tensor,              # (B, T, V, 3, 3)
        R: torch.Tensor,              # (B, T, V, 3, 3)
        t: torch.Tensor,              # (B, T, V, 3)
        uwt_weights: torch.Tensor,    # (B, T, V, J)
        view_mask: Optional[torch.Tensor] = None,  # (B, T, V)
        floor_height: Optional[torch.Tensor] = None,
        bone_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # returns refined_pred_3d (B,T,J,3), v54_loss, per_joint_energy (B,T,J)
```

## Equations

```
o_v = -R_v^T t_v
x_vj = K_v^{-1} [u_vj, 1]^T
d_vj = normalize(R_v^T x_vj)

z_vj = MLP_spatial( concat(f_vj, d_vj, o_v, uwt_weights_vj, log(uwt_weights_vj+ε)) )
α_vj = MLP_α(z_vj),  w_vj = softmax_v(α_vj/τ)
g_tj = Σ_v w_vj z_vj
e_tj = mean_{s∈window(t)} MLP_temp(g_sj)
h_tj = MLP_field( concat(g_tj, e_tj, c_tj) )
pred_3d_v54 = pred_3d_v53 + sigmoid(gate) · MLP_Δ(h_tj)
```

Losses (only active when enabled):

```
L_surface  = (1/BTJ) Σ s_j^2
L_ray      = (1/Σmask) Σ mask_vj·w_vj·||(p_j-o_v)×d_vj||_2
L_reproj   = (1/Σmask) Σ mask_vj·w_vj·||π_v(p_j) - u_vj||_2
L_bone     = (1/BTJ) Σ (b_tj - b̄_j)^2 / (b̄_j+ε)
L_floor    = (1/BT) Σ max(0, floor_height - y_root)^2
L_temporal = (1/(B(T-1)J)) Σ ||Δp_tj - Δp_{t+1,j}||_2^2

L_v54 = w_surface L_surface + w_ray L_ray + w_reproj L_reproj
      + w_bone L_bone + w_floor L_floor + w_temporal L_temporal
```

## Config flags

In `OmniMultiViewFusionV5`:

```python
use_v54_implicit_neural_geometry: bool = False,
v54_ing_hidden: int = 64,
v54_ing_n_layers: int = 2,
v54_ing_temporal_window: int = 3,
v54_ing_use_spatial_field: bool = True,
v54_ing_use_temporal_field: bool = True,
v54_ing_use_ray_alignment: bool = True,
v54_ing_use_physical_constraints: bool = True,
v54_ing_identity_init: bool = True,
v54_ing_residual_gate_init: float = -6.0,
v54_ing_min_views: int = 2,
v54_ing_surface_loss_weight: float = 0.01,
v54_ing_ray_loss_weight: float = 0.1,
v54_ing_reproj_loss_weight: float = 1.0,
v54_ing_bone_loss_weight: float = 0.001,
v54_ing_floor_loss_weight: float = 0.01,
v54_ing_temporal_loss_weight: float = 0.001,
v54_ing_loss_weight: float = 0.01,
v54_ing_warmup_epochs: int = 0,
```

Integration is immediately after the v53 PSC call in `omniview_fusion_v5.py`. The v54 loss is added to `epi_loss` with the same warmup gating used for v52/v53.

## Expected MPJPE impact

| Scenario | Expected delta vs v53 |
|---|---|
| Full views (H36M/MPI) | −0.5 to −1.2 mm |
| Sparse views (v46, k=2) | −1.0 to −2.5 mm |
| 3DPW actual (V=1) | −2 to −4 mm |
| Cross-domain | 5–12 % relative gain |

Conservative smoke target: `val_MPJPE` stays finite and improves over v53 by ≥0.4 mm on the full-view validation set.

## Risks

See `docs/swarm_iter28/reports/agent_implicit_neural_geometry_v54_risks.md` for identity-at-init, OOM, early-training instability, surface-energy collapse, and physical-constraint double-counting risks with concrete mitigations.

## 5-step implementation plan

1. **Module + unit test.** Create `motionflow_mv/fusion/implicit_neural_geometry_v54.py` and `tests/test_implicit_neural_geometry_v54.py`. Assert identity-at-init, warm-start from v53, and shape correctness.
2. **Wire into `OmniMultiViewFusionV5`.** Add flags, instantiate after v53 PSC, pass tensors, and add gated loss to `epi_loss`.
3. **Trainer/CLI + smoke config.** Expose flags in `experiments/train_omniview_fusion_v5_webbridge_multi.py`; add `configs/benchmark_v54_implicit_neural_geometry_smoke.yaml` and `scripts/run_v54_implicit_neural_geometry_smoke_local_4090.sh`.
4. **RTX 4090 smoke & ablation.** Run smoke (`clip_len=3, B=4, train_samples=50`). Ablate `use_temporal_field`, `use_ray_alignment`, and `use_physical_constraints`. Freeze v52/v53 weights for epoch 1.
5. **A800 full run & docs.** Queue the best config on A800-D, evaluate `MPJPE@k`, and update `docs/results_snapshot_2026_08_09.md`.
