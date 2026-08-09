# v53: Implicit Neural Geometry for Ray-Conditioned Pose Refinement

**Status:** Proposal (design-only stage)  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #193 (depends on v52 UWT, v46 SVG, v50 SEFH, v51 CDSVR)  

## Motivation

v52 Uncertainty-Weighted Triangulation (UWT) already learns per-view, per-joint precision weights and re-triangulates with a weighted DLT, but the actual 3-D estimate is still produced by an algebraic triangulation step. That step treats each joint independently and cannot exploit the fact that the true point must lie on the intersection of camera rays. **v53 closes this gap** by introducing an *implicit neural geometry* layer that directly reasons about the calibrated rays and the multi-view features, then refines the v52 output with a continuous, ray-conditioned residual.

The module is deliberately placed **after** v52 UWT in `OmniMultiViewFusionV5`, so it receives:

* an already good initial 3-D pose from v52,
* the per-view UWT weights it can trust or distrust,
* the same per-view ST features used by v52,
* full camera calibration `K, R, t`.

It fits the paper story as the next stage of **multi-view fusion and calibration → physical-space alignment**, turning a discrete triangulation result into a learnable, calibration-aware geometry refinement.

## Design principles

1. **Build on v52.** v53 consumes the v52 refined pose and UWT weights; it does not replace the triangulator.
2. **Identity at init.** Final layers are zero-initialized and the residual gate starts at 0, so loading a v52 checkpoint into a model with v53 enabled leaves the output unchanged.
3. **Calibration-aware.** The module computes per-view camera centers and ray directions from `K, R, t, points_2d`, so the learned field is tied to real camera geometry.
4. **No new data loader.** It reuses the existing WebBridge mixed manifest and variable-view mask.

## Proposed architecture

```text
Input:  pred_3d_v52   (B, T, J, 3)      <- v52 UWT refined pose
        features      (B, T, V, J, d)  <- per-view ST features
        points_2d     (B, T, V, J, 2)  <- observed 2-D keypoints
        K, R, t       (B, T, V, 3, 3/3) <- calibrated cameras
        uwt_weights   (B, T, V, J)      <- v52 predicted triangulation weights
        view_mask     (B, T, V)          <- variable-view mask
              │
              ▼
[Ray Builder]
        ├─ camera center  o_v = -R_v^T t_v                (B, T, V, 3)
        └─ ray direction  d_vj = R_v^T K_v^{-1} [u_vj, 1]  (B, T, V, J, 3)
              │
              ▼
[Per-(view,joint) Ray Embedding]
        ├─ z_vj = MLP( concat(f_vj, d_vj, o_v,
        │                       uwt_weights_vj,
        │                       log(uwt_weights_vj + ε)) )
        │
              ▼
[Implicit Neural Geometry Head]
        ├─ α_vj  = MLP_α(z_vj)  (B, T, V, J)   ray attention logits
        ├─ δ_vj  = MLP_δ(z_vj)  (B, T, V, J, 3) per-ray 3-D offset
        └─ s_j   = MLP_s(mean_v z_vj)  (B, T, J) optional signed-distance energy
              │
              
[View Aggregation]
        w_vj = softmax_v( α_vj / τ )
        Δp_j = Σ_v w_vj · δ_vj            (B, T, J, 3)
              │
              ▼
[Refinement]
        pred_3d_v53 = pred_3d_v52 + λ · Δp_j
```

### Module API

`motionflow_mv/fusion/implicit_neural_geometry_v53.py`:

```python
class ImplicitNeuralGeometryV53(nn.Module):
    def __init__(
        self,
        d: int = 64,
        n_views: int = 4,
        hidden: int = 64,
        n_layers: int = 2,
        temperature: float = 1.0,
        use_signed_distance: bool = True,
        use_ray_alignment: bool = True,
        identity_init: bool = True,
        residual_gate_init: float = 0.0,
        min_views: int = 2,
    ):
        ...

    def forward(
        self,
        pred_3d: torch.Tensor,          # (B, T, J, 3)
        features: torch.Tensor,         # (B, T, V, J, d)
        points_2d: torch.Tensor,        # (B, T, V, J, 2)
        K: torch.Tensor,                # (B, T, V, 3, 3)
        R: torch.Tensor,                # (B, T, V, 3, 3)
        t: torch.Tensor,                # (B, T, V, 3)
        uwt_weights: torch.Tensor,      # (B, T, V, J)
        view_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (refined_pred_3d, v53_geometry_loss, ray_attention_weights)."""
        ...
```

## Equations

### Ray construction

For view `v`, joint `j`, time `t`:

```
o_v = -R_v^T t_v                                 # camera center
x_vj = K_v^{-1} [u_vj, 1]^T                      # image-plane ray in view coords
d_vj = normalize( R_v^T x_vj )                   # world-space ray direction
```

where `u_vj ∈ R^2` is the 2-D keypoint.

### Implicit neural geometry

```
z_vj = MLP( concat(f_vj, d_vj, o_v,
                   u_vj, log(w_vj + ε)) )         # (B, T, V, J, hidden)

α_vj = MLP_α(z_vj)                               # ray attention logits
δ_vj = MLP_δ(z_vj)                               # per-ray 3-D offset
s_j  = MLP_s( mean_v(z_vj) )                     # signed-distance energy

w_vj = softmax_v( α_vj / temperature )
Δp_j = Σ_v w_vj · δ_vj
pred_3d_v53 = pred_3d_v52 + λ · Δp_j
```

At initialization, `MLP_α` and `MLP_δ` final layers are zero-initialized and `λ = 0`, giving:

```
α_vj ≈ 0   →   w_vj = 1 / V_uniform
δ_vj  0   →   pred_3d_v53 = pred_3d_v52
```

So a v52 checkpoint loads unchanged and v53 can be warm-started.

### Training losses

**Ray alignment loss** penalises the distance from the refined point to each visible ray:

```
r_vj = (pred_3d_v53 - o_v) × d_vj
L_ray = (1 / Σ mask_vj) Σ_vj mask_vj · w_vj · ||r_vj||_2
```

**Surface energy loss** keeps the implicit field near zero for plausible poses:

```
L_surface = (1 / BTJ) Σ_tj s_j^2
```

**Reprojection loss** keeps the refined pose faithful to the multi-view evidence:

```
L_reproj = (1 / Σ mask_vj) Σ_vj mask_vj · w_vj · ||π_v(pred_3d_v53_j) - u_vj||_2
```

**Total auxiliary loss**:

```
L_v53 = w_ray · L_ray + w_surface · L_surface + w_reproj · L_reproj
```

## Integration into `OmniMultiViewFusionV5`

### Constructor additions

```python
use_v53_implicit_neural_geometry: bool = False,
v53_ing_hidden: int = 64,
v53_ing_n_layers: int = 2,
v53_ing_temperature: float = 1.0,
v53_ing_use_signed_distance: bool = True,
v53_ing_use_ray_alignment: bool = True,
v53_ing_identity_init: bool = True,
v53_ing_residual_gate_init: float = 0.0,
v53_ing_min_views: int = 2,
v53_ing_loss_weight: float = 0.01,
v53_ing_warmup_epochs: int = 0,
```

Instantiation after the v52 UWT block in `__init__`:

```python
if self.use_v53_implicit_neural_geometry:
    from motionflow_mv.fusion.implicit_neural_geometry_v53 import ImplicitNeuralGeometryV53
    self.implicit_neural_geometry_v53 = ImplicitNeuralGeometryV53(
        d=self.d,
        n_views=n_views,
        hidden=v53_ing_hidden,
        n_layers=v53_ing_n_layers,
        temperature=v53_ing_temperature,
        use_signed_distance=v53_ing_use_signed_distance,
        use_ray_alignment=v53_ing_use_ray_alignment,
        identity_init=v53_ing_identity_init,
        residual_gate_init=v53_ing_residual_gate_init,
        min_views=v53_ing_min_views,
    )
```

### Forward additions

Apply v53 immediately after the v52 UWT call and before the physical-space losses:

```python
v53_loss = torch.tensor(0.0, device=device, dtype=pred_3d.dtype)
if self.use_v53_implicit_neural_geometry and self.implicit_neural_geometry_v53 is not None:
    pred_3d_v53, v53_loss, ray_attn = self.implicit_neural_geometry_v53(
        pred_3d=pred_3d_gn,             # v52 output
        features=feat.view(B, T, V, J, self.d),
        points_2d=points_2d.view(B, T, V, J, 2),
        K=K_corrected.view(B, T, V, 3, 3),
        R=R.view(B, T, V, 3, 3),
        t=t.view(B, T, V, 3),
        uwt_weights=uwt_weights.view(B, T, V, J),
        view_mask=view_mask_flat.view(B, T, V),
    )
    pred_3d_gn = pred_3d_v53
```

Then add `v53_loss` to `epi_loss` when active, mirroring the v52 pattern.

## Expected MPJPE impact

| Scenario | Expected delta vs v52 | Rationale |
|---|---|---|
| Full views (H36M/MPI) | −0.5 to −1.0 mm | Ray-alignment corrects small triangulation biases that UWT weights cannot remove. |
| Sparse views (v46, k=2) | −1.0 to −2.0 mm | Implicit geometry regularises the under-constrained two-view case. |
| 3DPW actual (V=1) | −1 to −3 mm | Single-view ray loss prevents the pose from drifting away from the visible ray. |
| Cross-domain / 3DPW | 5–10 % relative gain | Continuous field transfers better across skeleton/domain variations. |

Conservative smoke target: `val_MPJPE` stays finite and improves over v52 by ≥0.4 mm on the full-view validation set.

## Risks and mitigations

See the dedicated risk report: `docs/swarm_iter27/reports/agent_implicit_neural_geometry_v53_risks.md`.

## 5-step implementation plan

1. **Scaffold the module and unit test.** Create `motionflow_mv/fusion/implicit_neural_geometry_v53.py` with the ray builder, per-view MLP, view aggregation, and zero-initialized output heads. Add `tests/test_implicit_neural_geometry_v53.py` that asserts identity-at-init, shape correctness, and warm-start from a v52 checkpoint.
2. **Wire into `OmniMultiViewFusionV5`.** Add the v53 constructor flags, instantiate the module after v52 UWT, pass the required tensors, and add `v53_loss` to `epi_loss` gated by `v53_ing_warmup_epochs`.
3. **Trainer/CLI pass-through and smoke config.** Expose v53 flags in `experiments/train_omniview_fusion_v5_webbridge_multi.py`; add `configs/benchmark_v53_implicit_neural_geometry_smoke.yaml` and `scripts/run_v53_implicit_neural_geometry_smoke_local_4090.sh`.
4. **Smoke and ablation on RTX 4090.** Run smoke at `clip_len=3, B=4, train_samples=50`. Ablate `use_signed_distance`, `use_ray_alignment`, and `temperature`. Freeze v52 weights for the first epoch to preserve the warm start.
5. **Full A800 run and paper-story update.** Queue the best smoke config on A800-D, evaluate `MPJPE@k` for `k = 2,3,4` and full views, and update `docs/results_snapshot_2026_08_09.md` with v53 numbers.

## Relation to other variants

* **v52 UWT:** v53 refines the v52 output; it does not replace the weighted DLT.
* **v46 SVG / v50 SEFH / v51 CDSVR:** v53 uses their outputs (features, reliability weights, view masks) but does not alter their internal state.
* **v28/v40 physical losses:** v53 is complementary. It provides a continuous geometry prior, while physical losses enforce bone-length and floor constraints.
* **v47/v49 temporal:** v53 operates per-frame and can be stacked before or after temporal aggregation; in this design it is placed after v52 so the implicit field sees temporally-consistent poses.
