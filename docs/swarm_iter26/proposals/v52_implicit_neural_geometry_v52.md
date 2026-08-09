# v52: Implicit Neural Geometry Refinement for Calibration-Aware Pose Optimization

**Status:** Proposal (design-only stage)  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #190 (depends on v46–v51, especially v50 SEFH and v51 CDSVR)  

## Motivation

The current MotionFlow-MultiView pipeline (v25 → v51) fuses multi-view evidence with increasingly sophisticated heads—adaptive geometry, sparse-view reliability, temporal aggregation, domain adaptation, and self-evolution feedback. Yet no module treats the 3-D human body as a *continuous geometric field*. v52 closes that gap.

**Why implicit neural geometry matters:**
- Triangulation and residual MLPs operate on discrete joint coordinates. They ignore the fact that joints lie on a continuous articulated surface and that bone lengths, surface continuity, and camera geometry couple all joints together.
- A learned implicit field can encode a *pose manifold prior*: poses that are physically plausible should sit on or near the zero level-set of the field; implausible poses should have high energy.
- By conditioning the field on the same multi-view ST features used by v50/v51, v52 turns a black-box residual into a geometry-aware, calibration-aware refinement that is warm-startable and fully compatible with `OmniMultiViewFusionV5`.

**Paper-story fit:** v52 materialises the *physical-space alignment* stage of the pipeline: multi-view video → pose extraction → fusion/calibration → **implicit neural geometry refinement** → optimized motionflow.

## Design principles

1. **Build on v50/v51.** v52 consumes the outputs of SEFH (per-view reliability `R`, per-joint log-variance `Λ`) and CDSVR (cross-domain reliability offsets). It does not replace them.
2. **Identity at init.** The implicit field is initialized so that its residual correction is zero and its geometry energy is constant; a v51 checkpoint loads unchanged.
3. **Calibration-aware.** The refinement explicitly uses camera intrinsics/extrinsics and per-view weights so that geometry and calibration are jointly reasoned about.
4. **No new data loader.** v52 reuses the existing WebBridge mixed manifest and variable-view mask. It only needs `K`, `R`, `t`, `points_2d`, `pred_3d`, and ST features already computed in `OmniMultiViewFusionV5`.

## Proposed architecture

```text
Input:  pred_3d          (B, T, J, 3)      <- triangulated + GN pose
        feat_pooled     (B, T, J, d)      <- pooled ST features (mean over views)
        K, R, t         (B, T, V, 3, 3/3) <- calibrated cameras
        points_2d       (B, T, V, J, 2)   <- observed 2-D keypoints
        weights         (B, T, V, J)        <- v25/v46/v50 view weights
        view_mask       (B, T, V)           <- variable-view mask
        R_sefh          (B, T, V, J)        <- v50 per-view reliability
        Λ_sefh          (B, T, J)           <- v50 per-joint log-variance
                │
                ▼
[Canonical Frame Builder]
        ├─ Kinematic tree → local frame F_tj ∈ SO(3) per joint
        └─ Canonical coords c_tj = F_tj^T (p_tj - p_parent) / l_parent
                │
                ▼
[Implicit Neural Geometry (ING) Field]
        ├─ Positional enc. γ(c_tj) ∈ R^(2L)
        ├─ Geometry condition g_tj = MLP_geo(feat_pooled[t,j])
        ├─ SDF-like MLP f_θ: (γ(c), g) → (s, φ)
        │      s  ∈ R   signed-distance energy
        │      φ ∈ R^8 implicit geometry vector
        └─ Zero-initialized output head keeps s0, φ≈0 at start
                │
                ▼
[Geometry-Aware Residual Refiner]
        ├─ MLP_ψ: concat(φ_tj, Δc_tj, g_tj) → Δp_tj ∈ R^3
        ├─ Residual gate λ (init 0.0)
        └─ refined = pred_3d + λ · Δp
                │
                ▼
[Optional inner-loop camera/pose refinement]
        ├─ Fix θ, ψ; run 3 Gauss-Newton steps on
        │   E_total(P) = E_reproj(P; K,R,t, weights) + α · s_θ(P)^2 + β · E_bone(P)
        └─ Produces final refined pose P*
                │
                ▼
Output: refined_pred_3d (B, T, J, 3)
        v52_geometry_loss  (scalar)
```

### Module API

`motionflow_mv/fusion/implicit_neural_geometry_v52.py`:

```python
class ImplicitNeuralGeometryV52(nn.Module):
    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        in_channels: int = 64,
        hidden: int = 128,
        n_layers: int = 3,
        positional_encoding_dim: int = 8,
        use_surface_energy: bool = True,
        use_residual_refiner: bool = True,
        use_inner_loop_refinement: bool = False,
        inner_loop_steps: int = 3,
        inner_loop_lr: float = 1e-2,
        residual_gate_init: float = 0.0,
        surface_loss_weight: float = 0.01,
        bone_loss_weight: float = 0.001,
        reproj_loss_weight: float = 1.0,
    ):
        ...

    def forward(
        self,
        pred_3d: torch.Tensor,               # (B, T, J, 3)
        feat_pooled: torch.Tensor,            # (B, T, J, d)
        K: torch.Tensor,                      # (B, T, V, 3, 3)
        R: torch.Tensor,                      # (B, T, V, 3, 3)
        t: torch.Tensor,                      # (B, T, V, 3)
        points_2d: torch.Tensor,              # (B, T, V, J, 2)
        weights: torch.Tensor,                # (B, T, V, J)
        view_mask: Optional[torch.Tensor] = None,
        parents: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (refined_pred_3d, v52_geometry_loss)."""
        ...
```

## Equations

### Canonical frame
For joint `j` at time `t` with parent `π(j)` and bone length `b_j = ‖p_j − p_{π(j)}‖₂`:

```
R_tj = build_local_frame(p_j, p_{π(j)}, p_{π^2(j)})
c_tj = R_tj^T (p_j − p_{π(j)}) / b_j
```

`c_tj` is the joint's coordinate in its local bone frame. For the root joint we use the pelvis/hip coordinate system.

### Implicit field
Let `γ: R³ → R^(2L)` be sinusoidal positional encoding. The field is:

```
g_tj = MLP_geo(feat_pooled[t, j])                                    # (B, T, J, hidden)
(s_tj, φ_tj) = MLP_field( concat( γ(c_tj), g_tj ) )                # s ∈ R, φ ∈ R^8
```

The field is initialized with the final layer weights set to zero, so at `t=0`:

```
s_tj ≈ 0,   φ_tj ≈ 0  ⇒  refined_pred_3d  pred_3d
```

### Geometry-aware residual

```
Δp_tj = λ · tanh( MLP_ψ( concat(φ_tj, c_tj, g_tj) ) )
refined_pred_3d = pred_3d + Δp
```

`λ` is the residual gate, initialized to `residual_gate_init` (default 0.0). The `tanh` keeps the correction bounded.

### Training losses

**Surface energy loss** (encourages plausible pose manifold):
```
L_surface = (1 / (BTJ)) Σ |s_tj|²
```

**Bone-length consistency** (across time, to suppress jitter):
```
b_tj = ‖p_tj − p_{t,π(j)}‖₂
L_bone = (1 / (BTJ)) Σ_t Σ_j (b_tj − ̄b_j)² / (b_j + ε)
```
where `̄b_j = mean_t b_tj`.

**Reprojection guidance** (ties the field to multi-view evidence):
```
L_reproj = (1 / Σw) Σ_{v,j} w_{tvj} · ‖π_v(refined_pred_3d_{t,j}) − x_{tvj}‖₂
```

**Total auxiliary loss**:
```
L_v52 = w_s · L_surface + w_b · L_bone + w_r · L_reproj
```

At inference with `use_inner_loop_refinement=True`, the module optionally performs a few Gauss-Newton steps:
```
P* = argmin_P [ w_r · L_reproj(P) + w_s · L_surface(P) + w_b · L_bone(P) ]
```
while keeping `MLP_field` and `MLP_ψ` fixed.

## Integration into `OmniMultiViewFusionV5`

### Constructor additions

```python
use_v52_implicit_neural_geometry: bool = False,
v52_ing_hidden: int = 128,
v52_ing_n_layers: int = 3,
v52_ing_positional_encoding_dim: int = 8,
v52_ing_use_surface_energy: bool = True,
v52_ing_use_residual_refiner: bool = True,
v52_ing_use_inner_loop_refinement: bool = False,
v52_ing_inner_loop_steps: int = 3,
v52_ing_inner_loop_lr: float = 1e-2,
v52_ing_surface_loss_weight: float = 0.01,
v52_ing_bone_loss_weight: float = 0.001,
v52_ing_reproj_loss_weight: float = 1.0,
v52_ing_residual_gate_init: float = 0.0,
```

Instantiation inside `__init__` after the v50 SEFH block (so it can consume SEFH reliability and log-variance if available, but it does not require v50):

```python
self.use_v52_implicit_neural_geometry = use_v52_implicit_neural_geometry
if self.use_v52_implicit_neural_geometry:
    from motionflow_mv.fusion.implicit_neural_geometry_v52 import ImplicitNeuralGeometryV52
    self.implicit_neural_geometry_v52 = ImplicitNeuralGeometryV52(
        j=self.j,
        d=self.d,
        in_channels=self.d,
        hidden=v52_ing_hidden,
        n_layers=v52_ing_n_layers,
        positional_encoding_dim=v52_ing_positional_encoding_dim,
        use_surface_energy=v52_ing_use_surface_energy,
        use_residual_refiner=v52_ing_use_residual_refiner,
        use_inner_loop_refinement=v52_ing_use_inner_loop_refinement,
        inner_loop_steps=v52_ing_inner_loop_steps,
        inner_loop_lr=v52_ing_inner_loop_lr,
        surface_loss_weight=v52_ing_surface_loss_weight,
        bone_loss_weight=v52_ing_bone_loss_weight,
        reproj_loss_weight=v52_ing_reproj_loss_weight,
        residual_gate_init=v52_ing_residual_gate_init,
    )
```

### Forward additions

Apply after temporal aggregation (v47/v49) and before the final return. The module receives the ST-pooled features already computed by `feat_pooled = feat.mean(dim=1)` (shape `(B*T, J, d)`, reshaped to `(B, T, J, d)`):

```python
v52_loss = torch.tensor(0.0, device=device, dtype=pred_3d.dtype)
if (
    self.use_v52_implicit_neural_geometry
    and self.implicit_neural_geometry_v52 is not None
):
    refined, v52_loss = self.implicit_neural_geometry_v52(
        pred_3d=pred_3d,
        feat_pooled=feat_pooled.view(B, T, J, self.d),
        K=K_corrected.view(B, T, V, 3, 3),
        R=R.view(B, T, V, 3, 3),
        t=t.view(B, T, V, 3),
        points_2d=points_2d.view(B, T, V, J, 2),
        weights=weights.view(B, T, V, J),
        view_mask=view_mask_flat.view(B, T, V),
        parents=parents,
    )
    pred_3d = refined
```

Then add `v52_loss` to the existing `epi_loss`:

```python
epi_loss = epi_loss + v52_loss
```

The output tuple `(pred_3d, weights, visibility, L, epi_loss, ...)` is unchanged.

## Expected MPJPE impact

| Scenario | Expected delta vs v51 | Rationale |
|----------|----------------------|-----------|
| Full views (H36M/MPI) | −0.3 to −0.8 mm | Bone-length consistency and surface prior reduce small residual errors. |
| Sparse views (v46, k=2) | −0.8 to −1.5 mm | Geometry prior regularises under-constrained triangulation. |
| 3DPW actual (V=1) | −2 to −5 mm | Inner-loop reprojection refinement leverages single-view evidence. |
| Cross-domain gap | −5–10 % relative | Shared implicit surface prior reduces domain-specific drift. |

Conservative smoke target: `val_MPJPE` stays finite and improves over v51 by ≥0.5 mm on the full-view validation set.

## Risks and mitigations

See the dedicated risk report: `docs/swarm_iter26/reports/agent_implicit_neural_geometry_v52_risks.md`.

## 5-step implementation plan

1. **Scaffold module & smoke fixture** — create `motionflow_mv/fusion/implicit_neural_geometry_v52.py` with the canonical-frame builder, positional encoding, and zero-initialized MLP field. Add a CPU unit test in `tests/test_implicit_neural_geometry_v52.py` that asserts identity-at-init and shape correctness.
2. **Wire into `OmniMultiViewFusionV5`** — add the v52 constructor flags, instantiate the module after v47/v49, pass the required tensors, and add `v52_loss` to `epi_loss`. Keep the change behind `use_v52_implicit_neural_geometry`.
3. **Trainer/CLI pass-through** — expose v52 flags in `experiments/train_omniview_fusion_v5_webbridge_multi.py` and add a smoke config `configs/benchmark_v52_implicit_neural_geometry_smoke.yaml` plus script `scripts/run_v52_implicit_neural_geometry_smoke_local_4090.sh`.
4. **Smoke & ablate** — run the smoke on RTX 4090; ablate `use_surface_energy`, `use_residual_refiner`, `use_inner_loop_refinement` separately. Freeze all other weights for the first epoch to preserve v51 warm start.
5. **Full A800 run & paper-story doc** — queue a full run on A800-D using the best smoke config, evaluate `MPJPE@k` on H36M/MPI/AIST/3DPW, and update `docs/results_snapshot_2026_08_09.md` with v52 numbers.

## Relation to other variants

- **v25/v45:** v52 refines the triangulated pose *after* adaptive geometry fusion; it does not replace DLT/weights.
- **v47/v49:** v52 operates after temporal aggregation so that the implicit field sees temporally-smoothed poses and pooled temporal features.
- **v50 SEFH / v51 CDSVR:** v52 can optionally consume SEFH reliability and CDSVR offsets, but it only reads them; it does not modify the v50/v51 internal state.
- **v28/v40 physical losses:** v52 is complementary—physical losses constrain pose semantics, while v52 constrains the 3-D *geometric field* that the body occupies.
