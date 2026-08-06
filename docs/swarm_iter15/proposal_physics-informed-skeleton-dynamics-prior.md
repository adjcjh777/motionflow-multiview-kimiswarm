# Iter15 Proposal: Physics-Informed Skeleton Dynamics Prior

## One-sentence hypothesis

Adding a lightweight, differentiable physics-informed skeleton dynamics prior—enforcing temporal smoothness, constant bone-length ratios, and soft ground-contact constraints—on top of the iter14 anchor model will improve 3D pose robustness across views and produce more physically plausible motion without changing the existing cross-view attention / principal-point correction pipeline.

## Related existing files/modules

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py` — current iter14 anchor model (RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint).
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py` — parent residual model; the residual MLP head is the natural injection point.
- `motionflow_mv/losses/bone_length.py`, `motionflow_mv/losses/velocity.py`, `motionflow_mv/losses/reprojection.py` — existing geometric/temporal losses.
- `motionflow_mv/losses/__init__.py` — loss registry (to be extended).
- `motionflow_mv/fusion/__init__.py` / `ray_attention_temporal_crossview_residual_principal_point_module.py` — FusionModule registration (to be extended).
- `experiments/train_ray_attention_reproducible.py` — reproducible training harness used for smoke tests.
- `configs/train_ray_attention_reproducible.yaml` — config template for smoke runs.

## Proposed code changes

### 1. New loss module: `motionflow_mv/losses/physics_informed_dynamics.py`

Implements `PhysicsInformedSkeletonDynamicsLoss` with four weakly-supervised terms:

- `bone_length_temporal_variance`: penalizes *temporal variance* of each bone length over a clip, rather than matching a single GT skeleton. This is dataset-agnostic and lets the model learn subject-specific bone lengths.
- `jerk_smoothness`: penalizes third-order finite differences of 3D joint positions, encouraging physically smooth motion.
- `ground_contact`: when foot joint indices are provided, penalizes vertical velocity of the feet when they are close to the estimated ground plane.
- `center_of_mass_stability`: penalizes high-frequency COM jerk and large vertical COM acceleration.

Public API:

```python
loss_fn = PhysicsInformedSkeletonDynamicsLoss(
    parents=[...],          # parent-index list, -1 for roots
    foot_indices=[...],     # optional feet joints
    weights={"bone": 1.0, "jerk": 0.1, "contact": 0.5, "com": 0.2},
)
loss = loss_fn(pred_3d)  # pred_3d: (B, T, J, 3)
```

### 2. New model: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_physics_model.py`

Thin subclass of the iter14 anchor model that adds an optional **temporal skeleton dynamics refiner** after the per-frame triangulation:

```python
class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointPhysics(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    def __init__(self, ..., dynamics_hidden=128, dynamics_layers=1):
        ...
        self.dynamics_refiner = nn.GRU(3, dynamics_hidden, dynamics_layers, batch_first=True, bidirectional=True)
        self.dynamics_head = nn.Linear(dynamics_hidden * 2, 3)
```

- The refiner takes the raw triangulated sequence `pred_3d_raw` (already computed in the parent forward), runs a small bidirectional GRU per joint, and predicts a residual that is added to the refined 3D pose.
- It does **not** replace the existing residual MLP; it complements it.
- The physics loss is applied in the training script; the model itself only exposes the extra temporal dynamics residual path.

### 3. Registration (to be done when promoting to a running experiment)

- Add imports in `motionflow_mv/losses/__init__.py` for `physics_informed_dynamics_loss`.
- Add a new FusionModule wrapper `RayAttentionTemporalCrossviewResidualPrincipalPointPhysicsFusionModule` in a new module or extend the existing registration module.
- Update `motionflow_mv/fusion/__init__.py` to register it.

These registration steps are intentionally left out of the current skeleton so the repository stays in a reversible state.

## Training/smoke plan (≤5 epochs, RTX 4090)

1. Copy `configs/train_ray_attention_reproducible.yaml` to `configs/train_physics_dynamics_smoke.yaml`.
2. Point the smoke config to a small subset (e.g. `s_01` of H36M or the MPI-INF-3DHP S2/Seq1 smoke split) with `epochs: 5`, `batch_size: 16`.
3. In `experiments/train_ray_attention_reproducible.py` (or a dedicated trainer), instantiate the new physics model and add:
   ```python
   phys_loss = PhysicsInformedSkeletonDynamicsLoss(parents=parents_17, foot_indices=[3, 6, 10, 13])
   loss = mse_loss + 0.01 * phys_loss(pred_3d)
   ```
4. Run a single smoke epoch to verify:
   - forward/backward pass succeeds,
   - GPU memory stays within ~10 GB on the RTX 4090,
   - training loss decreases monotonically.
5. If the smoke is clean, run the full 5-epoch smoke on the same dataset.

**Estimated runtime:** 5 epochs on H36M `s_01` (~30k clips) at batch size 16 should take ~30–45 minutes on the RTX 4090. MPI-INF-3DHP smoke is similar.

## Success metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Clean MPJPE on MPI-INF-3DHP S2/Seq1 | ≤ 9.0 mm | Anchor is 9.32 mm; the prior should push below 9.0 mm. |
| Cross-view robustness | Reprojection error std. dev. ↓ 10% | More stable per-view weights across dropout/outlier augmentation. |
| Temporal consistency (jerk) | ↓ 15% average jerk vs. anchor on held-out clips | Physics loss directly targets this. |
| Bone-length variance over clip | ↓ 20% | Stronger skeletal consistency. |
| Training stability | No NaN/gradient explosion in ≤5 epochs | Required for further scaling. |

## Risk and fallback

- **Risk: auxiliary loss overpowers 3D MSE.** The physics terms are unsupervised and could initially dominate. Mitigation: start with very small weights (λ ≤ 0.01), and anneal them over the first epoch. Fallback: disable the most aggressive term (`ground_contact`) and keep only `jerk` + bone-length variance.
- **Risk: GRU dynamics refiner increases memory beyond RTX 4090 limits.** Mitigation: reduce `dynamics_hidden` to 64 or remove the GRU entirely and apply the physics loss only on the final 3D output (zero extra parameters). This is a pure loss-only fallback.
- **Risk: no improvement on MPI-INF-3DHP.** Mitigation: the change is reversible; if metrics do not improve, revert to the anchor and use the physics loss only as a regularizer in the final training stage (last 10 epochs) rather than end-to-end.
- **Risk: registration changes break existing checkpoints.** Mitigation: keep the new model unregistered during the smoke; only register it after a successful 5-epoch validation.
