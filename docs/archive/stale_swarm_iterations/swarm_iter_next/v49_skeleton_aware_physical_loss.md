# v49: Skeleton-Aware Physical Loss with Self-Evolution Feedback

**Status:** Proposal  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #168 (placeholder — create once v48-domain lands)  
**Depends on:** #164 (v48 domain generalization), #160 (v46 sparse-view generalization), #162 (v47 temporal aggregation)

---

## 1. Problem Statement

v40's `SkeletonPhysicalLossV40` applies a single scalar weight to each physical term (bone length, joint limit, symmetry, floor, collision). This is blind to two sources of information the model already produces:

1. **Per-joint / per-view uncertainty.** In v45/v46/v47, the triangulation and temporal heads emit per-joint log-variance and per-view reliability. Enforcing bone-length symmetry on a joint whose 2-D detections are uncertain can pull the model *away* from the correct geometry.
2. **Domain-specific skeleton statistics.** v48 exposes `dataset_id`, but v40 uses one global bone-length reference. Studio (H36M/MPI) and in-the-wild (3DPW) subjects have different average bone lengths and contact dynamics.

The result is that v40 can be **too rigid in uncertain regions and too loose in well-constrained regions**, slowing convergence and occasionally increasing `val_MPJPE`.

---

## 2. Proposed Approach

v49 replaces the static v40 loss with an **uncertainty-aware, self-evolving physical loss**. It keeps the same four v40 terms but scales each one by a per-joint confidence derived from the model's own reprojection and triangulation residuals. Over training, a small online EMA of per-joint bone-length distributions is maintained per domain and fed back as a refined prior.

### 2.1 High-level pipeline

```text
Input: 3D pose P_t from v47 temporal head  (B, T, J, 3)
        + per-joint log-variance λ_t          (B, T, J)
        + per-view reliability r_v              (B, T, V)
        + dataset_id d                          (B,)
        |
        ▼
[ v49 Skeleton-Aware Physical Loss ]
        |
        ├── Confidence weight:  w_j = sigmoid(-λ_t[j] * α + β)
        │     down-weights physical terms for uncertain joints.
        ├── Per-domain bone-length prior μ_d, σ_d
        │     updated online from confident predictions.
        ├── Existing v40 terms (bone, joint-limit, symmetry, floor, collision)
        │     weighted by w_j and by a term-specific scalar.
        └── Temporal consistency regulariser
              penalises rapid changes in bone length across frames.
```

### 2.2 Fit with v46-v48 and the paper pipeline

- **v46 Sparse-View Generalization:** when views are dropped, triangulated joints become noisier. v49's confidence weighting automatically reduces the physical loss in those frames, preventing bad sparse-view gradients from dominating.
- **v47 Temporal Aggregation:** the temporal head already enforces smooth 3-D trajectories. v49 adds a **temporal bone-length consistency** term so the skeleton shape is stable across time, not just joint positions.
- **v48 Domain Generalization:** per-domain bone-length EMAs let the physical prior adapt to studio vs. in-the-wild subjects without manual tuning. The domain discriminator in v48 remains unchanged.
- **Paper story:** the loss is a concrete example of the self-evolution loop — the model critiques its own uncertainty and uses that to shape the physical prior.

---

## 3. Concrete Code-Level Changes

### 3.1 New module

`motionflow_mv/losses/skeleton_physical_loss_v49.py`:

```python
class SkeletonPhysicalLossV49(nn.Module):
    def __init__(
        self,
        parents: List[int],
        symmetry_pairs: Optional[List[Tuple[int, int]]] = None,
        foot_indices: Optional[List[int]] = None,
        bone_weight: float = 0.05,
        joint_limit_weight: float = 0.01,
        symmetry_weight: float = 0.02,
        floor_weight: float = 0.02,
        collision_weight: float = 0.001,
        temporal_bone_weight: float = 0.01,
        uncertainty_temp: float = 1.0,
        uncertainty_bias: float = 0.0,
        use_per_domain_prior: bool = True,
        num_domains: int = 6,
        prior_ema_decay: float = 0.99,
        warmup_epochs: int = 2,
    ) -> None:
        ...

    def forward(
        self,
        pred: torch.Tensor,                 # (B, T, J, 3)
        target: Optional[torch.Tensor] = None,
        log_var: Optional[torch.Tensor] = None,   # (B, T, J)
        dataset_id: Optional[torch.Tensor] = None,  # (B,)
        epoch: int = 0,
    ) -> torch.Tensor:
        ...
```

### 3.2 Changes to existing files

| File | Change |
|------|--------|
| `motionflow_mv/losses/skeleton_physical_loss_v40.py` | Keep as-is; v49 is a subclass/wrapper that imports it. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Return per-joint log-variance `λ` alongside the 3-D pose when `use_v45_adaptive_geometry_fusion` is enabled. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add CLI flags (below); wire `log_var` and `dataset_id` into the loss call; update `metrics` dict. |
| `motionflow_mv/losses/__init__.py` | Export `SkeletonPhysicalLossV49`. |
| `configs/benchmark_v49_skeleton_physical_smoke.yaml` | New smoke config. |
| `scripts/run_v49_skeleton_physical_smoke_local_4090.sh` | New smoke script. |

### 3.3 New training flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_skeleton_physical_loss_v49` | bool | `False` | Master switch. |
| `v49_bone_weight` | float | `0.05` | Bone-length matching weight. |
| `v49_joint_limit_weight` | float | `0.01` | Joint-limit weight. |
| `v49_symmetry_weight` | float | `0.02` | Symmetry weight. |
| `v49_floor_weight` | float | `0.02` | Floor/contact weight. |
| `v49_collision_weight` | float | `0.001` | Self-collision weight. |
| `v49_temporal_bone_weight` | float | `0.01` | Temporal bone-length smoothness. |
| `v49_uncertainty_temp` | float | `1.0` | Sharpness of confidence weighting. |
| `v49_uncertainty_bias` | float | `0.0` | Offset for confidence weighting. |
| `v49_use_per_domain_prior` | bool | `True` | Maintain per-domain bone-length EMA. |
| `v49_prior_ema_decay` | float | `0.99` | EMA decay for online prior. |
| `v49_warmup_epochs` | int | `2` | Linear ramp-up of all physical terms. |

### 3.4 Minimal integration pseudocode

In the training step:

```python
if args.use_skeleton_physical_loss_v49:
    v49_loss = skeleton_physical_loss_v49(
        pred_3d,
        target=y,
        log_var=model_output.get("log_var"),  # (B, T, J)
        dataset_id=dataset_id,
        epoch=epoch,
    )
    loss = loss + v49_loss
    metrics["v49_phys_loss"] = v49_loss.item()
```

---

## 4. Risks / Failure Modes

| Risk | Mitigation |
|------|------------|
| Uncertainty gating collapses to zero and physical loss is ignored. | Initialise bias so `w_j ≈ 0.9`; clamp `w_j ∈ [0.1, 1.0]`. |
| Per-domain EMA diverges for small domains (e.g. 3DPW). | Fall back to global prior when domain count is low; use `num_domains` from manifest. |
| Temporal bone term over-smoothes fast motion. | Keep `v49_temporal_bone_weight` small (default 0.01) and ramp it in. |
| Interaction with v48 DDWL | Apply DDWL only to the MSE/reprojection terms; keep physical loss domain-weighted only through the per-domain prior. |
| v47 temporal head not yet landed | Gate the temporal term on `T > 1`; it degrades to a per-frame term if temporal aggregation is off. |

---

## 5. Success Metrics and Recommended Experiments

### 5.1 Success metrics

1. **No regression:** `val_MPJPE@full` within 1 mm of the v40/v48 baseline on the same data.
2. **Sparse-view gain:** on v46 2-view/3-view subsets, v49 improves MPJPE by ≥3% relative over v40.
3. **Plausibility:** lower mean per-frame bone-length variance and fewer floor penetrations on validation.
4. **Stability:** no NaN/OOM and no overfitting spike after epoch 1.

### 5.2 Recommended experiments

| Stage | Hardware | Config | Expected outcome |
|-------|----------|--------|------------------|
| Smoke | RTX 4090 | `configs/benchmark_v49_skeleton_physical_smoke.yaml` (d=64, 200 samples, 2 epochs) | `val_MPJPE` < 75 mm, no NaN, `v49_phys_loss` finite. |
| Full | A800-D | v48-domain checkpoint + v49 physical loss | Match or beat v48 `val_MPJPE@full`; ≥3% improvement at `MPJPE@2/3`. |
| Ablation | RTX 4090 | no uncertainty gating / no per-domain prior / no temporal bone term | Identify necessary components. |

### 5.3 Smoke YAML snippet

```yaml
model:
  use_v46_sparse_view_generalization: true
  use_v47_temporal_aggregation: true
  use_v48_domain_generalization: true

  # v49 physical loss
  use_skeleton_physical_loss_v49: true
  v49_bone_weight: 0.05
  v49_joint_limit_weight: 0.01
  v49_symmetry_weight: 0.02
  v49_floor_weight: 0.02
  v49_collision_weight: 0.001
  v49_temporal_bone_weight: 0.01
  v49_uncertainty_temp: 1.0
  v49_use_per_domain_prior: true
  v49_warmup_epochs: 2
```

---

## 6. Self-Evolution Feedback Loop

v49 is the physical-loss stage of the project's self-evolution loop:

1. **Forward pass:** v45/v46 produce 3-D poses and per-joint uncertainty `λ`.
2. **Residual computation:** reprojection and triangulation residuals refine the uncertainty estimate (v37/v45).
3. **Confidence weighting:** v49 converts `λ` into a confidence mask `w_j` and applies it to physical terms.
4. **Online prior update:** confident predictions update per-domain bone-length EMAs `μ_d, σ_d`, which feed back into the next iteration's bone-length loss.

The loop lets the physical prior **emerge from the data** rather than being hand-tuned, and it prevents the loss from overriding weak-but-correct geometric evidence.

---

## 7. Next Steps

1. Wait for v48-domain smoke results (#164).
2. Implement `SkeletonPhysicalLossV49` and unit tests in `tests/test_skeleton_physical_loss_v49.py`.
3. Wire the v49 flags into `experiments/train_omniview_fusion_v5_webbridge_multi.py`.
4. Run smoke on RTX 4090 and compare against the v40 baseline on the same v48 config.
5. Queue full A800 run from the best v48 checkpoint.

---

## See Also

- `docs/v40_physical_loss_improvements.md` — v40 baseline and pending collision/floor improvements.
- `docs/v41_domain_loss_redesign.md` — DDWL conventions to respect when combining with v49.
- `docs/proposals/v48_domain_generalization.md` — v48 dependency and per-domain flags.
- `motionflow_mv/losses/skeleton_physical_loss_v40.py` — baseline implementation.
