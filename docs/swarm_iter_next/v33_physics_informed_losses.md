# v33: Physics-Informed Losses (Floor, Bone, CoM, Contact)

**Direction slug:** `physics_informed_losses`  
**Target:** ICRA/CVPR 2027 — v33 next-iteration experiment  
**Status:** Design proposal; no source files modified.

---

## 1. Problem Statement and Motivation

Current best runs (v31/v32) already show that light physical priors help: the v29 physical-space temporal loss (`use_physical_space_temporal_loss_v29`) penalises foot-floor penetration, bone-length temporal drift and centre-of-mass (CoM) jitter, and v31 adds a self-collision penalty (`use_physical_collision_penalty_v31`).  However, the existing terms are hard-coded, independently weighted, and ignore two physically important cues:

1. **Ground contact dynamics** — feet close to the floor should have low vertical velocity (contact loss), which is only weakly captured by the floor-penetration term.
2. **Motion smoothness via jerk** — the CoM jitter term penalises second-order CoM changes, but it does not regularise the full skeleton’s jerk.
3. **No uncertainty-aware weighting** — physical losses are applied uniformly across batches regardless of per-sample view reliability.

The project already owns a reusable `PhysicsInformedSkeletonDynamicsLoss` in `motionflow_mv/losses/physics_informed_dynamics.py` that implements bone temporal variance, jerk smoothness, ground contact and CoM stability.  This proposal is to **unify and extend** the v29/v31 physical losses into a single v33 physics-informed loss block, expose it through the standard v5 model flag pattern, and make it uncertainty-aware so that its weight scales with triangulation confidence.

---

## 2. Existing Infrastructure (already in the repo)

- `motionflow_mv/losses/physics_informed_dynamics.py`  
  - `PhysicsInformedSkeletonDynamicsLoss(parents, foot_indices, weights={bone, jerk, contact, com})`
  - `bone_length_temporal_variance`, `jerk_smoothness_loss`, `ground_contact_loss`, `center_of_mass_stability_loss`
- `motionflow_mv/fusion/self_evolving_hierarchical_multiview_v29.py`
  - `PhysicalSpaceTemporalLossV29(floor, bone_temporal, com_jitter, warmup_epochs)`
- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Already accepts `use_physical_space_temporal_loss_v29`, `v29_floor_loss_weight`, `v29_bone_temporal_weight`, `v29_com_jitter_weight`, `use_physical_collision_penalty_v31`, etc.
  - Wires them into `forward()` at lines ~1195–1211.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - `build_model_from_args(...)` forwards all v29/v31 flags into the model.
  - `compute_loss()` is where any additional loss terms from the model output are folded into `epi_loss`.
- `scripts/launch_v32_a800_queue.py`
  - Current v32 combined run already enables `use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001`.

---

## 3. Proposed Architecture Changes

### 3.1 New module

Create `motionflow_mv/losses/physics_informed_losses_v33.py` containing a single `PhysicsInformedLossesV33` class.

```python
class PhysicsInformedLossesV33(nn.Module):
    def __init__(
        self,
        parents: List[int],
        foot_indices: List[int],
        floor_weight: float = 0.0,
        bone_temporal_weight: float = 0.0,
        com_jitter_weight: float = 0.0,
        contact_weight: float = 0.0,
        jerk_weight: float = 0.0,
        warmup_epochs: int = 0,
        uncertainty_gate: bool = False,
    ):
        ...

    def forward(self, pred_3d: Tensor, uncertainty: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        ...
```

The module composes the existing helpers:

- **Floor loss** — `floor_loss()` from `motionflow_mv/fusion/physical_space_alignment_v28.py` (or the new helper), using the foot-joint list auto-derived from the skeleton topology.
- **Bone temporal loss** — `bone_length_temporal_variance()` from `motionflow_mv/losses/physics_informed_dynamics.py`.
- **CoM jitter** — `center_of_mass_stability_loss()`.
- **Contact loss** — `ground_contact_loss()` when foot indices are supplied.
- **Jerk loss** — `jerk_smoothness_loss()` across the full skeleton.

Optional **uncertainty gate**: if the triangulation module returns per-sample per-joint uncertainty, scale the physics loss by `1 - exp(-uncertainty)` so that physically noisy samples are regularised more strongly.

### 3.2 Model hooks in `OmniMultiViewFusionV5`

Add new constructor flags (analogous to v29/v31):

- `use_physics_informed_losses_v33: bool = False`
- `v33_floor_weight: float = 0.0`
- `v33_bone_temporal_weight: float = 0.0`
- `v33_com_jitter_weight: float = 0.0`
- `v33_contact_weight: float = 0.0`
- `v33_jerk_weight: float = 0.0`
- `v33_warmup_epochs: int = 0`
- `v33_uncertainty_gate: bool = False`

In `__init__`, when `use_physics_informed_losses_v33` is true, instantiate `PhysicsInformedLossesV33` using the skeleton parent list (`H36M_17_PARENTS` or `MPI_INF_3DHP_28_PARENTS`) and auto-detected foot indices (leaf joints).  In `forward()`, after the final 3-D pose `(B, T, J, 3)` is produced, call the new loss and add it to `epi_loss`.

### 3.3 Training script hooks

In `experiments/train_omniview_fusion_v5_webbridge_multi.py`:

1. Add CLI arguments mirroring the flags above.
2. Forward them in `build_model_from_args()`.
3. No changes to `compute_loss()` are required because the model already returns `epi_loss` as its fifth output and the trainer adds it directly to the supervised loss.

### 3.4 Optional post-processing refiner (stretch)

A second v33 variant could wrap `PhysicsInformedLossesV33` as a tiny Gauss-Newton refiner at inference: freeze the pose predictor and project the prediction onto the physical constraints.  This is left as a follow-up; the first milestone is a training-time loss only.

---

## 4. Training Command / Ablation Flags

### Full smoke (RTX 4090)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --use_physics_informed_losses_v33 \
  --v33_floor_weight 0.01 \
  --v33_bone_temporal_weight 0.01 \
  --v33_com_jitter_weight 0.001 \
  --v33_contact_weight 0.005 \
  --v33_jerk_weight 0.001 \
  --v33_warmup_epochs 3
```

### A800 full run

Re-use the v32 `COMMON_FLAGS` from `scripts/launch_v32_a800_queue.py` and append:

```bash
--use_physics_informed_losses_v33 \
--v33_floor_weight 0.01 \
--v33_bone_temporal_weight 0.01 \
--v33_com_jitter_weight 0.001 \
--v33_contact_weight 0.005 \
--v33_jerk_weight 0.001 \
--v33_warmup_epochs 3 \
--v33_uncertainty_gate
```

### Ablation matrix

| Run | Floor | Bone temp | CoM | Contact | Jerk | Notes |
|-----|-------|-----------|-----|---------|------|-------|
| v33_floor_only | 0.01 | 0 | 0 | 0 | 0 | Isolated floor penalty |
| v33_bone_com | 0 | 0.01 | 0.001 | 0 | 0 | Replicates v29 baseline |
| v33_contact_jerk | 0 | 0 | 0 | 0.005 | 0.001 | New terms only |
| v33_full | 0.01 | 0.01 | 0.001 | 0.005 | 0.001 | Combined |
| v33_full+gate | same | same | same | same | same | With uncertainty gating |

---

## 5. Expected Metrics and Baseline to Beat

- **Primary metric:** validation `MPJPE` on the mixed H36M/MPI-INF-3DHP manifest `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`.
- **Baseline:** the v32 physical-space run in `launch_v32_a800_queue.py` which uses `v29_floor_loss_weight=0.01`, `v29_bone_temporal_weight=0.01`, `v29_com_jitter_weight=0.001`.  Reported best v29-style runs have reached ~28–40 mm on smoke subsets; the full-run target is to **not regress** the v32 baseline and improve by ≥1 mm on the mixed validation set.
- **Secondary metrics:**
  - Per-term loss magnitudes logged via `metrics` dict (floor, bone_temporal, com_jitter, contact, jerk).
  - Variable-view robustness: MPJPE when only 2–4 views are active (the v5 model already supports `use_variable_view_training`).
  - Calibration-robustness: MPJPE under camera perturbation curriculum (`cam_aug_schedule=extended_curriculum`).

---

## 6. Risks / Unknowns

1. **Weight sensitivity.**  Contact and jerk losses can dominate early training if their weights are too high; the `v33_warmup_epochs` linear ramp is required.
2. **Foot-index mismatch.**  Auto-detecting foot indices from leaf joints works for H36M/MPI 17-joint data but may be brittle for 28-joint MPI.  Consider a per-skeleton registry (already present in `cross_view_graph_attention.py`).
3. **Uncertainty gate dependency.**  The optional uncertainty-gated path needs a per-sample uncertainty signal; the model currently returns covariance `L`, not a scalar uncertainty.  If not available, disable `v33_uncertainty_gate` for the first ablation.
4. **Interaction with v28/v32 physical alignment.**  `use_physical_space_alignment_v28` and `use_physical_space_alignment_v32` already apply bounded residual corrections.  Running them together with v33 is allowed but should be ablated to avoid double-counting floor/bone terms.
5. **A800 read-only.**  This proposal only creates a design document; actual GPU runs must be launched from the local RTX 4090 or a properly authorised A800 session.

---

## 7. Files to Touch (implementation TODO)

- **New:** `motionflow_mv/losses/physics_informed_losses_v33.py`
- **Modify:** `motionflow_mv/fusion/omniview_fusion_v5.py` (add flags + instantiate/wire the new loss)
- **Modify:** `experiments/train_omniview_fusion_v5_webbridge_multi.py` (add CLI args and forward to `build_model_from_args`)
- **New:** `scripts/launch_v33_physics_informed_losses_a800_queue.py` (queue wrapper analogous to `launch_v32_a800_queue.py`)

This proposal document is read-only; no source files were modified.
