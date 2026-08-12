# v55 Temporal Consistency Loss (TCL)

## 1. Module name and one-line purpose

**Module:** `TemporalConsistencyLossV55` → `motionflow_mv/fusion/temporal_consistency_loss_v55.py`

**One-line purpose:** A loss-only velocity and acceleration regulariser that penalises high-frequency joint jitter weighted by per-joint v52/v54 uncertainty, improving temporal smoothness of the physically calibrated pose without adding any new forward parameters.

## 2. Where it sits in the OmniMultiViewFusionV5 forward pass

TCL is a **loss-only** module and does not alter the forward graph. It consumes the final 3-D pose estimate from the existing v54 PSC-v2 stage and adds a single auxiliary loss term to the total training objective.

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss
    ↓
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, psc_loss
    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_loss
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads → pred_3d_final
    ↓
v55 TemporalConsistencyLossV55
    (consumes pred_3d_final, uwt_weights, psc2 bone/contact masks)
    → tcl_loss
```

Placement in the loss stack: after all forward modules have produced `pred_3d_final`, the trainer calls `TemporalConsistencyLossV55(pred_3d_final, ...)` and adds `tcl_loss` to the total loss with a warm-up schedule.

## 3. Inputs, outputs, and shapes

### Inputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Final 3-D pose sequence in metres. |
| `uwt_weights` | `(B, T, J, V)` or `(B, T, J)` | Per-joint v52 uncertainty-derived weights (renormalised per joint). |
| `joint_mask` | `(B, T, J)` | Boolean mask for joints that are visible/triangulated. |
| `bone_mask` | `(B, T, B)` | Boolean mask for visible parent-child bones (from v54). |
| `contact_mask` | `(B, T, J)` | Float mask near 1.0 when the foot joint is likely in contact (from v54 contact head). |
| `domain_id` | `(B,)` | Optional per-sample domain index for domain-agnostic loss normalisation. |

### Outputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `tcl_loss` | scalar | Weighted sum of `L_vel`, `L_acc`, and optional `L_bone_vel` terms. |
| `loss_dict` | dict | Breakdown `{v55_tcl_vel, v55_tcl_acc, v55_tcl_bone_vel, v55_tcl_total}` for logging. |

## 4. Architecture: layers, heads, losses, identity-at-init mechanism

TCL has **no trainable layers** and therefore no forward parameters. It is implemented as a pure loss function with configurable weighting and masking.

### Loss terms

1. **Velocity smoothness `L_vel`**
   ```
   Δp[t] = p[t] - p[t-1]                          # (B, T-1, J, 3)
   w_vel[t] = min_joints(uwt_weights[t], uwt_weights[t-1]) * joint_mask[t] * joint_mask[t-1]
   L_vel = mean( w_vel * ||Δp||_2 )
   ```

2. **Acceleration smoothness `L_acc`**
   ```
   a[t] = p[t+1] - 2p[t] + p[t-1]                 # (B, T-2, J, 3)
   w_acc[t] = min_joints(uwt_weights[t+1], uwt_weights[t], uwt_weights[t-1]) * joint_mask[t+1] * joint_mask[t] * joint_mask[t-1]
   L_acc = mean( w_acc * ||a||_2 )
   ```

3. **Bone-length velocity `L_bone_vel` (optional)**
   ```
   b[t] = p_child[t] - p_parent[t]               # (B, T, B, 3)
   Δb[t] = b[t] - b[t-1]
   L_bone_vel = mean( bone_mask * ||Δb||_2 )
   ```
   This term discourages rapid bone-length fluctuations caused by jitter.

### Uncertainty and contact weighting

- Per-joint weights are derived from v52 UWT weights and renormalised to `[0, 1]`.
- Frames where the joint is part of a contact limb (foot/ankle with `contact_mask ≈ 1`) receive an additional multiplier `v55_tcl_contact_weight` to enforce near-zero foot velocity during contact.
- Loss is normalised by the number of valid (unmasked) joint-frames to avoid domination by missing data.

### Warm-up / identity-at-init mechanism

- **No new parameters** means the model is automatically identity-at-init in the forward pass.
- The loss contribution is **ramped from zero** for `v55_tcl_warmup_epochs` epochs using a linear schedule, ensuring the existing v54 checkpoint path is preserved at the start of training.
- Even after warm-up, `v55_tcl_loss_weight` starts at a small value (`0.01` default) and can be scaled up in ablations.

## 5. Expected MPJPE impact (full/sparse views) and main risks

| View setting | Expected MPJPE impact |
|--------------|----------------------|
| Full views   | `−0.3 to −0.8 mm` on average; larger on fast-motion clips (`−1.0 to −1.5 mm`). |
| Sparse `@2/3`| Small consistent gains (`−0.2 to −0.6 mm`) because smoothing reduces noise amplification when fewer views are available. |
| Smoke (RTX 4090) | ≤ `0.5 mm` change from the v54-PSC-v2 baseline because the loss is ramped and small. |

### Main risks and mitigations

| Risk | Symptom | Mitigation |
|------|---------|------------|
| **Over-smoothing of fast motion** | Joints lag behind rapid limb movements; MPJPE rises on action sequences. | Use acceleration term only (`v55_tcl_acc_weight`) and keep velocity weight small; make loss local to contact frames. |
| **Contact-mask errors amplify bias** | Feet pulled to floor during jumps. | Gate contact term with v54 velocity threshold; weight it down (`v55_tcl_contact_weight ≤ 0.01`). |
| **Loss weight too high, baseline regresses** | Smoke val_MPJPE rises by >1 mm. | Linear warm-up from 0 and default weight `0.01`; unit-test that zero loss weight recovers baseline exactly. |
| **Uncertainty weight collapse** | `uwt_weights` near zero for hard joints make the loss vanish. | Renormalise weights per-joint across time and clip minimum weight to `0.05`. |

## 6. Smoke acceptance criteria

1. **Identity-at-init / baseline preservation:** Loading the best v54 checkpoint and running one validation batch with `v55_tcl_loss_weight=0` yields the same `val_MPJPE` as v54 within `1e-4 mm`.
2. **No forward change:** With `use_v55_temporal_consistency_loss=False`, the model output is bit-identical to v54.
3. **Smoke stability:** On the RTX 4090 smoke config, `val_MPJPE@full` stays within `1 mm` of the v54 baseline.
4. **NaN/Inf/OOM:** No numerical instability through one full smoke epoch.
5. **Sparse-view parity:** `MPJPE@2` and `MPJPE@3` do not worsen versus the v54 baseline.
6. **Loss sanity:** With synthetic constant-velocity sequences, `L_acc = 0` and `L_vel` is proportional to the injected noise level.

## 7. Required new files and files to modify

### New files

- `motionflow_mv/fusion/temporal_consistency_loss_v55.py` — pure loss module implementing `TemporalConsistencyLossV55`.
- `configs/benchmark_v55_temporal_consistency_loss_smoke.yaml` — smoke config copied from v54 with v55 flags enabled.
- `scripts/run_v55_temporal_consistency_loss_smoke_local_4090.sh` — smoke launch script warm-starting from the best v54 checkpoint.
- `tests/test_temporal_consistency_loss_v55.py` — unit tests for loss sanity, warm-up schedule, weight clipping, and identity-at-init (zero weight == baseline).

### Files to modify

- `motionflow_mv/fusion/omniview_fusion_v5.py` — add the v55 flag block to `__init__`; optionally cache `uwt_weights` and v54 masks; no change to forward output.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — after computing the total loss, call `TemporalConsistencyLossV55` on `pred_3d_final`; add `v55_tcl_loss` to the total loss with `v55_tcl_loss_weight` and warm-up guard.
- `scripts/launch_v33_a800_queue.py` — add `v55_temporal_consistency_loss_on_v54` A800 queue entry.
- `AGENTS.md` — append v55 conventions to the status tables once the smoke passes.

## Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_temporal_consistency_loss` | bool | `False` | Master toggle |
| `v55_tcl_loss_weight` | float | `0.01` | Multiplier on the total TCL loss |
| `v55_tcl_vel_weight` | float | `0.5` | Weight of `L_vel` inside the total TCL loss |
| `v55_tcl_acc_weight` | float | `1.0` | Weight of `L_acc` inside the total TCL loss |
| `v55_tcl_bone_vel_weight` | float | `0.1` | Weight of `L_bone_vel` inside the total TCL loss |
| `v55_tcl_contact_weight` | float | `0.01` | Extra weight on contact-foot joints |
| `v55_tcl_min_uwt_weight` | float | `0.05` | Floor for per-joint uncertainty weights |
| `v55_tcl_warmup_epochs` | int | `1` | Epochs over which the loss weight linearly ramps from 0 |
| `v55_tcl_use_contact` | bool | `True` | Use v54 contact mask for foot-velocity loss |
| `v55_tcl_use_bone_vel` | bool | `True` | Enable bone-length velocity term |

## A800 full-run criteria

- **Base:** best available v54-PSC-v2 checkpoint, warm-starting with TCL weight ramped from zero.
- **Settings:** same as v54 scaled run (`d=128`, `n_st_layers=2`, `batch_size=16`, `clip_len=9`, `train_samples=10000`, 5 epochs, early stopping after 2 epochs without improvement).
- **Flags:** `use_v55_temporal_consistency_loss=True`, `v55_tcl_loss_weight=0.01`, `v55_tcl_vel_weight=0.5`, `v55_tcl_acc_weight=1.0`, `v55_tcl_bone_vel_weight=0.1`, `v55_tcl_warmup_epochs=1`.
- **Evaluation:** run `experiments/eval_variable_views.py` every epoch and report `MPJPE@2/3/4/full`, per-domain breakdown, and the logged `v55_tcl_*` loss components.
- **Go/no-go:** proceed to a scaled run only if full-run `MPJPE@full` improves over v54 or if `MPJPE@2/3` improves by ≥ 1 mm with no full-view regression.
