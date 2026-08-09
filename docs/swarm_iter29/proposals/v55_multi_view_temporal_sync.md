# v55 Multi-View Temporal Sync (MVTS)

**One-line purpose:** Refine the v54 physically-calibrated 3-D pose by per-joint temporal attention over `(time, view)` tokens, letting clean or temporally-consistent views borrow information across the clip while preserving the warm-started v54 baseline.

---

## 1. Module name and tracking

- **Module:** `motionflow_mv/fusion/multi_view_temporal_sync_v55.py`
- **Class:** `MultiViewTemporalSyncV55`
- **Branch:** `v55-mvts`
- **Tracking issue:** #208
- **Base:** reuses v54 PSC-v2 output and v52 UWT robustness weights
- **Depends on:** v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT, v53-PSC, v54-PSC-v2

---

## 2. Placement in `OmniMultiViewFusionV5`

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
v55 MultiViewTemporalSyncV55
    (consumes pred_3d_psc2, uwt_weights, points_2d, K, R, t, view_mask, domain_id)
    → pred_3d_mvts, mvts_loss
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

v55 sits **after** the v54 physical-space calibration block and **before** any final pose head or residual MLP, so downstream modules consume a temporally-synchronized 3-D pose.

---

## 3. Inputs, outputs, and shapes

**Inputs**

| Tensor | Shape | Description |
|---|---|---|
| `pred_3d_psc2` | `(B, T, J, 3)` | v54 calibrated 3-D pose |
| `uwt_weights` | `(B, T, V, J)` | v52 per-(view, joint) triangulation weights |
| `points_2d` | `(B, T, V, J, 2)` | Input 2-D keypoints |
| `confidences` | `(B, T, V, J)` | Input confidence scores |
| `K` | `(B, T, V, 3, 3)` | Camera intrinsics |
| `R` | `(B, T, V, 3, 3)` | Camera rotations |
| `t` | `(B, T, V, 3)` | Camera translations |
| `view_mask` | `(B, T, V)` | Valid-view mask |
| `domain_id` | `(B, T)` | Domain index (optional, for embeddings) |

**Outputs**

| Tensor | Shape | Description |
|---|---|---|
| `pred_3d_mvts` | `(B, T, J, 3)` | Temporally synchronized 3-D pose |
| `mvts_loss` | scalar | Auxiliary sync + temporal consistency loss |

---

## 4. Architecture

### 4.1 Token construction (per joint)

For each joint `j` and time `t`, build `V` view tokens from:

- Per-view reprojection residual `r_{t,v,j}` of `pred_3d_psc2`.
- v52 UWT weight `w_{t,v,j}` (log-weight + normalized weight).
- Per-view ray direction and camera center embedding.
- Optional domain embedding.

Each token is projected to `v55_mvts_token_dim` (default `16`) by a linear layer whose **output is zero-initialized** so that the initial attention input is zero and no view gets a privileged signal at init.

### 4.2 (time, view) attention

For each joint independently:

1. Flatten the `T × V` tokens into a single sequence of length `T·V` (or windowed if `v55_mvts_window` is set).
2. Add learned positional embeddings for relative time step and view index.
3. Apply a shallow Transformer encoder:
   - hidden dim: `v55_mvts_hidden` (default `64`)
   - num layers: `v55_mvts_n_layers` (default `2`)
   - num heads: `v55_mvts_n_heads` (default `4`)
   - dropout: `v55_mvts_dropout` (default `0.1`)

The encoder learns to attend across time and views, borrowing clean or temporally-consistent observations to suppress single-frame outliers.

### 4.3 Residual correction and gate

- Pool the attended tokens back to `(B, T, J, hidden)` by mean-pooling over views.
- A tiny MLP maps to a 3-D per-joint correction `Δp`.
- A scalar gate `g = σ(gate_logit)` controls the correction.
- `pred_3d_mvts = pred_3d_psc2 + g · Δp`

**Identity-at-init mechanism:**

- The final MLP that produces `Δp` is **zero-initialized**.
- `gate_logit` is initialized to `v55_mvts_residual_gate_init = -6.0`, so `g  0.0025` at init.
- Positional embeddings and intermediate attention layers are initialized normally; only the correction path is zeroed.

At init, `pred_3d_mvts ≈ pred_3d_psc2` and `mvts_loss ≈ 0`, preserving the v54 checkpoint exactly.

### 4.4 Losses

| Loss | Weight | Description |
|---|---|---|
| `L_reproj` | `v55_mvts_reproj_weight` (default `0.1`) | Reprojection of corrected pose, weighted by `uwt_weights` |
| `L_temporal` | `v55_mvts_temporal_weight` (default `0.01`) | Temporal velocity smoothness on corrected pose, weighted by `1 - uwt_weights` (uncertain joints are smoothed more) |
| `L_mvts` | `v55_mvts_loss_weight` (default `1.0`) | `v55_mvts_loss_weight * (L_reproj + L_temporal)` |

The loss is added to `epi_loss` under key `v55_mvts` only after `v55_mvts_warmup_epochs`.

---

## 5. Expected MPJPE impact and risks

| View setting | Expected impact |
|---|---|
| Full views | `−0.5` to `−1.0 mm` |
| Sparse `@2` / `@3` | `−1.5` to `−3.0 mm` by borrowing clean views across time |

**Main risks and mitigations**

| Risk | Symptom | Mitigation |
|---|---|---|
| **Temporal over-smoothing** | Fast motion blurred; wrists/ankles lag | Keep attention shallow (`n_layers=2`), use per-joint attention (no mixing across joints), and weight smoothness by uncertainty |
| **Memory blow-up from `T×V` tokens** | RTX 4090 OOM on smoke | Default `v55_mvts_window=None` uses full clip; if OOM, add `window` flag for local temporal windows |
| **Identity-at-init regression** | v54 checkpoint changes `>0.1 mm` when v55 enabled | Zero-init final correction MLP and gate logit `−6.0`; unit test enforces `||pred_mvts - pred_psc2||_∞ < 1e-4` |
| **Double-counting with v47/v49 temporal heads** | Final residual temporal head and MVTS both regularize time | v55 is placed before the final head, so the head can still learn; keep v55 loss weights low and warm-start |

---

## 6. Smoke acceptance criteria

- `val_MPJPE@full` is within `1 mm` of the v54-PSC-v2 baseline on the same smoke config.
- No NaN, Inf, or OOM through at least one full epoch.
- Identity-at-init: loading a v54 checkpoint with v55 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
- `||pred_3d_mvts - pred_3d_psc2||_∞ < 1e-4` at init on a synthetic batch.
- `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline.
- Temporal smoothness: mean per-joint acceleration is finite and non-increasing for at least `80%` of test clips.

---

## 7. Required new files and files to modify

**New files**

- `motionflow_mv/fusion/multi_view_temporal_sync_v55.py` — `MultiViewTemporalSyncV55` module.
- `configs/benchmark_v55_multi_view_temporal_sync_smoke.yaml` — smoke config copied from v54 with v55 flags enabled.
- `scripts/run_v55_multi_view_temporal_sync_smoke_local_4090.sh` — smoke launch script warm-starting from the best v54 checkpoint.
- `tests/test_multi_view_temporal_sync_v55.py` — unit tests for identity-at-init, token shapes, gradient flow, and smoke sanity.

**Files to modify**

- `motionflow_mv/fusion/omniview_fusion_v5.py` — add constructor flag, instantiate `MultiViewTemporalSyncV55` when enabled, insert the call after the v54 PSC-v2 block, and add `mvts_loss` to `epi_loss`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — aggregate `loss_dict["v55_mvts"]` with `v55_mvts_loss_weight` and warmup guard.
- `scripts/launch_v33_a800_queue.py` — add A800 full-run entry `v55_multi_view_temporal_sync_on_v54`.

---

## 8. Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_multi_view_temporal_sync` | bool | `False` | Master toggle |
| `v55_mvts_hidden` | int | `64` | Transformer hidden dimension |
| `v55_mvts_n_layers` | int | `2` | Transformer encoder layers |
| `v55_mvts_n_heads` | int | `4` | Attention heads |
| `v55_mvts_token_dim` | int | `16` | Per-view token dimension |
| `v55_mvts_dropout` | float | `0.1` | Dropout in Transformer |
| `v55_mvts_identity_init` | bool | `True` | Zero-init correction MLP and gate |
| `v55_mvts_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init |
| `v55_mvts_use_view_positional_embedding` | bool | `True` | Add learned view-index embedding |
| `v55_mvts_use_time_positional_embedding` | bool | `True` | Add learned relative-time embedding |
| `v55_mvts_window` | int | `None` | Optional local temporal window length (None = full clip) |
| `v55_mvts_loss_weight` | float | `1.0` | Multiplier on `L_mvts` |
| `v55_mvts_reproj_weight` | float | `0.1` | Weight of reprojection term |
| `v55_mvts_temporal_weight` | float | `0.01` | Weight of temporal smoothness term |
| `v55_mvts_warmup_epochs` | int | `0` | Epochs before `mvts_loss` contributes |
