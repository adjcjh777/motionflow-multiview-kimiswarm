# Swarm Iteration 26 — v52 Uncertainty-Weighted Triangulation

**Goal:** Make the core triangulation step learnable by predicting per-view, per-joint precision weights that drive a differentiable weighted DLT, building on v45/v46/v50/v51.

**Status:** proposal / design-only  
**Tracking issue:** #192  
**Base branch:** `v52-uwt`  
**Depends on:** v45-AGF, v46-SVG, v50-SEFH, v51-CDSVR

## Definition of done

- `motionflow_mv/utils/geometry.py` exposes a batched `weighted_dlt_triangulate` that supports `view_mask` and variable views.
- `UncertaintyWeightedTriangulationV52` module is added and identity-at-init.
- `OmniMultiViewFusionV5` can toggle v52 via a single config flag and wires the module after v25/v45 triangulation and before v46/v47/v48/v49/v50/v51.
- Trainer passes `domain_id` and adds the auxiliary `uwt_loss` to the total loss.
- Smoke config + script are created and pass on the local RTX 4090.
- A800 queue entry is added for the full run.
- Unit tests cover identity-at-init, masked variable views, and gradient flow.

## Module, files, and integration points

**Module:** `motionflow_mv/fusion/uncertainty_weighted_triangulation_v52.py`  
**Class:** `UncertaintyWeightedTriangulationV52`

**Files to modify:**

- `motionflow_mv/utils/geometry.py` — add batched `weighted_dlt_triangulate` with `view_mask` and damping support.
- `motionflow_mv/fusion/uncertainty_weighted_triangulation_v52.py` — new module (precision MLP, weighted DLT, residual correction, uncertainty loss).
- `motionflow_mv/fusion/omniview_fusion_v5.py` — add constructor flag, instantiate module, insert call after v25/v45 triangulation, add `uwt_loss` to `epi_loss`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — pass `domain_id` into the model forward and aggregate `uwt_loss`.
- `configs/benchmark_v52_uwt_smoke.yaml` — smoke config copied from v51 CDSVR smoke with v52 flags enabled.
- `scripts/run_v52_uwt_smoke_local_4090.sh` — smoke launch script.
- `scripts/launch_v33_a800_queue.py` — A800 full-run entry.
- `tests/test_uncertainty_weighted_triangulation_v52.py` — unit tests for identity, masking, and gradients.

**Integration point in `OmniMultiViewFusionV5`:**

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v52 UncertaintyWeightedTriangulationV52
    (consumes feat, points_2d, pred_3d_init, cameras, view_mask, domain_id)
    → pred_3d_ref, uwt_loss
    ↓
v46/v47/v48/v49/v50/v51 downstream heads
```

The module is placed so that downstream v46/v47/v48/v49/v50/v51 heads operate on the already precision-weighted 3-D estimate, avoiding double-counting of raw triangulation weights.

## Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v52_uncertainty_weighted_triangulation` | bool | `False` | Master toggle |
| `v52_uwt_hidden` | int | `64` | Precision MLP hidden dimension |
| `v52_uwt_n_layers` | int | `2` | Precision MLP layers |
| `v52_uwt_weight_type` | str | `"per_view_joint"` | `"per_view_joint"` / `"per_view"` / `"per_joint"` |
| `v52_uwt_temperature` | float | `1.0` | Temperature on log-precision |
| `v52_uwt_use_geometry_bias` | bool | `True` | Include ray/epipolar geometry features |
| `v52_uwt_use_feature_bias` | bool | `True` | Include ST feature statistics |
| `v52_uwt_identity_init` | bool | `True` | Zero-initialize final precision MLP layer |
| `v52_uwt_min_weight` | float | `0.05` | Floor on normalized triangulation weight |
| `v52_uwt_loss_weight` | float | `0.01` | Weight of auxiliary uncertainty regularization loss |
| `v52_uwt_damping` | float | `1e-4` | Ridge damping on `(A^T W A)^{-1}` |
| `v52_uwt_warmup_epochs` | int | `0` | Epochs before `uwt_loss` is active |

## Implementation steps (in order)

1. **Add weighted DLT geometry utility.** Implement `weighted_dlt_triangulate` in `motionflow_mv/utils/geometry.py` with batched SVD/pseudo-inverse, `view_mask` support, and ridge damping. Add a unit test that triangulates 2-, 3-, and 4-view synthetic points and checks finite gradients.

2. **Implement `UncertaintyWeightedTriangulationV52`.** Create `motionflow_mv/fusion/uncertainty_weighted_triangulation_v52.py` with the precision MLP, weighted DLT, zero-initialized residual correction, and the auxiliary loss (entropy + consistency term). Keep `weight_type` pluggable and ensure no normalization after the final zero-initialized projection.

3. **Wire into `OmniMultiViewFusionV5`.** Add the v52 flag block to `__init__`, instantiate the module when enabled, and call it in `forward` immediately after the v25/v45 triangulation block. Pass the returned `uwt_loss` into the existing `epi_loss` dictionary with key `v52_uwt`.

4. **Update trainer.** In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, ensure `domain_id` is forwarded to the model and that `loss_dict["v52_uwt"]` (if present) is added to the total loss with weight `v52_uwt_loss_weight`. Honor `v52_uwt_warmup_epochs` by zeroing the loss before the warmup.

5. **Create smoke config and script.** Copy `configs/benchmark_v51_cdsvr_smoke.yaml` to `configs/benchmark_v52_uwt_smoke.yaml`, enable `use_v52_uncertainty_weighted_triangulation`, and create `scripts/run_v52_uwt_smoke_local_4090.sh` that warm-starts from the best available v51 checkpoint.

6. **Add unit/integration tests.** Create `tests/test_uncertainty_weighted_triangulation_v52.py` with three checks: (a) identity-at-init (`|pred_ref - pred_init| < 1e-4`), (b) masked variable views produce correct shapes and no NaN, (c) gradients flow through the precision MLP and weighted DLT.

7. **Run RTX 4090 smoke and ablation.** Run the smoke against the v51 baseline on the same seed and manifest. Ablate `v52_uwt_use_geometry_bias` and `v52_uwt_use_feature_bias`. If smoke passes, run a 500-sample/2-epoch medium ablation comparing v51 vs. v51+UWT.

8. **Queue A800 full run.** Add an entry to `scripts/launch_v33_a800_queue.py` on top of the best v51 checkpoint. Set `d=128`, `train_samples=10000`, 5 epochs, and report `MPJPE@k` plus per-domain metrics every epoch.

## Smoke acceptance criteria (RTX 4090)

- `val_MPJPE@full` is within `1 mm` of the v51-CDSVR baseline on the same smoke config.
- No NaN, Inf, or OOM through at least one full epoch.
- Identity-at-init: loading a v51 checkpoint with v52 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
- Weight sanity: in the first 500 training steps, `max(weight) / mean(weight) < 10` for at least 95 % of joints.
- `MPJPE@2` and `MPJPE@3` are not worse than the v51 baseline.

## A800 full-run criteria

- Base: best available v51-CDSVR checkpoint, warm-starting all new v52 parameters from identity.
- Settings: `d=128`, `n_st_layers=2 or 3`, `batch_size=16`, `clip_len=9`, `train_samples=10000`, 5 epochs, early stopping after 2 epochs without improvement.
- Flags: `use_v52_uncertainty_weighted_triangulation=True`, `v52_uwt_hidden=64`, `v52_uwt_loss_weight=0.01`, `v52_uwt_use_geometry_bias=True`.
- Evaluation: run `experiments/eval_variable_views.py` every epoch and report `MPJPE@2/3/4/full` plus per-domain (H36M / MPI / WebBridge / 3DPW actual) breakdown.
- Go/no-go: proceed to a v52-scaled run only if full-run `MPJPE@full` improves over v51 or if `MPJPE@2/3` improves by ≥ 1 mm with no full-view regression.

## Main risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| **Weight collapse to one view** | `max(weight) / mean(weight)` explodes; MPJPE rises. | Add entropy regularizer in `uwt_loss`; clamp `precision` to `[min_weight, 1/min_weight]`; monitor weight ratio in smoke. |
| **Ill-conditioned weighted DLT** | NaN/Inf in `lstsq` or backward pass, especially at 2 views. | Use SVD pseudo-inverse with `v52_uwt_damping`; skip loss for joints with `< 2` visible views; unit-test on 2-view cases. |
| **Double-counting with v45/v46/v51** | Effective weights become over-conservative and sparse-view metrics regress. | Treat v52 as the *primary* triangulation weight; keep v45/v46/v51 weights for downstream refinement only; freeze v45/v46/v51 for the first epoch. |
| **Identity-at-init failure** | v51 checkpoint regresses at epoch 0 even with v52 enabled but not yet trained. | Zero-initialize the final precision MLP and residual MLP layers; do not place LayerNorm/BatchNorm after the final output projection; add identity test. |
| **Gradient instability in early training** | Loss spikes or divergence in first epoch. | Use a small learning-rate multiplier on precision MLP parameters for the first epoch; enable `v52_uwt_warmup_epochs=1`; clip gradients globally. |

## Notes

- Do not implement any code; this plan is for design review and agent assignment only.
- If the smoke shows that UWT conflicts with v45 adaptive geometry fusion, run an ablation disabling `use_v45_adaptive_geometry_fusion` and keep only UWT + v51.
- Keep the module dependency optional: `OmniMultiViewFusionV5` must still load and run when `use_v52_uncertainty_weighted_triangulation=False`.
