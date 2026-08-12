# Swarm Iteration 27 — v53 Physical-Space Calibration

**Goal:** Calibrate the v52 uncertainty-weighted triangulation output against physical invariants (floor plane and canonical bone lengths) using a gated residual refiner, producing `PhysicalSpaceCalibrationV53`.

**Status:** proposal / design-only  
**Tracking issue:** #201  
**Base branch:** `v53-psc`  
**Depends on:** v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT

## Definition of done

- `PhysicalSpaceCalibrationV53` module is added and identity-at-init.
- `OmniMultiViewFusionV5` can toggle v53 via a single config flag and wires the module immediately after v52 UWT and before v28/v31 physical losses and v46/v47/v48/v49/v50/v51 downstream heads.
- Trainer forwards `domain_id` and adds the auxiliary `psc_loss` to the total loss with a warmup guard.
- Smoke config + script are created and pass on the local RTX 4090.
- A800 queue entry is added for the full run.
- Unit tests cover identity-at-init, masked variable views, and per-domain canonical bone lengths.

## Module, files, and integration points

**Module:** `motionflow_mv/fusion/physical_space_calibration_v53.py`  
**Class:** `PhysicalSpaceCalibrationV53`

**Files to modify:**

- `motionflow_mv/fusion/physical_space_calibration_v53.py` — new module (floor calibration head, bone-length calibration head, gated physical residual refiner, and `psc_loss`).
- `motionflow_mv/fusion/omniview_fusion_v5.py` — add constructor flag, instantiate module when enabled, insert call after the v52 UWT block, add `psc_loss` to `epi_loss`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — pass `domain_id` into the model forward and aggregate `psc_loss` with `v53_psc_loss_weight` and warmup.
- `configs/benchmark_v53_physical_space_calibration_smoke.yaml` — smoke config copied from the v52 UWT smoke with v53 flags enabled.
- `scripts/run_v53_physical_space_calibration_smoke_local_4090.sh` — smoke launch script that warm-starts from the best available v52 checkpoint.
- `scripts/launch_v33_a800_queue.py` — A800 full-run entry.
- `tests/test_physical_space_calibration_v53.py` — unit tests for identity, per-domain canonical lengths, and gradient flow.

**Integration point in `OmniMultiViewFusionV5`:**

```textnpoints_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss
    ↓
v53 PhysicalSpaceCalibrationV53
    (consumes pred_3d_uwt, uwt_weights, points_2d, K, R, t, view_mask, domain_id)
    → pred_3d_psc, psc_loss, floor_height, bone_scale
    ↓
v28/v31 physical losses and v46/v47/v48/v49/v50/v51 downstream heads
```

PSC is placed so that downstream modules operate on a physically calibrated pose and so that existing physical losses (v28 floor, v31 collision) can be reduced or zeroed without removing the new calibration signal.

## Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v53_physical_space_calibration` | bool | `False` | Master toggle |
| `v53_psc_hidden` | int | `64` | Residual MLP hidden dimension |
| `v53_psc_n_layers` | int | `2` | Residual MLP depth |
| `v53_psc_identity_init` | bool | `True` | Zero-initialize final residual layers and gate |
| `v53_psc_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init |
| `v53_psc_use_uwt_weights` | bool | `True` | Use v52 UWT weights as robustness signal |
| `v53_psc_use_floor` | bool | `True` | Enable floor-plane head |
| `v53_psc_use_bone_scale` | bool | `True` | Enable canonical bone-length head |
| `v53_psc_loss_weight` | float | `1.0` | Multiplier on total `L_psc` |
| `v53_psc_floor_weight` | float | `0.01` | Weight of `L_floor` |
| `v53_psc_bone_weight` | float | `0.1` | Weight of `L_bone` |
| `v53_psc_reproj_weight` | float | `0.1` | Weight of reprojection consistency term |
| `v53_psc_warmup_epochs` | int | `0` | Epochs before `psc_loss` contributes to total loss |
| `v53_psc_min_visible_views` | int | `2` | Skip PSC for joints with fewer visible views |

## Implementation steps (in order)

1. **Implement `PhysicalSpaceCalibrationV53`.** Create `motionflow_mv/fusion/physical_space_calibration_v53.py` with three sub-heads: (a) uncertainty-weighted foot-height floor estimator, (b) per-domain canonical bone-length head with soft residual weighting, and (c) a gated physical residual MLP. Enforce identity-at-init by zero-initializing the final residual MLP layer and setting the gate logit to `−6.0`.

2. **Add per-domain canonical skeleton support.** Initialize canonical bone lengths from the training-set empirical mean per domain; use `domain_id` to select the correct canonical skeleton when `v48_domain_generalization` is enabled. Expose a unit test that checks canonical lengths per domain and identity-at-init for each.

3. **Wire into `OmniMultiViewFusionV5`.** Add the v53 flag block to `__init__`, instantiate the module when enabled, and call it in `forward` immediately after the v52 UWT block. Pass the returned `psc_loss` into the existing `epi_loss` dictionary with key `v53_psc`, honoring `v53_psc_warmup_epochs`.

4. **Update trainer.** In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, ensure `domain_id` is forwarded to the model and that `loss_dict["v53_psc"]` (if present) is added to the total loss with weight `v53_psc_loss_weight` only after the warmup epochs.

5. **Create smoke config and script.** Copy `configs/benchmark_v52_uwt_smoke.yaml` to `configs/benchmark_v53_physical_space_calibration_smoke.yaml`, enable `use_v53_physical_space_calibration`, and create `scripts/run_v53_physical_space_calibration_smoke_local_4090.sh` that warm-starts from the best available v52 checkpoint.

6. **Add unit/integration tests.** Create `tests/test_physical_space_calibration_v53.py` with three checks: (a) identity-at-init (`||pred_psc - pred_uwt||_∞ < 1e-4`), (b) per-domain canonical bone lengths produce correct shapes and no NaN, (c) gradients flow through the residual MLP and the three loss terms.

7. **Run RTX 4090 smoke and ablation.** Run the smoke against the v52 baseline on the same seed and manifest. Ablate `v53_psc_use_floor`, `v53_psc_use_bone_scale`, and `v53_psc_reproj_weight`. If smoke passes, run a 500-sample/2-epoch medium ablation comparing v52 vs. v52+PSC.

8. **Queue A800 full run.** Add an entry to `scripts/launch_v33_a800_queue.py` on top of the best v52 checkpoint. Set `d=128`, `train_samples=10000`, 5 epochs, and report `MPJPE@k` plus per-domain metrics every epoch.

## Smoke acceptance criteria (RTX 4090)

- `val_MPJPE@full` is within `1 mm` of the v52-UWT baseline on the same smoke config.
- No NaN, Inf, or OOM through at least one full epoch.
- Identity-at-init: loading a v52 checkpoint with v53 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
- Floor sanity: estimated `floor_height` is finite and foot-joint heights are non-negative in at least `90%` of frames.
- Bone-scale sanity: per-bone scale ratios stay in `[0.5, 2.0]` for at least `95%` of bones.
- `MPJPE@2` and `MPJPE@3` are not worse than the v52 baseline.

## A800 full-run criteria

- Base: best available v52-UWT checkpoint, warm-starting all new v53 parameters from identity.
- Settings: `d=128`, `n_st_layers=2 or 3`, `batch_size=16`, `clip_len=9`, `train_samples=10000`, 5 epochs, early stopping after 2 epochs without improvement.
- Flags: `use_v53_physical_space_calibration=True`, `v53_psc_hidden=64`, `v53_psc_loss_weight=1.0`, `v53_psc_floor_weight=0.01`, `v53_psc_bone_weight=0.1`, `v53_psc_reproj_weight=0.1`.
- Evaluation: run `experiments/eval_variable_views.py` every epoch and report `MPJPE@2/3/4/full` plus per-domain (H36M / MPI / WebBridge / 3DPW actual) breakdown.
- Go/no-go: proceed to a v53-scaled run only if full-run `MPJPE@full` improves over v52 or if `MPJPE@2/3` improves by ≥ 1 mm with no full-view regression.

## Main risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| **Floor-plane assumption is violated** | Feet/ankles pulled to a non-existent floor during jumps/elevated captures; MPJPE rises. | Make the floor head gated and optional (`v53_psc_use_floor`); start with a tiny loss weight and warmup; use a soft floor loss that penalizes only below the plane. |
| **Canonical bone-length prior conflicts with a new dataset** | Bone-length correction biases MPI/3DPW poses toward H36M proportions. | Initialize canonical lengths per domain from training-set empirical mean; use `domain_id` when `v48_domain_generalization` is on; keep correction gated and identity-at-init. |
| **Identity-at-init fails and v52 checkpoints regress** | v52 checkpoint changes by `>0.1 mm` when PSC is enabled before training. | Zero-initialize the final residual MLP layer and set the residual gate logit to `−6.0`; add a unit test asserting `||pred_psc - pred_uwt||_∞ < 1e-4`. |
| **Extra parameters and auxiliary losses cause overfitting** | Smoke validation MPJPE rises after adding PSC. | Keep the module small (`hidden=64`, `n_layers=2`); set loss weights an order of magnitude below the main pose loss; disable heads selectively in ablation. |
| **Interaction with existing physical losses (v28/v31)** | Double-counting physical constraints over-constrains the pose. | Document ordering `v52 → v53 PSC → v28/v31`; reduce or zero `v28_floor_loss_weight` when PSC floor is active; run a four-way ablation on smoke configs. |

## Notes

- Do not implement any code; this plan is for design review and agent assignment only.
- If the smoke shows that PSC conflicts with v28 physical loss, run an ablation disabling `v28_floor_loss_weight` and keep only PSC.
- Keep the module dependency optional: `OmniMultiViewFusionV5` must still load and run when `use_v53_physical_space_calibration=False`.
