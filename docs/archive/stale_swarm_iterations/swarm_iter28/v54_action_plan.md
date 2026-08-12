# Swarm Iteration 28 — v54 Physical-Space Calibration v2

**Goal:** Tighten the physical-space alignment stage by adding a skeleton-graph, joint-level physical refiner on top of v53 PSC, producing `PhysicalSpaceCalibrationV2V54`.

**Status:** proposal / design-only  
**Tracking issue:** #207  
**Base branch:** `v54-psc-v2`  
**Depends on:** v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT, v53-PSC

## Definition of done

- `PhysicalSpaceCalibrationV2V54` module is added and identity-at-init.
- `OmniMultiViewFusionV5` can toggle v54 via a single config flag and wires the module immediately after v53 PSC and before the final residual MLP / v47/v49 temporal / v50 SEFH heads.
- Trainer forwards `domain_id` and adds the auxiliary `psc2_loss` to the total loss with a warmup guard.
- Smoke config + script are created and pass on the local RTX 4090.
- A800 queue entry is added for the full run.
- Unit tests cover identity-at-init, per-domain canonical bone scales, floor sanity, and GNN gradient flow.

## Module, files, and integration points

**Module:** `motionflow_mv/fusion/physical_space_calibration_v2_v54.py`  
**Class:** `PhysicalSpaceCalibrationV2V54`

**Files to modify:**

- `motionflow_mv/fusion/physical_space_calibration_v2_v54.py` — new module (floor/contact head, per-domain canonical bone-length head, skeleton-graph physical refiner, `psc2_loss`).
- `motionflow_mv/fusion/omniview_fusion_v5.py` — add constructor flag, instantiate module when enabled, insert call after the v53 PSC block, add `psc2_loss` to `epi_loss`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — pass `domain_id` into the model forward and aggregate `psc2_loss` with `v54_psc2_loss_weight` and warmup.
- `configs/benchmark_v54_psc_v2_smoke.yaml` — smoke config copied from the v53 PSC smoke with v54 flags enabled.
- `scripts/run_v54_psc_v2_smoke_local_4090.sh` — smoke launch script that warm-starts from the best available v53 checkpoint.
- `scripts/launch_v33_a800_queue.py` — A800 full-run entry.
- `tests/test_physical_space_calibration_v2_v54.py` — unit tests for identity, per-domain canonical lengths, floor/contact, and gradient flow.

**Integration point in `OmniMultiViewFusionV5`:**

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss
    ↓
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, floor_height, bone_scale, psc_loss
    ↓
v54 PhysicalSpaceCalibrationV2V54
    (consumes pred_3d_psc, uwt_weights, points_2d, K, R, t, view_mask, domain_id)
    → pred_3d_psc2, psc2_loss, floor_height_v2, bone_scale_v2
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

PSC-v2 is placed so that downstream modules operate on a locally physically calibrated pose. It does not replace v53; it refines it.

## Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v54_physical_space_calibration_v2` | bool | `False` | Master toggle |
| `v54_psc2_hidden` | int | `64` | GNN / MLP hidden dimension |
| `v54_psc2_n_layers` | int | `2` | Refiner MLP depth |
| `v54_psc2_num_domains` | int | `8` | Number of domains for per-domain bone scales |
| `v54_psc2_identity_init` | bool | `True` | Zero-initialize final residual layers and gate |
| `v54_psc2_residual_gate_init` | float | `-6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init |
| `v54_psc2_use_floor` | bool | `True` | Enable floor-plane head |
| `v54_psc2_use_contact` | bool | `True` | Enable velocity-gated foot-floor contact loss |
| `v54_psc2_use_bone_scale` | bool | `True` | Enable per-domain canonical bone-scale head |
| `v54_psc2_use_temporal_smoothness` | bool | `True` | Enable temporal smoothness on the correction |
| `v54_psc2_use_gnn` | bool | `True` | Use skeleton-graph refiner (fallback: per-joint MLP) |
| `v54_psc2_gnn_layers` | int | `1` | Number of GNN layers |
| `v54_psc2_loss_weight` | float | `1.0` | Multiplier on total `L_psc2` |
| `v54_psc2_floor_weight` | float | `0.01` | Weight of `L_floor` |
| `v54_psc2_bone_weight` | float | `0.05` | Weight of `L_bone` |
| `v54_psc2_contact_weight` | float | `0.01` | Weight of `L_contact` |
| `v54_psc2_temporal_weight` | float | `0.01` | Weight of `L_temporal` |
| `v54_psc2_reproj_weight` | float | `0.1` | Weight of reprojection consistency term |
| `v54_psc2_contact_velocity_thresh` | float | `0.3` | Foot velocity threshold (m/s) for contact loss |
| `v54_psc2_min_visible_views` | int | `2` | Skip bones/joints with fewer visible views |
| `v54_psc2_warmup_epochs` | int | `0` | Epochs before `psc2_loss` contributes to total loss |

## Implementation steps (in order)

1. **Implement `PhysicalSpaceCalibrationV2V54`.** Create `motionflow_mv/fusion/physical_space_calibration_v2_v54.py` with four sub-heads: (a) UWT-weighted foot-height floor estimator + soft floor loss, (b) velocity-gated contact loss, (c) per-domain canonical bone log-scale head with soft residual weighting, and (d) a shallow skeleton-graph residual refiner. Enforce identity-at-init by zero-initializing the final GNN/MLP layer and setting the gate logit to `−6.0`.

2. **Add per-domain canonical skeleton support.** Initialize canonical bone log-scales to zero and use `domain_id` to select the correct domain when `v48_domain_generalization` is enabled. Mask out bones whose parent or child joint has fewer than `v54_psc2_min_visible_views` visible views. Expose a unit test that checks canonical scales per domain and identity-at-init for each.

3. **Wire into `OmniMultiViewFusionV5`.** Add the v54 flag block to `__init__`, instantiate the module when enabled, and call it in `forward` immediately after the v53 PSC block. Pass the returned `psc2_loss` into the existing `epi_loss` dictionary with key `v54_psc2`, honoring `v54_psc2_warmup_epochs`.

4. **Update trainer.** In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, ensure `domain_id` is forwarded to the model and that `loss_dict["v54_psc2"]` (if present) is added to the total loss with weight `v54_psc2_loss_weight` only after the warmup epochs.

5. **Create smoke config and script.** Copy `configs/benchmark_v53_physical_space_calibration_smoke.yaml` to `configs/benchmark_v54_psc_v2_smoke.yaml`, enable `use_v54_physical_space_calibration_v2`, and create `scripts/run_v54_psc_v2_smoke_local_4090.sh` that warm-starts from the best available v53 checkpoint.

6. **Add unit/integration tests.** Create `tests/test_physical_space_calibration_v2_v54.py` with four checks: (a) identity-at-init (`||pred_psc2 - pred_psc||_∞ < 1e-4`), (b) per-domain canonical bone scales produce correct shapes and no NaN, (c) floor/contact sanity on synthetic data, (d) gradients flow through the GNN and the four loss terms.

7. **Run RTX 4090 smoke and ablation.** Run the smoke against the v53 baseline on the same seed and manifest. Ablate `v54_psc2_use_floor`, `v54_psc2_use_contact`, `v54_psc2_use_bone_scale`, and `v54_psc2_use_gnn`. If smoke passes, run a 500-sample/2-epoch medium ablation comparing v53 vs. v53+PSC-v2.

8. **Queue A800 full run.** Add an entry to `scripts/launch_v33_a800_queue.py` on top of the best v53 checkpoint. Set `d=128`, `train_samples=10000`, 5 epochs, and report `MPJPE@k` plus per-domain metrics every epoch.

## Smoke acceptance criteria (RTX 4090)

- `val_MPJPE@full` is within `1 mm` of the v53-PSC baseline on the same smoke config.
- No NaN, Inf, or OOM through at least one full epoch.
- Identity-at-init: loading a v53 checkpoint with v54 enabled and no training step changes `val_MPJPE` by `< 0.1 mm`.
- Floor sanity: estimated `floor_height_v2` is finite and foot-joint heights are non-negative in at least `90%` of frames when `use_floor=True`.
- Bone-scale sanity: per-bone scale ratios `exp(s_b)` stay in `[0.5, 2.0]` for at least `95%` of bones.
- Contact sanity: contact loss is zero for frames where all foot joints have velocity above the threshold.
- `MPJPE@2` and `MPJPE@3` are not worse than the v53 baseline.

## A800 full-run criteria

- Base: best available v53-PSC checkpoint, warm-starting all new v54 parameters from identity.
- Settings: `d=128`, `n_st_layers=2 or 3`, `batch_size=16`, `clip_len=9`, `train_samples=10000`, 5 epochs, early stopping after 2 epochs without improvement.
- Flags: `use_v54_physical_space_calibration_v2=True`, `v54_psc2_hidden=64`, `v54_psc2_loss_weight=1.0`, `v54_psc2_floor_weight=0.01`, `v54_psc2_bone_weight=0.05`, `v54_psc2_contact_weight=0.01`, `v54_psc2_temporal_weight=0.01`, `v54_psc2_reproj_weight=0.1`.
- Evaluation: run `experiments/eval_variable_views.py` every epoch and report `MPJPE@2/3/4/full` plus per-domain (H36M / MPI / WebBridge / 3DPW actual) breakdown.
- Go/no-go: proceed to a v54-scaled run only if full-run `MPJPE@full` improves over v53 or if `MPJPE@2/3` improves by ≥ 1 mm with no full-view regression.

## Main risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| **GNN over-smooths anatomical detail** | Wrists/ankles collapse toward limb means; MPJPE rises. | Keep graph shallow (`gnn_layers=1`), hidden dim `64`, gate logit `−6.0`; clamp per-joint residual during first epoch. |
| **Floor/contact constraints hurt non-upright motion** | Feet pulled to floor during jumps/sitting; MPJPE rises. | Use soft floor loss (penalize below plane only); gate contact by foot velocity (`contact_velocity_thresh=0.3 m/s`); make floor/contact optional via flags. |
| **Canonical bone scales conflict across datasets** | Bone-length correction biases MPI/3DPW poses toward H36M proportions. | Initialize log-scales to zero; select per-domain scales via `domain_id`; mask invisible bones. |
| **Identity-at-init fails and v53 checkpoints regress** | v53 checkpoint changes by `>0.1 mm` when PSC-v2 is enabled before training. | Zero-initialize final GNN/MLP layer, bone-scale output, and gate logit `−6.0`; add unit test asserting `||pred_psc2 - pred_psc||_∞ < 1e-4`. |
| **Extra GNN + temporal losses cause overfitting or OOM** | Smoke validation MPJPE rises after adding PSC-v2; RTX 4090 OOM. | Keep graph sparse (parent-child only), use temporal stride for losses if needed, provide `use_gnn=False` fallback, and keep module small (`hidden=64`). |

## Notes

- Do not implement any code; this plan is for design review and agent assignment only.
- If the smoke shows that PSC-v2 conflicts with v53 bone/floor losses, run an ablation disabling the overlapping v53 terms and keep only PSC-v2's local heads.
- Keep the module dependency optional: `OmniMultiViewFusionV5` must still load and run when `use_v54_physical_space_calibration_v2=False`.
