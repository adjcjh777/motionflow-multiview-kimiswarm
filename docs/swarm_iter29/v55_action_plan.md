# Swarm Iteration 29 — v55 Outlier-Robust Reliability

**Goal:** Improve the robustness of the upstream per-view triangulation weights by adding an identity-at-init Outlier-Robust Reliability (OR2) module after v45 geometry fusion and before v52 Uncertainty-Weighted Triangulation.

**Status:** proposal / design-only  
**Tracking issue:** #208  
**Base branch:** `v55-orr`  
**Depends on:** v45-AGF, v46-SVG, v47-temporal, v48-domain, v50-SEFH, v51-CDSVR, v52-UWT, v53-PSC, v54-PSC-v2

## Definition of done

- `OutlierRobustReliabilityV55` module is added and identity-at-init.
- `OmniMultiViewFusionV5` can toggle v55 OR2 via a single config flag and wires the module immediately after v45 geometry fusion, passing refined `weights_orr` into v52 UWT.
- Trainer forwards `domain_id` (if needed) and adds the auxiliary `orr_loss` to the total loss with a warmup guard.
- Smoke config + script are created and pass on the local RTX 4090.
- A800 queue entry is added for the full run.
- Unit tests cover identity-at-init, Cauchy scale bounds, weight clamping, and outlier rejection.

## Module, files, and integration points

**Module:** `motionflow_mv/fusion/outlier_robust_reliability_v55.py`  
**Class:** `OutlierRobustReliabilityV55`

**Files to modify:**

- `motionflow_mv/fusion/outlier_robust_reliability_v55.py` — new module (per-(view, joint) outlier score, Cauchy M-estimator, residual gate, optional inlier-consistency loss).
- `motionflow_mv/fusion/omniview_fusion_v5.py` — add constructor flag, instantiate module when enabled, insert call **between v45 geometry fusion and v52 UWT**, pass `weights_orr` to v52, and add `orr_loss` to `epi_loss` under key `v55_orr`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — aggregate `loss_dict["v55_orr"]` with `v55_orr_loss_weight` only after `v55_orr_warmup_epochs`.
- `configs/benchmark_v55_orr_smoke.yaml` — smoke config copied from the v54 PSC-v2 smoke with v55 OR2 flags enabled.
- `scripts/run_v55_orr_smoke_local_4090.sh` — smoke launch script that warm-starts from the best available v54 PSC-v2 checkpoint.
- `scripts/launch_v33_a800_queue.py` — A800 full-run entry.
- `tests/test_outlier_robust_reliability_v55.py` — unit tests for identity-at-init, Cauchy scale bounds, weight clamping, and gradient flow.

**Integration point in `OmniMultiViewFusionV5`:**

```text
points_2d, confidences, K, R, t
    ↓
v25/v45 geometry fusion → pred_3d_init, weights_init
    ↓
v55 OutlierRobustReliabilityV55
    (consumes pred_3d_init, weights_init, points_2d, confidences, K, R, t, view_mask)
    → weights_orr, orr_loss
    ↓
v52 UncertaintyWeightedTriangulationV52 → pred_3d_uwt, uwt_weights, uwt_loss
    ↓
v53 PhysicalSpaceCalibrationV53 → pred_3d_psc, psc_loss
    ↓
v54 PhysicalSpaceCalibrationV2V54 → pred_3d_psc2, psc2_loss
    ↓
final residual MLP / v47/v49 temporal / v50 SEFH heads
```

OR2 is placed so that all downstream physical/temporal/SEFH modules consume a triangulation that has already had outlier views down-weighted. It does not replace v45 or v52; it pre-conditions the weights before v52 refines them.

## Config flags and defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v55_outlier_robust_reliability` | bool | `False` | Master toggle |
| `v55_orr_hidden` | int | `64` | MLP hidden dimension |
| `v55_orr_n_layers` | int | `2` | MLP depth |
| `v55_orr_identity_init` | bool | `True` | Zero-init final output layer and gate |
| `v55_orr_residual_gate_init` | float | `−6.0` | Gate logit so `σ(gate) ≈ 0.0025` at init |
| `v55_orr_cauchy_gamma_init` | float | `1.0` | Initial Cauchy scale `γ` |
| `v55_orr_use_geometry_bias` | bool | `True` | Include reprojection/ray/epipolar features |
| `v55_orr_use_feature_bias` | bool | `True` | Include learned v45 feature tokens |
| `v55_orr_min_weight` | float | `0.05` | Floor for refined weights |
| `v55_orr_loss_weight` | float | `0.01` | Multiplier on `orr_loss` |
| `v55_orr_warmup_epochs` | int | `0` | Epochs before `orr_loss` contributes |
| `v55_orr_use_entropy_reg` | bool | `False` | Toggle entropy regularizer |

## Implementation steps (in order)

1. **Implement `OutlierRobustReliabilityV55`.** Create `motionflow_mv/fusion/outlier_robust_reliability_v55.py` with a per-(view, joint) feature MLP, Cauchy M-estimator inlier likelihood, residual gate, and optional inlier-consistency/entropy losses. Enforce identity-at-init by zero-initializing the final MLP layer and setting the gate logit to `−6.0`.

2. **Add per-view feature construction.** Build the per-(view, joint) feature from geometry cues (ray direction, reprojection residual, epipolar distance, triangulation angle) and/or v45 feature tokens, controlled by `v55_orr_use_geometry_bias` and `v55_orr_use_feature_bias`.

3. **Wire into `OmniMultiViewFusionV5`.** Add the v55 OR2 flag block to `__init__`, instantiate the module when enabled, and call it in `forward` immediately after the v45 geometry fusion block. Feed `weights_orr` into v52 UWT and add `orr_loss` to the existing `epi_loss` dictionary with key `v55_orr`, honoring `v55_orr_warmup_epochs`.

4. **Update trainer.** In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, aggregate `loss_dict["v55_orr"]` with weight `v55_orr_loss_weight` only after the warmup epochs.

5. **Create smoke config and script.** Copy `configs/benchmark_v54_psc_v2_smoke.yaml` to `configs/benchmark_v55_orr_smoke.yaml`, enable `use_v55_outlier_robust_reliability`, and create `scripts/run_v55_orr_smoke_local_4090.sh` that warm-starts from the best available v54 PSC-v2 checkpoint.

6. **Add unit/integration tests.** Create `tests/test_outlier_robust_reliability_v55.py` with checks for: (a) identity-at-init (`weights_orr == weights_init` to within `1e-4`), (b) Cauchy scale bounds, (c) weight clamping to `[min_weight, 1.0]`, (d) gradient flow, and (e) synthetic outlier rejection (a corrupted view receives weight below `0.2` in ≥80% of cases).

7. **Run RTX 4090 smoke and ablation.** Run the smoke against the v54 PSC-v2 baseline on the same seed and manifest. Ablate `v55_orr_use_geometry_bias` and `v55_orr_use_feature_bias`. If the smoke passes, run a 500-sample/2-epoch medium ablation comparing v54 vs. v54+OR2.

8. **Queue A800 full run.** Add an entry to `scripts/launch_v33_a800_queue.py` on top of the best v54 PSC-v2 checkpoint. Set `d=128`, `train_samples=10000`, 5 epochs, and report `MPJPE@k` plus per-domain metrics every epoch.

## Smoke acceptance criteria (RTX 4090)

- `val_MPJPE@full` is within `1 mm` of the v54-PSC-v2 baseline on the same smoke config.
- `MPJPE@2` and `MPJPE@3` are not worse than the v54 baseline; target ≥ `1 mm` improvement on sparse views.
- No NaN, Inf, or OOM through at least one full epoch.
- Identity-at-init: loading a v54 PSC-v2 checkpoint with v55 OR2 enabled and no training step changes `val_MPJPE@full` by `< 0.1 mm`.
- Weight sanity: ≥95% of `weights_orr` are finite and in `[0.05, 1.0]`; mean refined weight is between `0.3` and `0.9` for non-masked tokens.
- Outlier rejection: on synthetic outlier injection, the corrupted view receives a weight below `0.2` in ≥80% of cases.
- Loss sanity: `orr_loss` is finite and stable after the first 100 steps.

## A800 full-run criteria

- **Base:** best available v54-PSC-v2 checkpoint, warm-starting all new v55 OR2 parameters from identity.
- **Settings:** `d=128`, `n_st_layers=2 or 3`, `batch_size=16`, `clip_len=9`, `train_samples=10000`, 5 epochs, early stopping after 2 epochs without improvement.
- **Flags:** `use_v55_outlier_robust_reliability=True`, `v55_orr_hidden=64`, `v55_orr_loss_weight=0.01`, `v55_orr_use_geometry_bias=True`, `v55_orr_use_feature_bias=True`, `v55_orr_cauchy_gamma_init=1.0`.
- **Evaluation:** run `experiments/eval_variable_views.py` every epoch and report `MPJPE@2/3/4/full` plus per-domain (H36M / MPI / WebBridge / 3DPW actual) breakdown.
- **Go/no-go:** proceed to a scaled run only if full-run `MPJPE@full` improves over v54 or if `MPJPE@2/3` improves by ≥ `1 mm` with no full-view regression.

## Main risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| **Gate fails closed or open** | `weights_orr` collapses to uniform or to zero. | Initialize gate to `−6.0`; clamp `weights_orr` to `[min_weight, 1.0]`; add identity-at-init unit test. |
| **Cauchy scale drifts** | `γ` becomes very small and rejects valid views. | Parameterize `γ` via `softplus + 0.5` lower bound; add loss weight warmup. |
| **v52 UWT overwrites OR2 weights** | No visible gain if v52 re-learns its own weights. | Feed `weights_orr` as the *initial* v52 weight and keep the v52 residual small; ablate `v55_orr_loss_weight`. |
| **Conflicts with v51 CDSVR reliability** | Double suppression of rare but correct views. | Gate OR2 on v51 domain-conditioned reliability only at init; keep both branches additive. |
| **Sparse-view degeneracy** | With `min_views=2`, OR2 may reject a needed second view. | Enforce `v55_orr_min_weight > 0`; mask the top-`min_views` highest `weights_init` from being suppressed. |
| **Identity-at-init regression** | v54 PSC-v2 checkpoint changes by `>0.1 mm` when OR2 is enabled. | Zero-init final MLP layer and gate; unit test `||weights_orr − weights_init||_∞ < 1e-4`. |

## Notes

- Do not implement any code; this plan is for design review and agent assignment only.
- Keep the module optional: `OmniMultiViewFusionV5` must still load and run when `use_v55_outlier_robust_reliability=False`.
- If the smoke shows that OR2 conflicts with v51 CDSVR, run an ablation disabling `v51` and keep OR2 only, or vice versa.
- If OR2 passes smoke, stack the v55 runner-ups (TCL, MVTS, GAAP, TTSR) only after OR2 is proven on the A800 full run.
