# v51 Neural Bundle Adjustment v2 (NBAv2)

**Focus area:** geometric refinement / self-evolution loop  
**Module file (proposed):** `motionflow_mv/fusion/neural_bundle_adjustment_v51.py`  
**Status:** design proposal — no GPU run required

## 1. Architecture description

NBAv2 closes the self-evolution loop by adding a lightweight, differentiable bundle-adjustment step after the v50 Self-Evolution Feedback Head (SEFH). Instead of relying on a one-shot DLT triangulation, NBAv2 iteratively refines the initial 3-D pose using per-view reliability scores and per-joint log-variance produced by v50. The core is a small 2-layer MLP that predicts a Gauss-Newton-style residual update in pose space, plus an optional per-camera extrinsic correction branch. A zero-initialized residual gate keeps the module identity-at-init so the strong v46/v48/v50 full-view baseline is preserved.

Inputs to `NeuralBundleAdjustmentV2` are:

- `P_0`: initial 3-D joint positions from DLT/v45 adaptive triangulation,
- `x_2d`: 2-D keypoints per view,
- `K`, `R`, `t`: camera intrinsics and extrinsics,
- `w_v`: per-view reliability from v50 SEFH (`V`,),
- `σ_j`: per-joint log-variance from v50 SEFH (`J`,).

The refinement runs `v51_nba_num_steps` (default 1) update steps. Each step computes the weighted reprojection residual

```
r_{v,j} = w_v · exp(-σ_j) · ρ_j · (π_v(P_j) - x_{v,j})
```

where `π_v(·)` is the projection for view `v` and `ρ_j` is a learned attention-like weight from a tiny 1-D Conv1D over joints. The MLP then maps the flattened residual vector to a 3-D pose correction `ΔP` and, if enabled, a per-camera SE(3) correction `Δξ_v`. Corrections are added with a residual gate initialized to zero. No full Jacobian or Schur complement is formed; the update is learned but constrained by the reprojection residual, so it behaves like a single GN step in feature space.

## 2. New config flags with defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_neural_bundle_adjustment_v2` | bool | `False` |
| `v51_nba_hidden` | int | `64` |
| `v51_nba_num_steps` | int | `1` |
| `v51_nba_use_camera_correction` | bool | `False` |
| `v51_nba_reproj_weight` | float | `1.0` |
| `v51_nba_temporal_weight` | float | `0.3` |
| `v51_nba_camera_correction_weight` | float | `0.01` |
| `v51_nba_identity_init_gate` | bool | `True` |
| `loss.v51_nba_loss_weight` | float | `0.01` |

## 3. Loss term

```
L_nba = λ · [ (1/VJ) Σ_{v,j} w_v exp(-σ_j) ||π_v(P_j + ΔP_j) - x_{v,j}||_2
            + α · (1/(T-1)J) Σ_t ||ΔP_t - ΔP_{t-1}||_2
            + β · ||Δξ||_2 ]
```

where `λ = loss.v51_nba_loss_weight`, `α = v51_nba_temporal_weight`, and `β = v51_nba_camera_correction_weight`. The first term is the uncertainty-weighted reprojection negative log-likelihood; the second enforces temporal smoothness; the third regularizes the optional camera correction.

## 4. Evaluation metric

- `MPJPE@k` for `k = 2, 3, 4, full` via `experiments/eval_variable_views.py`.
- Mean reprojection error (MRE) after refinement.
- Spearman(`w_v`, reprojection residual) > 0.3 to confirm v50 reliability remains meaningful.
- Per-joint uncertainty calibration (ECE-style binning on `exp(-σ_j)` vs. absolute error).

## 5. Expected MPJPE impact

- `MPJPE@2`: −2 to −4 mm versus v50 baseline.
- `MPJPE@3`: −1 to −2 mm.
- `MPJPE@full`: ±0.5 mm (no regression expected thanks to identity-at-init).
- 3DPW actual mode: −3 to −5 mm on `MPJPE@2`, because NBAv2 can down-weight noisy wild-world detections using v50 uncertainty.

## 6. Main risk and mitigation

**Risk:** Camera-correction branch or iterative refinement can overfit to the training rig's calibration, hurting cross-domain 3DPW actual performance and destabilizing the v50 baseline.

**Mitigation:**
- Disable camera correction by default (`v51_nba_use_camera_correction=False`).
- Initialize the residual gate to zero and freeze the base model for the first epoch, mirroring v50 SEFH practice.
- Cap iterations at one and clamp corrections to a small bounded range.
- Smoke-test with `loss.v51_nba_loss_weight=0.001` before using the default 0.01.

## 7. Integration notes

- Wire `NeuralBundleAdjustmentV2` into `motionflow_mv/fusion/omniview_fusion_v5.py` after `SelfEvolutionFeedbackHeadV50` when `use_v51_neural_bundle_adjustment_v2=True`.
- Add a guard that raises if `use_v50_self_evolution_feedback_head=False`, because NBAv2 consumes SEFH outputs.
- Smoke config: `configs/benchmark_v51_neural_bundle_adjustment_v2_smoke.yaml`.
- Acceptance: `val_MPJPE@full` within 1 mm of v50, `MPJPE@2` improves ≥2 mm, no NaN/OOM.
