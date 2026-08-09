# v50 Self-Evolution Feedback Head (SEFH)

## Module name
`SelfEvolutionFeedbackHeadV50` — a lightweight, gradient-safe feedback head that closes the prediction↔uncertainty loop by predicting updated per-view reliability and per-joint uncertainty from reprojection, temporal, and epipolar residuals.

## Architecture description
The head sits after the v48 domain-invariant 3D pose output (or after v47 temporal aggregation when v48 is disabled). It takes the current 3D pose estimate, the input 2D keypoints, and camera parameters, computes per-view reprojection residuals, temporal inconsistency residuals (frame-to-frame 3D velocity), and epipolar-line-distance residuals, and feeds these into a small MLP that predicts a refined reliability weight for every `(view, joint)` pair and a log-variance uncertainty for each joint. The output is used to re-weight the triangulation/geometry-fusion input in a single additional refinement step, with the entire operation kept inside the training graph. A residual gate initialized near identity ensures the module does not perturb the already-strong v46/v48 baseline at startup.

## New config flags and defaults
- `use_v50_self_evolution_feedback_head` (default `False`) — master switch.
- `v50_sefh_hidden` (default `64`) — hidden dimension of the residual MLP.
- `v50_sefh_num_layers` (default `2`) — number of MLP layers.
- `v50_sefh_dropout` (default `0.1`) — dropout inside the head.
- `v50_sefh_reproj_weight` (default `1.0`) — weight for reprojection residual branch.
- `v50_sefh_temporal_weight` (default `0.5`) — weight for temporal residual branch.
- `v50_sefh_epipolar_weight` (default `0.5`) — weight for epipolar residual branch.
- `v50_sefh_max_refinement_steps` (default `1`) — number of refinement iterations, capped to avoid v27-style TTE instability.
- `v50_sefh_identity_init_gate` (default `True`) — initialize the final output gate to near-zero so the module is identity at init.

## Loss term
A combined self-evolution loss `L_sefh = v50_sefh_loss_weight * (L_reproj_nll + L_residual_smooth + L_reliability_entropy)` with default `v50_sefh_loss_weight = 0.01`.
- `L_reproj_nll`: negative log-likelihood of 2D reprojection under the predicted reliability and joint log-variance.
- `L_residual_smooth`: L2 penalty on the change in reliability across time to prevent flickering.
- `L_reliability_entropy`: small entropy regularization that keeps reliability distribution from collapsing to all-one.

## Evaluation metric
Primary: `MPJPE@full` and `MPJPE@k` for `k = 2, 3, 4` via the canonical v49 `MPJPE@k` protocol. Secondary: Spearman correlation between predicted per-view reliability and the corresponding reprojection residual (target `> 0.3`). Tertiary: per-joint log-variance calibration (average predicted uncertainty vs. per-joint MPJPE).

## Expected MPJPE impact
Based on v46-SVG smoke epoch-1 `val_MPJPE = 32.97 mm`, enabling v50 SEFH on top of the v48-lite stack is expected to improve sparse-view robustness by `MPJPE@2 -2 to -4 mm` and `MPJPE@3 -1 to -2 mm`, while keeping `MPJPE@full` within `0.5 mm` of the v48 baseline. The full-view gain is expected to be small (≈ -0.3 mm) because the baseline is already strong; the main benefit is reliability-aware refinement under missing views and domain shift.

## Main risk / mitigations
- **Risk: Collapse to identity / no learning signal.** The head may learn to output near-uniform reliability and fail to close the feedback loop. **Mitigation:** enforce identity-at-init via the zero-initialized gate, add the auxiliary residual-reduction loss, and freeze base weights for the first epoch so only the head updates.
- **Risk: v27 TTE-style instability.** Multiple iterative refinement steps can diverge. **Mitigation:** cap `v50_sefh_max_refinement_steps = 1`, clamp reliability to `[0.05, 1.0]`, and supervise the refinement with a single-step residual loss rather than unrolled recursion.
- **Risk: Confounding with v37/v39 reliability heads.** Earlier self-critique modules already consume reprojection residuals. **Mitigation:** treat v50 as the *unified* replacement for v37/v39 when enabled; add a config check that raises if `use_v37_self_critique_reliability` or `use_v39_reliability_coupled_refinement` is active simultaneously.
- **Risk: Extra latency and memory.** The head adds a small MLP but still sits on the critical path. **Mitigation:** keep hidden dim at 64 by default; profile on the v49-lite 4090 smoke before promoting to A800.

## Next action
Create `motionflow_mv/fusion/self_evolution_feedback_head_v50.py` and a smoke config `configs/benchmark_v50_self_evolution_feedback_head_smoke.yaml`, warm-start from the best v48-lite checkpoint, and validate that `MPJPE@full` stays within 1 mm of v48 while `MPJPE@2/3` improves.
