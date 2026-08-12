# v50 Dynamic View Reliability (DynamicViewReliabilityV50)

## Architecture

`DynamicViewReliabilityV50` is a lightweight, inference-time adaptive head that recomputes per-view, per-joint reliability every frame from diagnostic consistency signals. It sits immediately after the v46 Sparse-View Generalization (SVG) reliability head and consumes: (1) the base v46/v37 reliability score, (2) the per-view reprojection residual produced by the v49 Self-Evolution Feedback Head, (3) the 2D keypoint temporal jump magnitude in the current frame, and (4) the cross-view epipolar agreement residual. These four signals are concatenated per (view, joint, frame), passed through a 2-layer MLP with hidden size `v50_dvr_hidden`, and mapped to a residual reliability update via a sigmoid-activated gate: `r_vjt^final = sigmoid(r_vjt^base + MLP(features))`. The module is identity-at-init (the MLP starts near zero and the sigmoid centers at 0.5), so the baseline is preserved. The final dynamic weights multiply the per-view contributions before adaptive triangulation/aggregation, letting the model suppress transiently corrupted views (motion blur, partial occlusion, calibration drift) without touching camera geometry or learned view embeddings.

## New config flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v50_dynamic_view_reliability` | bool | `False` | Enable the dynamic view reliability head. |
| `v50_dvr_hidden` | int | `32` | Hidden size of the 2-layer MLP. |
| `v50_dvr_input_features` | list[str] | `["v46_reliability", "reproj_residual", "temporal_jump", "epipolar_error"]` | Diagnostic signals fed to the head. |
| `v50_dvr_loss_weight` | float | `0.01` | Weight of the auxiliary residual-reweighting loss. |
| `v50_dvr_min_reliability` | float | `0.05` | Floor on dynamic reliability to avoid zero-weighted views. |
| `v50_dvr_temporal_window` | int | `3` | Number of neighbouring frames used to compute `temporal_jump`. |
| `v50_dvr_entropy_reg` | float | `0.001` | Small entropy regularizer on the gate to prevent collapse. |

## Loss term

`L_dvr = λ * mean_v,j ( r_vjt * ρ_vjt / (sum_u r_ujt + ε) ) - β * H(r)`

`ρ_vjt` is the per-view normalized reprojection residual, `λ` is `v50_dvr_loss_weight` (default 0.01), `β` is `v50_dvr_entropy_reg` (default 0.001), and `H` is the per-frame reliability entropy. This loss pushes reliability down for views with large reprojection residuals while the entropy term prevents the trivial uniform-collapse solution.

## Evaluation metric

Primary metrics are `MPJPE@full`, `MPJPE@2`, `MPJPE@3`, and `MPJPE@4` reported by `experiments/eval_variable_views.py`. Secondary diagnostics: Spearman correlation between the final dynamic reliability and the per-view reprojection residual, and per-view reliability calibration error (`E[|r - 1_{residual<threshold}|]`).

## Expected MPJPE impact

Relative to the v46-SVG baseline (`MPJPE@2 ~` current smoke, full-run target ~35-40 mm), dynamic reliability is expected to improve the sparse-view regime the most. Target: `MPJPE@2` -3 mm, `MPJPE@3` -2 mm, `MPJPE@full` within 1 mm. The gain should be larger on 3DPW actual-mode sequences where transient occlusion and calibration drift are common.

## Main risk / mitigations

**Collapse to uniform reliability.** The head may learn to ignore the diagnostic inputs and emit near-constant weights, providing no benefit. Mitigations: identity-at-init, residual gate, entropy regularization, and freezing the base v46/v37 weights for the first epoch. Smoke success gate: Spearman(reliability, residual) < -0.2 before full run is allowed.
