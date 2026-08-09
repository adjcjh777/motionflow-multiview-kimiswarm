# v51 Uncertainty-Guided View Dropout (UGVD)

## One-line idea

Replace v46's uniform view dropout with a reliability-conditioned policy: lower-reliability views are dropped more often during training, so the model learns from the hardest sparse subsets.

## Architecture

Add `UncertaintyGuidedViewDropoutV51` in `motionflow_mv/data/uncertainty_guided_view_dropout_v51.py`.

- **Inputs**: per-view reliability `r_v ∈ [0,1]` from v50 SEFH, or a reprojection-residual fallback.
- **Drop probability**: `p_v = clamp(p_base · (1 - r_v)^α, p_min, p_max)`. With `identity_at_init=True` or during warmup, fall back to the uniform `p_base`.
- **Sampling**: draw a keep mask from `Bernoulli(1 - p_v)`, then enforce `min_views` by keeping the most reliable views and sampling the rest. No gradient flows through the sampling step.
- **Curriculum**: ramp `α` from `0` to `v51_ugvd_alpha` over `v51_ugvd_curriculum_warmup_epochs`.

When disabled, behavior is identical to v46.

## Config flags

| Flag | Type | Default |
|---|---|---|
| `use_v51_uncertainty_guided_view_dropout` | bool | `False` |
| `v51_ugvd_base_dropout` | float | `0.3` |
| `v51_ugvd_min_views` | int | `2` |
| `v51_ugvd_reliability_source` | str | `"v50_sefh"` |
| `v51_ugvd_alpha` | float | `1.0` |
| `v51_ugvd_temperature` | float | `0.5` |
| `v51_ugvd_p_max` | float | `0.6` |
| `v51_ugvd_p_min` | float | `0.0` |
| `v51_ugvd_use_curriculum` | bool | `True` |
| `v51_ugvd_curriculum_warmup_epochs` | int | `1` |
| `v51_ugvd_identity_at_init` | bool | `True` |
| `loss.v51_ugvd_rate_loss_weight` | float | `0.001` |

`v51_ugvd_reliability_source` choices: `"v50_sefh"`, `"reprojection_residual"`, `"uniform"`.

## Loss term

```text
L_uvd = λ_rate · ( mean_v(p_v) - p_base )²
```

This anchors the mean dropout rate to `p_base`, preventing collapse to all-low or all-high dropout.

## Evaluation metric

- Primary: `MPJPE@k` for `k = 2, 3, 4, full` via `experiments/eval_variable_views.py`.
- Diagnostic: `Spearman(dropout probability, reprojection residual)` plus per-residual-bucket drop rate.

## Expected MPJPE impact

| Metric | Expected change |
|---|---|
| `MPJPE@2` | −2 to −4 mm |
| `MPJPE@3` | −1 to −2 mm |
| `MPJPE@4` | −0.5 to −1 mm |
| `MPJPE@full` | ±0.5 mm |

Gains are largest in the sparse-view regime.

## Main risk and mitigation

**Risk**: An uncalibrated reliability signal may drop informative views or keep noisy ones, regressing sparse-view accuracy.

**Mitigation**:
1. `identity_at_init=True` plus curriculum start as proven v46 uniform dropout.
2. `p_max` and `min_views` cap per-view removal.
3. Rate loss anchors the mean drop rate to `p_base`.
4. Smoke with `v51_ugvd_reliability_source="uniform"` first.

## Paper fit

v51 extends the v37/v39 self-critique and v50 self-evolution loop by letting the model drop the views it trusts least, training directly on the hardest sparse subsets. This hardens sparse-view and cross-domain robustness while preserving the strong full-view baseline.
