# v51 Skeleton-Aware Uncertainty Gating

## Module

**`SkeletonAwareUncertaintyGateV51`** – a skeleton-structured regularization module for the per-joint aleatoric uncertainty produced by the v50 Self-Evolution Feedback Head (SEFH).  The v50 SEFH already predicts per-joint log-variance from reprojection, temporal, and epipolar residuals, but those estimates are made independently per joint and can be poorly calibrated when views are dropped or the subject is out-of-domain.  `SkeletonAwareUncertaintyGateV51` forces the uncertainty field to respect the human kinematic graph: uncertainty should propagate along bones, and symmetric limbs should agree on their reliability.

### Architecture description

The module takes as input the current 3-D pose estimate `X ∈ R^(J×3)` and the per-joint log-variance `s ∈ R^J` from v50 SEFH.  It builds a fixed skeleton graph with two edge types: **bone** edges (parent → child) and **symmetry** edges (left ↔ right mirror pairs).  For each joint, a tiny edge-type-conditional GNN computes messages from its skeleton neighbors:

```
m_i = Σ_{j∈N(i)} MLP_e([s_i, s_j, log(1 + ||x_i - x_j||), b_ij])
```

where `b_ij` is a learned edge-type embedding and `||x_i - x_j||` is the current bone length.  The refined log-variance is:

```
s_i' = s_i + gate · (W m_i + b)
```

`gate` is a per-joint scalar initialized to zero so the module is a strict no-op at startup.  A separate **skeleton-consistency residual**

```
c_i = Σ_{j∈N(i)} | ||x_i - x_j|| - l_ij^* | / exp(s_i')
```

is also computed, where `l_ij^*` is a learned or data-driven rest bone length.  This residual is fed back to the v50 SEFH as an additional geometric cue, closing the self-evolution loop through the skeleton prior.

### New config flags

| Flag | Type | Default |
|---|---|---|
| `use_v51_skeleton_aware_uncertainty` | bool | `False` |
| `v51_sau_hidden` | int | `64` |
| `v51_sau_num_layers` | int | `2` |
| `v51_sau_edge_types` | list[str] | `["bone", "symmetry"]` |
| `v51_sau_identity_init` | bool | `True` |
| `v51_sau_bone_length_prior` | str | `"learned"` |
| `v51_sau_temperature` | float | `1.0` |
| `loss.v51_sau_nll_weight` | float | `0.01` |
| `loss.v51_sau_consistency_weight` | float | `0.005` |

### Loss term

```
L_sau = loss.v51_sau_nll_weight * NLL(x_gt, x_pred, s')
      + loss.v51_sau_consistency_weight * Σ_i c_i
      + 1e-4 * mean(gate^2)
```

`NLL` is the Gaussian negative log-likelihood using the refined per-joint variance.  The consistency term penalizes bone-length violations weighted inversely by predicted uncertainty, so the model is encouraged to flag uncertain joints as uncertain rather than forcing them into anatomically implausible poses.

### Evaluation metric

Primary: `MPJPE@k` for `k = 2, 3, 4, full` via `experiments/eval_variable_views.py`.
Secondary: uncertainty calibration metrics — expected calibration error (ECE) of the per-joint error against `exp(s')`, and `Spearman(uncertainty, error) > 0.35`.  Per-joint reports for wrists/ankles isolate distal-joint gains.

### Expected MPJPE impact

- `MPJPE@full`: ±0.5 mm (regularization, no regression target).
- `MPJPE@2`: −2 to −4 mm, because sparse-view hallucination of occluded joints is guided by bone/symmetry uncertainty propagation.
- `MPJPE@3`: −1 to −2 mm.
- Calibration improves most for wrists and ankles, where independent per-joint uncertainty from v50 is noisiest.

### Main risk / mitigation

- **Uncertainty over-smoothing.**  The GNN could collapse all joints to a common variance, erasing per-joint signal.  Mitigation: zero-initialized `gate`, residual connection, and a small `v51_sau_temperature`; freeze the gate for the first epoch.
- **Over-constraint on non-standard poses.**  Learned bone priors may hurt highly articulated actions.  Mitigation: use soft consistency loss with learned rest lengths rather than hard constraints, and keep the gate small via L2 regularization.
- **Dependency on v50 SEFH.**  If SEFH is disabled, the module has no input uncertainty to refine.  Mitigation: raise a config error when `use_v51_skeleton_aware_uncertainty=True` but `use_v50_self_evolution_feedback_head=False`; alternatively, fall back to a fixed high-variance initialization.

### Integration and smoke plan

**File**: `motionflow_mv/fusion/skeleton_aware_uncertainty_v51.py`

Insert after `SelfEvolutionFeedbackHeadV50` in `motionflow_mv/fusion/omniview_fusion_v5.py`.  Smoke on `configs/benchmark_v50_self_evolution_feedback_head_smoke.yaml` plus the new flag.  Success gate: `val_MPJPE@full` within 1 mm of v50 baseline, `MPJPE@2` improves by ≥2 mm, and `Spearman(uncertainty, error)` increases by ≥0.05.
