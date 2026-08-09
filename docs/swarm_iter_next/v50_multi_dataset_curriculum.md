# v50 Cross-Dataset Difficulty Curriculum (CD-Curriculum)

## Module overview

The v50 **Cross-Dataset Difficulty Curriculum** treats the WebBridge / H36M / MPI / 3DPW mix as a curriculum-learning problem rather than a static weighted average. It maintains online per-domain difficulty scores, progressively increases the exposure of harder domains as training progresses, and dynamically reweights the v41 per-domain MSE loss. The goal is to close the multi-domain gap (the spread between best- and worst-domain val_MPJPE) without regressing full-view accuracy on the cleanest studio data.

## Architecture

A small stateful scheduler lives in the training loop (`experiments/train_omniview_fusion_v5_webbridge_multi.py`). Every `v50_mdc_update_every_n_steps` steps it reads the per-domain MSE from the current batch and an optional per-domain validation MPJPE buffer, computes a smoothed difficulty score `d_i = moving_average(loss_i / loss_ref)`, and maps it to sampling/loss weights `w_i  softmax(temperature * d_i + bonus_hard)`. A linear warmup schedule is applied over `v50_mdc_warmup_epochs` to avoid immediately drowning the model in noisy WebBridge/3DPW examples. The module is identity-at-init (uniform weights) and only reweights existing losses; no new network parameters are introduced.

## New config flags

| Flag | Default | Description |
|---|---|---|
| `use_v50_multi_dataset_curriculum` | `False` | Enable the cross-dataset curriculum scheduler. |
| `v50_mdc_warmup_epochs` | `1` | Epochs over which weights stay near uniform before the curriculum activates. |
| `v50_mdc_schedule_epochs` | `5` | Total epochs over which hard-domain weights is ramped up. |
| `v50_mdc_min_domain_weight` | `0.10` | Floor on any domain's loss/sampling weight to prevent domain collapse. |
| `v50_mdc_update_every_n_steps` | `100` | Steps between online difficulty score updates. |
| `v50_mdc_temperature` | `1.0` | Softmax temperature controlling how aggressively to favor hard domains. |
| `v50_mdc_use_val_feedback` | `True` | Whether to fold per-domain validation MPJPE into difficulty scores. |
| `v50_mdc_hard_domain_bonus` | `0.20` | Extra weight bias for domains with historically high error. |

## Loss / objective

No new loss term is required. The module reweights the existing v41 per-domain MSE: `L_total = Σ_i w_i(t) · L_i`, where `w_i(t)` are the curriculum-produced domain weights at step `t`. The weights are normalized so that `Σ_i w_i = number_of_domains` to preserve total gradient scale on average.

## Evaluation metric

Primary metrics: full-view `val_MPJPE`; per-domain `val_MPJPE` for WebBridge/H36M/MPI/3DPW; `domain_gap = max_i MPJPE_i − min_i MPJPE_i`; and `MPJPE@2/3/4` from `experiments/eval_variable_views.py`.

## Expected impact

- Full-view `val_MPJPE`: **−0.8 to −1.5 mm** on the mixed manifest.
- Domain gap: **−2 to −3 mm** by preventing the model from ignoring hard domains.
- Worst-domain MPJPE (typically WebBridge or 3DPW actual): **−2 to −4 mm**.
- `MPJPE@2/3`: modest gain of **−1 to −2 mm** because harder domains are often those with sparse or noisy views.

## Main risks and mitigations

- **Oscillating weights destabilize training.** Mitigation: momentum-smoothed difficulty scores and updates only every `N` steps.
- **Collapse to easy domains.** Mitigation: `v50_mdc_min_domain_weight` floor plus the explicit `hard_domain_bonus`.
- **Validation feedback is noisy during smoke runs.** Mitigation: fall back to training-loss-only difficulty when validation per-domain estimates are unavailable; disable `use_val_feedback` for smoke configs.
- **Curriculum competes with v48 domain adapter.** Mitigation: keep v48 enabled as the domain-invariant feature branch and treat v50 as a data/loss-weighting wrapper; ablate with and without v48.
