# v50-3DPW: Self-Evolving Domain Adapter for 3DPW Actual Mode

## Architecture

v50-3DPW extends the v48 domain-conditional pipeline with a **test-time self-evolving domain adapter (SEDA)** that is activated only when the input is tagged as *unknown/in-the-wild* (i.e. 3DPW actual mode). The module sits after the v48 domain adapter and before the final pose head. It runs a small, frozen-at-init refinement block that is updated on-the-fly using self-supervised losses computed from multi-view geometry, without access to ground-truth 3D. Concretely, it predicts per-joint residual corrections and per-view reliability offsets from the v48 output, then performs one or two iterative re-triangulation steps. The block is a two-layer MLP with residual connections and layer normalization, sharing the same hidden dimension as v48 (`v48_dg_hidden`).

## New config flags

- `use_v50_3dpw_seda` (default `False`): enable the 3DPW self-evolving domain adapter.
- `v50_seda_hidden` (default `64`): hidden size of the refinement MLP.
- `v50_seda_n_steps` (default `2`, max `3`): number of self-evolution iterations per batch during training/eval.
- `v50_seda_lr` (default `1e-4`): learning rate for test-time parameter updates.
- `v50_seda_use_ema` (default `True`): maintain an EMA copy of the adapter weights to prevent drift.
- `v50_seda_freeze_first_epoch` (default `True`): keep the adapter frozen for the first epoch so it learns a stable identity prior.

## Loss term

A single composite loss is added only for 3DPW actual-mode samples (or their synthetic proxy):

\[
\mathcal{L}_{\text{v50}} = \lambda_{\text{reproj}} \cdot \mathcal{L}_{\text{reproj}} + \lambda_{\text{bone}} \cdot \mathcal{L}_{\text{bone}} + \lambda_{\text{domain}} \cdot \mathcal{L}_{\text{adv}}
\]

where `λ_reproj=1.0`, `λ_bone=0.1`, and `λ_domain=0.01` are defaults. `L_reproj` is the mean squared 2D reprojection residual across all visible views, `L_bone` is the skeleton bone-length variance relative to the H36M prior, and `L_adv` is a gradient-reversal domain-confusion term aligned with v48. During training the adapter is updated with a small number of gradient steps; during evaluation the same steps are applied but weights are reset per sequence to avoid cross-sequence leakage.

## Evaluation metric

Primary: **MPJPE@k on 3DPW actual-mode** for `k = 2,3,4` and full views, measured by `experiments/eval_variable_views.py`. Secondary: per-domain `MPJPE` on WebBridge/H36M/MPI val to confirm no regression on known domains. Tertiary: Spearman correlation between adapter-predicted reliability and reprojection residuals.

## Expected MPJPE impact

Based on the v48 domain-adapter smoke and the residual-reliability trend seen in v37/v39, closing the 3DPW actual-mode domain gap with geometric self-supervision should lower **MPJPE@3 on 3DPW actual by 4–6 mm** and **MPJPE@2 by 5–8 mm**, while keeping full-view WebBridge/H36M val within 0.5 mm of the v48 baseline. If the 3DPW actual gap is currently ~15–20 mm over the studio baseline, v50 targets reducing that to under 10 mm.

## Main risk / mitigations

- **Risk: Test-time adaptation destabilizes training or leaks across batches.** Mitigation: reset adapter state per sequence; use EMA; cap gradient steps at `v50_seda_n_steps`; freeze for the first epoch.
- **Risk: The adapter collapses to identity and provides no gain.** Mitigation: auxiliary `L_reproj` and `L_bone` are strong geometric priors; monitor residual-reliability Spearman and abort if it stays below 0.2.
- **Risk: 3DPW actual-mode labels are sparse, making smoke validation noisy.** Mitigation: run the v50 smoke on the existing v48 3DPW subset with label-withheld self-supervision only; compare to v48 oracle numbers to bound the possible gain before committing A800 cycles.