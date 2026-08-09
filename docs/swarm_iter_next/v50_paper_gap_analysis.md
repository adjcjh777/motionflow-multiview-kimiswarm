# v50: Cross-Domain Sparse-View Reliability Gap Closer (v50-CDSVG)

## Architecture

`CrossDomainSparseViewReliabilityGapCloserV50` sits immediately after the v48 domain adapter (or after v47 temporal aggregation when v48 is disabled). Its purpose is to close a specific hole in the v49 paper story: current sparse-view methods learn per-view reliability, but they do not explicitly model how *domain shift* interacts with *view dropout*, so out-of-domain sequences suffer disproportionately when only 2–3 views are available. The module takes the v46 per-view reliability scores, the current domain embedding from v48, and per-view reprojection residuals, and passes them through a small two-layer cross-attention block. Domain embeddings attend to view embeddings; the output is a domain-conditioned per-view reliability offset and a per-joint log-variance uncertainty scale. The final reliability is the product of the v46 reliability and the predicted offset, and the uncertainty scale is applied to the supervised MPJPE loss. The module is identity-at-init: the offset MLP is initialized to zero and the log-variance to zero, so enabling the flag does not perturb the already-trained v47/v48 baseline.

## New config flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v50_cross_domain_sparse_view_reliability` | bool | `False` | Master switch. |
| `v50_cdsvg_hidden` | int | `64` | Hidden dim of the domain×view attention block. |
| `v50_cdsvg_num_heads` | int | `4` | Attention heads in the domain×view block. |
| `v50_cdsvg_loss_weight` | float | `0.01` | Weight λ of the auxiliary reliability/uncertainty loss. |
| `v50_cdsvg_offset_min` | float | `0.05` | Floor on the reliability offset to avoid zero views. |
| `v50_cdsvg_use_domain_label` | bool | `True` | Use v48 domain embedding; if False, fall back to an unsupervised domain cluster. |
| `v50_cdsvg_uncertainty_temperature` | float | `1.0` | Temperature on the predicted log-variance. |

## Loss term

For a clip with `V` visible views, predicted per-view reliability offsets `w_v`, reprojection residuals `r_v`, and predicted per-joint log-variance `σ_j`, the auxiliary loss is:

```
L_cdsvg = λ · [ (1/V) Σ_v w_v · Huber(||r_v||, δ) + (1/J) Σ_j exp(-σ_j) · e_j + γ · Var(w_v) ]
```

- `e_j` is the per-joint 3-D error.
- The `Var(w_v)` term prevents collapse to a uniform reliability map.
- Default `λ = 0.01`; `δ = 50` mm; `γ = 0.1`.

## Evaluation metric

Primary metrics are `MPJPE@k` for `k = 2, 3, 4, full` reported by `experiments/eval_variable_views.py`. Secondary metrics target the paper gap directly: per-domain `MPJPE@k` on WebBridge, H36M, MPI, and 3DPW actual; Spearman correlation between predicted per-view reliability offset and the corresponding reprojection residual (target `> 0.35`); and an expected-calibration-error (ECE) style score measuring how well the predicted log-variance matches the actual per-joint MPJPE.

## Expected MPJPE impact

Given the current v46-SVG smoke epoch-1 `val_MPJPE = 32.97 mm` on the in-domain mix, the cross-domain sparse-view regime is the place where the paper currently lacks strong numbers. Closing the domain×view interaction gap should improve **3DPW actual `MPJPE@2` by 5–7 mm** and **`MPJPE@3` by 3–4 mm**, while keeping in-domain `MPJPE@full` within ±0.5 mm of the v48 baseline. If the 3DPW actual full-view gap is currently ~15–20 mm over the studio baseline, v50-CDSVG targets cutting the sparse-view portion of that gap by roughly one third.

## Main risk / mitigations

| Risk | Mitigation |
|------|------------|
| **3DPW actual domain labels are sparse or missing.** | Fall back to the v48 domain classifier embedding, or to an unsupervised `v50_cdsvg_use_domain_label=False` cluster when labels are absent. |
| **Reliability offsets collapse to a uniform map.** | The `Var(w_v)` regularizer plus identity-at-init initialization keeps the baseline stable; clamp offsets to `> 0.05`. |
| **Confounds with v48 domain adapter and v46 reliability head.** | Treat v50-CDSVG as an optional add-on after v48; add a config check that raises if `use_v48_domain_generalization=False` and `v50_cdsvg_use_domain_label=True`. |
| **Auxiliary loss competes with v40 physical loss or v48 domain-adversarial loss.** | Linear warmup over one epoch and start ablations at `λ=0.001` before committing the default `0.01`. |

## Next action

Create `motionflow_mv/fusion/cross_domain_sparse_view_reliability_v50.py`, add the flags to `motionflow_mv/fusion/omniview_fusion_v5.py`, wire the loss into `experiments/train_omniview_fusion_v5_webbridge_multi.py`, and smoke-test with `configs/benchmark_v50_cdsvg_smoke.yaml`, warm-starting from the best available v48 checkpoint.
