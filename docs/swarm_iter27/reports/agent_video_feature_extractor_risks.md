# v53 Video Feature Extractor — Risk Register

## Risk 1: Temporal conv overfits on short training clips

- **Likelihood:** Medium
- **Impact:** The causal temporal encoder memorizes clip-level patterns rather than generalizing across sequence boundaries, leading to higher `val_MPJPE` than the v52 baseline.
- **Mitigation:** Keep the temporal kernel small (`v53_vfe_kernel_size=3`) and the depth shallow (`v53_vfe_n_layers=2`). Add dropout (`v53_vfe_mixer_dropout=0.1`) and use the identity gate so the model can fall back to per-frame features early in training. Ablation: run smoke with the mixer disabled (`v53_vfe_use_spatial_mixer=false`) to isolate the temporal-only gain.

## Risk 2: Added compute cost before the v52 UWT head

- **Likelihood:** High
- **Impact:** The extra temporal conv and joint/view mixer increase memory consumption and wall-clock time, especially for full A800 runs with large `T` (e.g., `clip_len=13` or `25`).
- **Mitigation:** Implement the temporal encoder as grouped causal convolutions (`groups=d`) so the per-sample cost is small. Make the spatial mixer optional via `v53_vfe_use_spatial_mixer`. Benchmark a single forward pass during smoke and set a hard stop if throughput drops more than 10% relative to v52.

## Risk 3: Identity gate collapses to zero and never trains

- **Likelihood:** Medium
- **Impact:** Because the gate is initialized to zero and the residual path is zero-initialized, the optimizer may leave the gate near zero, rendering v53 a no-op and wasting the module.
- **Mitigation:** Initialize the gate to a small positive value (`v53_vfe_gate_init=1e-2`) in the first A800 ablation, or add a tiny auxiliary loss that encourages non-zero gate usage (e.g., L2 penalty on `(1 - g)`). Monitor the gate magnitude in logs and warn if it remains below `1e-3` after one epoch.

## Risk 4: Causal padding creates boundary artifacts at clip edges

- **Likelihood:** Medium
- **Impact:** The first few frames of each clip receive zero-padded temporal context, which can degrade triangulation quality near the clip start and propagate errors into downstream temporal heads (v47/v49-Lite).
- **Mitigation:** Use replication padding instead of zero padding for the causal conv. When `T` is small, prefer `v53_vfe_kernel_size=3` over larger kernels to reduce the receptive field at the clip boundary. Evaluate per-frame MPJPE variance to detect edge artifacts.

## Risk 5: Interaction with v52 UWT consistency/entropy losses becomes unstable

- **Likelihood:** Low–Medium
- **Impact:** v53 changes the feature distribution feeding into v52 UWT, which relies on `feature_bias` (mean/std over views) and reprojection residuals. If the refined features shift too far, the pre-trained v52 MLP dynamics may destabilize, causing NaN/Inf in the UWT consistency loss.
- **Mitigation:** Keep the output projection zero-initialized and the gate initialized to zero, so v53 starts as an identity transform. Use `v52_uwt_warmup_epochs` to delay the UWT loss until after v53 has produced stable features. Clip the refined features to `[-10, 10]` after the residual addition if needed.
