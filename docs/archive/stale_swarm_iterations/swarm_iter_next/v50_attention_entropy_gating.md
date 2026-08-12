# v50: Attention Entropy Gating (AEG)

## Architecture

`AttentionEntropyGatingV50` adds a lightweight entropy head on top of the multi-view/joint cross-attention maps produced by the v25/v45 geometry-fusion blocks and the v46 sparse-view reliability path. For each attention map, we compute the per-sample Shannon entropy across the attended dimension (view or joint), normalize it by the logarithm of the number of attended positions, and feed the resulting per-(view, joint) entropy scalar through a tiny two-layer MLP. The output is a soft attention-reliability gate that is multiplied against the residual update before it is added to the triangulation-based pose. The gate is fused with the existing v37 self-critique reliability score and the v46 reliability head via a product-and-rescale rule, so when attention is diffuse (high entropy) the residual is suppressed, and when attention is peaked (low entropy) the residual is allowed to pass. The module is identity-at-init: the entropy-to-gate MLP is initialized to produce ones, and a residual connection around the gated block preserves the baseline pose when the gate is disabled.

## Config Flags

```yaml
use_v50_attention_entropy_gating: false        # master switch
v50_aeg_hidden: 64                             # MLP hidden dim
v50_aeg_num_heads: 4                           # heads for which to log attention
v50_aeg_entropy_weight: 0.01                   # entropy regularization loss weight
v50_aeg_gate_temperature: 1.0                  # softmax temperature on attention
v50_aeg_min_entropy_clip: 0.01                 # floor to avoid log(0)
v50_aeg_max_entropy_penalty: 0.1               # cap on entropy loss magnitude
v50_aeg_fusion_mode: "product"                 # "product" | "sum" with v37/v46 reliability
```

## Loss Term

`loss_v50_aeg_entropy = v50_aeg_entropy_weight * mean(entropy * residual_confidence)` where `residual_confidence` is the reprojection residual normalized to [0,1] so that high entropy is penalized only where the model also thinks it is confident. The loss is clamped to `v50_aeg_max_entropy_penalty`. The gate itself is not directly supervised; it is learned end-to-end through the MPJPE loss and the entropy regularizer.

## Evaluation Metric

Primary: `MPJPE@k` for `k = 2,3,4,full` from `experiments/eval_variable_views.py`. Secondary: Pearson correlation between per-joint entropy and per-joint reprojection error; target `r > 0.25`.

## Expected MPJPE Impact

On the v46-SVG smoke baseline (epoch-1 val_MPJPE 32.97 mm), adding AEG is expected to improve sparse-view robustness by `MPJPE@2 -2 to -3 mm` and `MPJPE@3 -1 to -2 mm`, while full-view `MPJPE@full` stays within `±0.5 mm` of the baseline. The largest gains are anticipated on sequences with occluded or noisy views where diffuse attention currently leaks into unreliable views.

## Main Risk / Mitigations

**Risk:** The gate learns to suppress all attention (high-entropy everywhere) and the residual path collapses to zero, erasing the v25/v45 triangulation signal.

**Mitigations:** (1) Identity-at-init plus a residual connection so the default forward pass is unchanged. (2) Clamp the gate to `[0.1, 1.0]` to prevent zero gates. (3) Apply the entropy loss only inside the regularizer, not on the gate itself. (4) Warm-start: freeze the AEG head for the first 500 steps so the baseline pose is stable before the gate is learned.
