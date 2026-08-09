# v50 Skeleton-Aware Joint-Joint Attention Bias

## Module

**`SkeletonAwareAttentionBiasV50`** – a lightweight, identity-at-init attention bias that injects the human kinematic skeleton into the joint-level self-attention used by the v47 temporal aggregation head (and optionally the v34 view-joint graph attention).  Instead of asking the model to learn from scratch that wrists are connected to elbows or that left/right joints are symmetric, we add a learned, edge-type-conditional bias to the standard scaled-dot-product attention scores.

### Architecture description

The bias is added to the attention pre-softmax logits:

```
attention_score = (Q @ K^T) / sqrt(d_k) + gate(head) * skeleton_bias(i, j)
skeleton_bias(i, j) = sum over edge-types e  [ embedding_e[type(e), i, j] ]
```

`type(e)` enumerates the fixed SMPL/H36M edge types: parent-child bone, left-right symmetry pair, and leaf/endpoint status.  Each type has a small lookup table (`v50_saab_embedding_dim=32`) indexed by the ordered joint pair.  A per-head scalar `gate` is initialized to zero so the module is a strict no-op at start and can be learned gradually.  An optional degree normalization divides the bias by the square-root of each joint’s skeleton connectivity to prevent hub joints from dominating.

The module is inserted after the v46 sparse-view reliability head, where joint tokens already carry multi-view geometry.  By operating purely on the `(joint, joint)` attention map, it adds no extra tokens and only a small embedding table (~a few KB), so it is cheap even at A800 full scale.

### New config flags

| Flag | Default | Description |
|---|---|---|
| `use_v50_skeleton_aware_attention_bias` | `False` | Master switch for the module. |
| `v50_saab_apply_to` | `"temporal"` | Where to apply the bias: `"temporal"` (v47 head) or `"graph"` (v34 view-joint graph). |
| `v50_saab_edge_types` | `["bone", "symmetry", "endpoint"]` | Skeleton edge types used to build the bias. |
| `v50_saab_embedding_dim` | `32` | Dimension of each edge-type embedding. |
| `v50_saab_init_gate` | `0.0` | Initial value of the per-head soft gate; zero means identity-at-init. |
| `v50_saab_use_degree_norm` | `True` | Normalize bias by joint connectivity degree. |

### Loss term

No new loss term is required; the skeleton bias acts as a structural prior on attention.  If desired, an optional auxiliary loss can be added:

```
L_saab = v50_saab_reg_weight * mean(gate^2)
```

with `v50_saab_reg_weight=1e-4` to keep the gates small and stable.  This is not expected to change the overall loss landscape materially.

### Evaluation metric

Primary metrics are `val_MPJPE@full` and the sparse-view `MPJPE@2/3/4` from the v46/v47 evaluation protocol.  We will also report per-joint MPJPE for wrists and ankles to isolate the benefit on distal joints that most often suffer in sparse-view settings.

### Expected MPJPE impact

- `val_MPJPE@full`: −0.5 to −1.0 mm improvement by regularizing implausible joint configurations without changing the geometry backbone.
- `MPJPE@2/3`: −2 to −3 mm improvement, driven by better hallucination of occluded joints using bone/symmetry priors.
- Wrist/ankle MPJPE should drop disproportionately because these leaf joints benefit most from the kinematic bias.

These numbers assume the module is stacked on top of a v48-domain baseline and smoke-tested with `d=64`, `clip_len=9`, `train_samples=500`.

### Main risk / mitigations

- **Over-constraint on non-standard poses.** Sports or highly articulated motions may violate normal kinematic priors.  Mitigation: the gate is initialized to zero and learned; we also freeze the bias for the first epoch and only update the gate afterwards.
- **Interference with v47 temporal patterns.** If the temporal head relies on non-skeletal temporal correlations, the bias could suppress them.  Mitigation: apply only to the joint-joint self-attention sub-layer and keep the temporal cross-frame attention untouched.
- **Combinatorial flag explosion with v34 graph path.** Using the same bias in two places could create two learned variants.  Mitigation: if `v50_saab_apply_to="graph"`, share a single `SkeletonAwareAttentionBiasV50` instance with the temporal head; otherwise keep only the temporal path.

### Smoke experiment

`configs/benchmark_v50_skeleton_aware_attention_bias_smoke.yaml`: stack `use_v50_skeleton_aware_attention_bias=True` on the v49-lite or v48-domain smoke config.  Success gate: `val_MPJPE@full` within 1 mm of the baseline, `MPJPE@2` improves by ≥2%, and attention entropy for anatomically connected joints increases relative to the no-bias run.
