# v52 Domain-Conditional Normalization — Risks and Mitigations

## 1. Redundancy with v48 domain-conditional FiLM

**Risk:** v48 already applies domain-conditional affine shifts via FiLM/conditional BN on the same `(B, T, V, J, d)` feature tensor. Adding v52 on top may duplicate the adaptation, increasing parameters without improving accuracy.

**Mitigation:**
- Make v52 group-wise (default `v52_dcn_num_groups=4`) rather than full FiLM, so it modulates channel subspaces rather than every channel independently.
- Run an ablation where v52 is stacked on a *disabled* v48 baseline first; if it replicates most of v48's gain, the two modules can be merged in a future iteration.
- Keep `v52_dcn_identity_init=True` so the default behavior is no-op and only the gradient decides whether the module is useful.

## 2. Instability from conditional normalization

**Risk:** Learning affine parameters per domain can overfit small domains (e.g., AIST++ or 3DPW actual) or explode when combined with LayerNorm, especially if `v52_dcn_num_groups` is too small and the MLP has many layers.

**Mitigation:**
- Clamp the predicted scale to `(1 + γ)` with `γ` initialized to zero and constrained to `[-0.5, 0.5]` via `tanh` clipping.
- Use a shallow 2-layer MLP with dropout (`v52_dcn_dropout=0.1`) and avoid batch normalization inside the domain MLP.
- Start the smoke test with `v52_dcn_num_groups=8` or higher (fewer parameters per group) and only reduce if the ablation shows benefit.

## 3. View-count conditioning noise for short clips

**Risk:** `v52_dcn_use_view_count_conditioning` appends the active view count, which is noisy or constant when `T` is small (smoke configs often use `T=3` or `T=5`). The extra input may then harm rather than help sparse-view generalization.

**Mitigation:**
- Pool the view count over the temporal dimension before feeding it to the domain MLP.
- Gate the view-count embedding with a scalar weight initialized near zero so it warms up gradually.
- Smoke-test with `v52_dcn_use_view_count_conditioning=False` first; only enable if it improves `MPJPE@2/3`.

## 4. Interaction with v50/v51 auxiliary heads

**Risk:** v52 changes the distribution of tokens entering the ST transformer. The downstream residual MLP, v50 SEFH, and v51 CDSVR were trained on the previous distribution and may need re-tuning or re-warmstarting.

**Mitigation:**
- When enabling v52 on an existing v50/v51 checkpoint, freeze all upstream modules and the v50/v51 heads for the first epoch; train only v52.
- Include a compatibility test that loads a v51 checkpoint, enables v52, and checks that the first forward pass is numerically identical (identity-at-init).
- Monitor the v50 SEFH auxiliary loss in the smoke test; if it spikes, scale it down or freeze the head.

## 5. Regression on single-domain benchmarks

**Risk:** The extra domain-conditional capacity may slightly hurt H36M-only or MPI-only performance if the model starts to overfit to the multi-domain mixture or if the added MLP introduces optimization noise.

**Mitigation:**
- Always compare the v52 smoke run against the parent baseline on the *same* seed and manifest.
- Use a small `v52_dcn_hidden` (32 or 64) and a high identity-init bias; treat v52 as a residual refinement of LayerNorm rather than a replacement.
- If a single-domain regression is observed, add an additional domain label for "single-domain mode" or allow `domain_id=None` to fall back to a shared default normalization.
