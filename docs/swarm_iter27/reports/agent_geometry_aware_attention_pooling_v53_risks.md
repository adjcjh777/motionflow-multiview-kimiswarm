# v53 Geometry-Aware Attention Pooling — Risk Register

## 1. Geometry embedding is unstable under noisy calibration

**Risk:** The GAAP module computes ray directions, camera centres, baseline length, and epipolar distances from `K, R, t`. If calibration is noisy or a view has degenerate intrinsics, the geometry embedding can blow up (`NaN/Inf`), corrupting the attention scores and degrading `val_MPJPE`.

**Mitigation:**
- Clamp all geometry quantities to finite ranges (`[-10, 10]` after log scaling).
- Use `torch.nan_to_num` on the geometry embedding before the MLP.
- Normalize the embedding with a running mean/std computed on the first training batch and updated with EMA.
- Add an unit test in `tests/test_geometry_aware_attention_pooling_v53.py` that feeds synthetic degenerate cameras and checks the module still returns `pred == pred_init` (identity) and no NaNs.

## 2. Cross-view attention adds `O(V^2)` memory and latency

**Risk:** For `V=8` views and multi-head attention over `(B*T*J)` tokens, the attention matrix is `(B*T*J, V, V)`. With large batch and clip length this can increase peak memory, causing OOM on the RTX 4090 smoke run.

**Mitigation:**
- Implement the attention as a view-only (not time) cross-product, so memory scales with `V^2` not `(T*V)^2`.
- Default to `v53_gaap_n_layers=1` and `v53_gaap_n_heads=4`.
- If smoke OOMs, add an optional low-rank attention mode (`v53_gaap_low_rank=True`) that projects keys/values to `64` dims and computes attention via matrix multiply over a reduced head dimension.
- Profile peak memory with `torch.cuda.max_memory_allocated()` during smoke.

## 3. Module overfits and overrides the v52 warm start

**Risk:** Even with zero-initialized output projections, the geometry MLP and attention may still learn small non-zero updates during the first epoch of a small smoke run, shifting `val_MPJPE` by more than the 0.1 mm warm-start tolerance and hurting reproducibility.

**Mitigation:**
- Gate the entire GAAP update with a scalar `γ` initialized to `0.0` and only expose it after `v53_gaap_warmup_epochs`.
- Add a warm-start unit test: load a v52 checkpoint, enable v53, run one forward pass, assert `|pred - pred_init| < 1e-4` and `gaap_loss == 0` (because the target equals the uniform attention).
- Start the smoke run with `v53_gaap_loss_weight=0.0` and only enable the auxiliary loss once the smoke MPJPE is stable.

## 4. Interaction with v48 domain-adapted features

**Risk:** v48 injects domain-conditional FiLM/conditional BN into the ST-transformer tokens. If the geometry embedding is computed from camera parameters that do not carry domain information, GAAP may learn domain-specific attention patterns that fail to generalize from H36M/MPI to 3DPW.

**Mitigation:**
- Concatenate the optional `domain_id` into the geometry MLP input as a learned domain embedding, so the attention can adapt per domain.
- Run the smoke on a mixed-domain mini-batch and check per-domain `MPJPE@full`; if 3DPW degrades relative to v52, disable `v53_gaap_use_epipolar_bias` and rely only on ray geometry.
- Keep the module optional and off by default in domain-generalization configs until the full A800 numbers prove it helps.

## 5. Auxiliary loss dominates the geometry loss budget

**Risk:** `L_gaap` pulls the attention distribution toward v52 weights. If `v53_gaap_loss_weight` is too large, it can override the epipolar consistency loss `epi_loss` and the v52 UWT auxiliary loss, causing the optimizer to ignore other training signals.

**Mitigation:**
- Default `v53_gaap_loss_weight=0.01` and scale it by the number of active joints to keep the gradient magnitude below 1% of the total loss.
- Monitor `gaap_loss / epi_loss` ratio in the smoke logs; if it exceeds 0.05, lower the weight.
- Make `v53_gaap_warmup_epochs>=1` the default so the loss only activates after the v52 baseline is stable.
