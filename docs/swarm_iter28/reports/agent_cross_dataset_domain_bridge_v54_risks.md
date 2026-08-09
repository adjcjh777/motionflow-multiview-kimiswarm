# Agent Risk Assessment: v54 Cross-Dataset Domain Bridge (CDDB)

## 1. Affine transform can diverge from invertible/identity behavior

**Risk:** The domain-conditional scale `s_d` can drift far from 1.0 or the offset `o_d` can grow large during training, especially on small or outlier domains. This makes the canonical pose space unstable and the inverse denormalization unreliable.

**Mitigation:**
- Parameterize scale as `exp(log s_d)` and initialize the final layer of `MLP_aff` to zero.
- Clip `exp(log s_d)` to `[0.5, 2.0]` during forward passes.
- Add a residual regularization loss (`v54_cddb_residual_weight`) that penalizes large `|s_d - 1|` and `|o_d|`.
- Verify identity-at-init on a v53 checkpoint: `val_MPJPE` should change by ≤ 0.1 mm.

## 2. Canonical refiner overfits to the dominant training domain

**Risk:** If most training samples come from one dataset (e.g., H36M), the skeleton-aware canonical refiner may learn domain-specific pose artifacts rather than a true domain-agnostic prior. This can hurt rare or under-represented domains such as 3DPW or WebBridge actual.

**Mitigation:**
- Use the moment-matching consistency loss (`v54_cddb_consistency_weight`) to force all domains into the same canonical distribution.
- Monitor per-domain `val_MPJPE` and per-domain canonical pose statistics in TensorBoard.
- Provide a fallback flag `v54_cddb_use_canonical_refiner=False` to disable the refiner and keep only the affine normalization, which is less prone to overfitting.

## 3. Conflict with v53 physical-space calibration

**Risk:** v53 already applies floor, bone-length, and reprojection constraints to the 3D pose. CDDB adds another pose-refinement layer that may fight v53, causing oscillations or degraded physical plausibility.

**Mitigation:**
- Make CDDB identity-at-init using `v54_cddb_residual_gate_init=-6.0` so it starts by passing the v53 output through unchanged.
- Keep the loss weight small initially via `v54_cddb_warmup_epochs` and only ramp it after v53 losses have stabilized.
- Re-use v52 UWT weights so that CDDB corrections are weighted by per-view confidence and do not override v53's view-aware calibration.

## 4. Domain labels unavailable at inference or mismatch with v48/v51

**Risk:** At test time, domain labels may be missing, or the domain taxonomy used by CDBB may differ from the one used by v48/v51 (e.g., v48 `num_domains=6`, v51 CDSVR expects the same). This can lead to out-of-bounds embeddings or inconsistent domain conditioning.

**Mitigation:**
- Share the domain vocabulary and `num_domains` across v48, v51, and v54; default `v54_cddb_num_domains` to the same value as `v48_dg_num_domains`.
- Map unknown/missing domain labels to a learned "unknown" embedding (`v54_cddb_unknown_domain_id`) that is trained to behave like identity.
- Validate that all domain IDs in the mixed loader are within `[0, v54_cddb_num_domains - 1]` at the start of each epoch.

## 5. Extra latency and memory in the forward pass

**Risk:** CDDB adds two small MLPs, an embedding table, and a consistency-loss computation. On long clips or large batch sizes this can increase per-step time and peak GPU memory.

**Mitigation:**
- Keep `v54_cddb_hidden=64` and `v54_cddb_n_layers=2` as defaults; the module operates only on the `(B*T, J, 3)` pose tensor, not on high-dimensional view features, so the overhead is small.
- Compute the consistency loss only when `v54_cddb_consistency_weight > 0` and skip it during inference.
- Profile a single forward pass on the smoke config; target <5% increase in step time compared to the v53 baseline.
