# v53 Bone-Length-Aware Fusion: Risk Register

**Module:** `bone_length_aware_fusion_v53`  
**Author:** design-swarm agent  
**Related proposal:** `docs/swarm_iter27/proposals/v53_bone_length_aware_fusion_v53.md`

---

## 1. Warm-start leakage

**Risk:** Despite zero-initialising the final layers and setting the residual gate to `0.0`, the bone-length transformer, bone-type embedding, or the GNN may introduce small non-identity biases (e.g. layer-normalisation statistics, positional embeddings) so that loading a v52 checkpoint with v53 enabled changes `val_MPJPE` by more than the `0.1 mm` tolerance.

**Mitigation:**
- Use only zero-initialised output projections for `MLP_length`, `MLP_gate`, and the `GNN_skeleton`.
- Keep `v53_bone_residual_gate` as a scalar `nn.Parameter` initialised to `0.0` and clamped to `[0, 1]` only when explicitly scheduled.
- Add a deterministic unit test that runs a fixed batch through the module with `v53_bone_residual_gate=0.0` and asserts `||pred_3d' - pred_3d||_∞ < 1e-5`.

---

## 2. Canonical-skeleton collapse

**Risk:** The learnable bone-length prior means `μ_j` could collapse to a single canonical skeleton. If the variance `σ_j^2` is initialised too small, the model may over-regularise all subjects toward an same skeleton; if too large, the prior becomes ineffective.

**Mitigation:**
- Initialise `σ_j^2` from the empirical bone-length variance of H36M training subjects.
- Use a soft-constraint loss rather than a hard projection: penalise only `0.5 * (l'_j - μ_j)^2 / σ_j^2`.
- Add a small entropy bonus on `log σ_j^2` to prevent collapse to near-zero variance.

---

## 3. Conflict with v28/v40 physical-space alignment

**Risk:** v53 enforces bone-length consistency, while v28/v40 physical-space alignment already penalises bone-length, symmetry, floor, and collision constraints. The two modules may fight each other or double-count the bone-length signal, leading to oscillating losses or degraded pose quality.

**Mitigation:**
- Start v53 training from a v28/v40 checkpoint and keep `v53_bone_loss_weight` low (`0.005–0.01`).
- Only enable v53 after `v53_bone_loss_warmup_epochs` (default `0`) so that the physical loss has already stabilised.
- Monitor the ratio of bone-loss gradient norm to total loss gradient norm and clip if it exceeds `0.1`.

---

## 4. Cross-dataset skeleton mismatch

**Risk:** H36M (17 joints), MPI-INF-3DHP (28 joints), WebBridge, and 3DPW may use different skeleton topologies and parent arrays. A single `parents` argument hard-wired for one skeleton will silently produce wrong bone vectors on another dataset.

**Mitigation:**
- Always pass the runtime `parents` tensor from the data loader / model caller rather than hard-coding it in the module.
- Add an assertion that `parents.max() < J` and that no cycles exist.
- Provide dataset-specific parent maps (e.g. `H36M_17_PARENTS`, `MPI_INF_3DHP_28_PARENTS`) and select at runtime based on the domain label.

---

## 5. Sparse-view over-constraint

**Risk:** With only 2–3 views, joint visibility is low and many bones may be marked invalid. Enforcing bone-length consistency on sparse or noisy observations could pull the pose toward the prior and away from the true 2-D evidence, hurting `MPJPE@2`.

**Mitigation:**
- Only apply the correction to bones where both endpoints have visibility in at least `v53_bone_min_visible_views` views (default `2`).
- Weight each bone by the product of endpoint uncertainties from v52 log-precision, so uncertain bones contribute less to the loss.
- Include an ablation flag `v53_bone_use_uncertainty` (default `True`) that falls back to uniform weighting if needed.
