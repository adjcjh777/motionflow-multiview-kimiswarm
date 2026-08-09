# v52 Video Feature Extractor — Risk Report

## Risk 1: Factorized attention has cubic memory in T, V, and J

**Description:** The three factorized branches each compute attention along one axis of the `(B, T, V, J, d)` tensor. Even though each branch factorizes the full 5D attention, the temporal branch still scales as `O(B * T^2 * V * J * d_model / n_heads)` and the cross-view branch as `O(B * T * V^2 * J * d_model / n_heads)`. With `T = 27`, `V = 8`, `J = 17`, and `d_model = 64`, the attention maps can exceed RTX 4090 memory during smoke testing.

**Impact:** OOM during smoke on RTX 4090 or reduced batch size, slowing iteration.

**Mitigation:**
- Default `v52_video_feat_d_model=64` and limit to `v52_video_feat_n_heads=4`.
- Restrict the temporal branch to a local window (e.g., `±7` frames) instead of full self-attention.
- Add an optional gradient-checkpointing flag `v52_video_feat_checkpoint_attn`.
- Smoke with `clip_len=9` first to validate memory before larger `T`.

---

## Risk 2: Identity initialization is broken by non-zero bias or projection

**Description:** The module is intended to be a no-op at initialization. If the input/output projection has a non-zero bias, the residual path introduces a constant offset, and the module no longer preserves a v25/v46 baseline. Similarly, if `alpha` is not strictly zero at step 0, early training is affected.

**Impact:** Regression on the baseline MPJPE even though the flag is supposed to be warm-start friendly.

**Mitigation:**
- Initialize all linear projections in the residual branch to zero (weights and bias).
- Initialize `alpha = 0.0` and enforce `v52_video_feat_warmup_steps >= 100`.
- Add a unit test in `tests/test_video_feature_extractor_v52.py` asserting that `VideoFeatureExtractorV52(feat) == feat` at init with `alpha=0`.

---

## Risk 3: Branch gating collapses to a single branch

**Description:** The adaptive branch fusion uses a softmax over three branch weights. If one branch (e.g., temporal) is consistently easiest to exploit, the gate may collapse to a near-one-hot vector and the other branches become dormant. This undermines the goal of factorized video feature extraction.

**Impact:** The module effectively reduces to a single attention branch and fails to capture multi-view geometry or skeleton kinematics.

**Mitigation:**
- Add an entropy regularizer on the gate distribution with a small weight (e.g., `v52_video_feat_gate_entropy_weight=0.01`).
- Alternatively, use a fixed initial branch schedule that forces each branch to be active for the first `N` warmup steps.
- Log per-branch gate histograms in smoke/eval to detect collapse.

---

## Risk 4: Redundancy with v47/v49 temporal and v34/v36 graph modules

**Description:** v47 and v49 already refine poses temporally after triangulation, and v34/v36 operate on per-frame view-joint graphs. v52 extracts temporal and skeleton features at the feature level before triangulation. The three mechanisms may overlap, and stacking all of them may overfit or cause conflicting gradients.

**Impact:** No MPJPE gain or degraded generalization due to over-parameterization.

**Mitigation:**
- Smoke v52 first on top of the bare v25/v46 baseline without v47/v49/v34/v36 enabled.
- Ablation: disable v52 and re-enable v47/v49 to quantify redundancy.
- If redundancy is high, repurpose v52 as a pre-processor only, replacing the per-frame feature path rather than augmenting it.

---

## Risk 5: Skeleton graph attention is dataset-dependent

**Description:** The skeleton branch uses a fixed joint parent graph (`H36M_17_PARENTS` or `MPI_INF_3DHP_28_PARENTS`). If the batch contains mixed skeleton formats or if new datasets with different joint orders are added, the graph edges may be incorrect and propagate errors.

**Impact:** Degraded MPJPE on datasets with a different skeleton topology, or silent shape mismatches during mixed-batch training.

**Mitigation:**
- Pass the skeleton parents explicitly through `OmniMultiViewFusionV5.forward` from the data loader or a model attribute.
- Default the graph branch to a learned fully-connected attention (like the other branches) when `skeleton_parents` is `None`.
- Add a configuration flag `v52_video_feat_use_skeleton_graph` that can be disabled for cross-dataset smoke tests.
