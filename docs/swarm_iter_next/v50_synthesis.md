# v50 Design Space Synthesis

Synthesized from 20 sub-agent notes in `docs/swarm_iter_next/v50_*.md`. The project is at the v46 sparse-view generalization milestone (local smoke `val_MPJPE@full ≈ 32.97 mm`; A800 v46 full run in progress), with v47 temporal, v48 domain, and v49-Lite temporal queued. v50 is the first explicitly "self-evolution" design iteration: every proposal is optional, identity-at-init, and targets the sparse-view / cross-domain frontier.

## 1. Executive Summary

v50 sharpens the MotionFlow-MultiView roadmap around a single paper narrative: **making the pipeline critique and refine its own geometry**. The 20 proposals fall into three clusters: (1) triangulation and view-reliability modules that close the uncertainty loop; (2) temporal and domain modules that adapt to longer clips and wild data; and (3) training-curriculum and augmentation ideas that harden sparse-view generalization. Most are warm-startable on the proven v46 backbone, so they can be smoke-tested on the local RTX 4090 without waiting for v47/v48. The highest-priority work is the module that unifies reprojection, temporal, and epipolar residuals into a single self-evolution feedback head, because it directly addresses the ICRA/CVPR 2027 story while carrying the lowest architectural dependency risk.

## 2. Ranking Table of the 20 v50 Proposals

| Proposal | Key idea | Expected MPJPE impact | Risk level | Feasibility |
|---|---|---|---|---|
| Self-Evolution Feedback Head v50 | Close the prediction↔uncertainty loop by predicting per-view reliability and per-joint log-variance from residuals. | MPJPE@2 −2 to −4 mm; MPJPE@3 −1 to −2 mm; full-view ±0.5 mm. | Medium | 5 |
| Outlier View Rejection v2 | Soft geometric + learned per-(view, joint) keep weights with a sparse-aware guard. | MPJPE@2 −3 to −5 mm; MPJPE@3 −1.5 to −2.5 mm. | Medium | 5 |
| Cross-Domain Sparse-View Reliability Gap Closer | Condition v46 reliability on v48 domain embedding to fix domain×view dropout interactions. | 3DPW actual MPJPE@2 −5 to −7 mm; MPJPE@3 −3 to −4 mm. | Medium | 4 |
| Uncertainty-Aware Triangulation v50 | Replace single-pass DLT with a gradient-safe, two-step precision-weighted re-triangulation. | MPJPE@2 −3 to −5 mm; MPJPE@3 −2 to −3 mm. | Medium | 4 |
| Dynamic View Reliability v50 | Inference-time residual update of v46 reliability from reprojection, temporal jump, and epipolar signals. | MPJPE@2 −3 mm; MPJPE@3 −2 mm. | Medium | 4 |
| Skeleton-Aware Attention Bias v50 | Inject kinematic skeleton edge types as a learned, gated attention bias. | MPJPE@2 −2 to −3 mm; distal joints improve most. | Low | 5 |
| Cross-View Attention Refinement v50 | Residual geometry-biased cross-view attention layer before triangulation. | MPJPE@2 −2 to −3 mm; MPJPE@3 −1 to −2 mm. | Medium | 4 |
| Sparse-View Triangulation Agreement Loss v50 | Penalize disagreement between subset triangulations and the fused pose. | MPJPE@2 −2 to −3 mm; full-view ±0.5 mm. | Low | 5 |
| Two-View Extreme Dropout v50 | Train with explicit 2-view subsets and a pairwise view-synergy scorer. | MPJPE@2 −3 to −5 mm. | Medium | 4 |
| Attention Entropy Gating v50 | Suppress diffuse attention via an entropy-to-reliability gate. | MPJPE@2 −2 to −3 mm; MPJPE@3 −1 to −2 mm. | Medium | 4 |
| Multi-Dataset Difficulty Curriculum v50 | Online reweighting of v41 per-domain MSE by curriculum difficulty. | Domain gap −2 to −3 mm; full-view −0.8 to −1.5 mm. | Low | 5 |
| Physical-Space Alignment Refiner v50 | Post-triangulation learned correction gated by bone/floor/symmetry priors. | MPJPE@2 −2 to −4 mm; full-view −0.5 to −1.0 mm. | Medium | 3 |
| Lightweight Geometry BA v50 | Tiny differentiable bundle-adjustment step after triangulation. | MPJPE@2 −2 to −4 mm; full-view −0.5 to −1.5 mm. | Medium-High | 3 |
| Camera Calibration Noise Robustness v50 | Synthetic calibration noise + bounded per-camera correction head. | MPJPE@2 −2 to −4 mm; 3DPW actual −4 to −6 mm. | Medium | 3 |
| View Synthetic Augmentation v50 | Training-only virtual camera views projected from the current 3-D pose. | MPJPE@2 −3 to −5 mm. | Medium | 3 |
| Hierarchical Temporal Pyramid v50 | Multi-resolution temporal pyramid replacing v47's single-scale transformer. | MPJPE@2 −2.5 to −3.5 mm; MPJPE@full −1.2 to −2.0 mm. | Medium | 3 |
| Adaptive Temporal Window v50 | Per-joint causal short/long temporal branch selection. | MPJPE@2 −2 to −3 mm; Jitter −10 %. | Medium | 3 |
| 3DPW Self-Evolving Domain Adapter v50 | Test-time self-supervised refinement block for 3DPW actual mode. | 3DPW actual MPJPE@3 −4 to −6 mm; MPJPE@2 −5 to −8 mm. | High | 2 |
| Neural Multi-View Stereo Fusion v50 | Epipolar cost-volume stereo branch fused with triangulation features. | MPJPE@2 −2 to −4 mm. | High | 2 |
| Efficient Attention for Long Clips v50 | Memory-efficient hierarchical temporal attention for longer clips. | MPJPE@2 (clip_len=25) −2 to −3 mm; memory <60 % of v47. | Medium | 3 |

## 3. Top-3 Detailed Comparison

### 1. Self-Evolution Feedback Head v50

- **Architecture**: A lightweight 2-layer MLP that consumes reprojection, temporal, and epipolar residuals and predicts updated per-view reliability and per-joint log-variance; a residual gate initialized near identity preserves the v46/v48 baseline at startup.
- **Flags / defaults**: `use_v50_self_evolution_feedback_head` (False), `v50_sefh_hidden` (64), `v50_sefh_num_layers` (2), `v50_sefh_dropout` (0.1), `v50_sefh_reproj_weight` (1.0), `v50_sefh_temporal_weight` (0.5), `v50_sefh_epipolar_weight` (0.5), `v50_sefh_max_refinement_steps` (1), `v50_sefh_identity_init_gate` (True), plus `loss.v50_sefh_loss_weight` (0.01).
- **Loss**: `L_sefh = v50_sefh_loss_weight * (L_reproj_nll + L_residual_smooth + L_reliability_entropy)`.
- **Eval metric**: `MPJPE@k` for k = 2,3,4,full; Spearman(reliability, residual) target > 0.3.
- **Risk / mitigation**: Risk of collapse to uniform reliability is mitigated by identity-at-init, clamping reliability to [0.05, 1.0], freezing base weights for the first epoch, and capping refinement steps at one.
- **Why selected**: It is the purest expression of the v50 self-evolution narrative, unifies the earlier v37/v39 reliability heads, and adds only a small MLP with minimal memory and latency overhead.

### 2. Cross-Domain Sparse-View Reliability Gap Closer

- **Architecture**: Two-layer cross-attention block where domain embeddings attend to view embeddings, producing a domain-conditioned per-view reliability offset and a per-joint log-variance uncertainty scale.
- **Flags / defaults**: `use_v50_cross_domain_sparse_view_reliability` (False), `v50_cdsvg_hidden` (64), `v50_cdsvg_num_heads` (4), `v50_cdsvg_loss_weight` (0.01), `v50_cdsvg_offset_min` (0.05), `v50_cdsvg_use_domain_label` (True), `v50_cdsvg_uncertainty_temperature` (1.0).
- **Loss**: `L_cdsvg = λ · [ (1/V) Σ_v w_v · Huber(||r_v||, δ) + (1/J) Σ_j exp(-σ_j) · e_j + γ · Var(w_v) ]`.
- **Eval metric**: `MPJPE@k` for k = 2,3,4,full; per-domain `MPJPE@k`; Spearman(reliability offset, residual) > 0.35; ECE-style uncertainty calibration.
- **Risk / mitigation**: Risk of confounding with v48/v46 is mitigated by treating it as an optional add-on, identity-at-init offset MLP, and a config check that raises when `use_v48_domain_generalization=False` but domain labels are requested.
- **Why selected**: It targets the largest remaining paper gap — out-of-domain sparse-view performance on 3DPW actual — and directly maps to the ICRA/CVPR 2027 claim of cross-domain sparse-view robustness.

### 3. Outlier View Rejection v2

- **Architecture**: Iterative soft per-(view, joint) keep-weight refinement that combines normalized reprojection residuals with the v37/v39 learned reliability score, followed by a weighted DLT re-triangulation and a sparse-aware guard.
- **Flags / defaults**: `use_v50_outlier_view_rejection_v2` (False), `v50_ovr_v2_num_iterations` (2), `v50_ovr_v2_soft_temperature` (0.1), `v50_ovr_v2_residual_threshold` (2.5), `v50_ovr_v2_min_keep_fraction` (0.5), `v50_ovr_v2_reliability_gate_threshold` (0.1), `v50_ovr_v2_loss_weight` (0.01).
- **Loss**: `L_outlier = - Σ_{v,j} [ w_{v,j} log ρ_{v,j} + (1 - w_{v,j}) log(1 - ρ_{v,j}) ]`, weighted by `v50_ovr_v2_loss_weight`.
- **Eval metric**: `MPJPE@k` for k = 2,3,4,full; Spearman(reliability, residual) target > 0.3.
- **Risk / mitigation**: Risk of over-aggressive rejection is mitigated by `min_keep_fraction` floor, identity-at-init weights near 1, and clamping `ρ` to [0.1, 0.95].
- **Why selected**: It builds on the proven v33 outlier work and the v37 reliability head, delivering the largest expected sparse-view gain at moderate risk.

## 4. Top-1 Recommendation: Self-Evolution Feedback Head v50

**Module name and file path**: `SelfEvolutionFeedbackHeadV50` → `motionflow_mv/fusion/self_evolution_feedback_head_v50.py`

**Exact config flags and defaults**:

| Flag | Type | Default |
|---|---|---|
| `use_v50_self_evolution_feedback_head` | bool | `False` |
| `v50_sefh_hidden` | int | `64` |
| `v50_sefh_num_layers` | int | `2` |
| `v50_sefh_dropout` | float | `0.1` |
| `v50_sefh_reproj_weight` | float | `1.0` |
| `v50_sefh_temporal_weight` | float | `0.5` |
| `v50_sefh_epipolar_weight` | float | `0.5` |
| `v50_sefh_max_refinement_steps` | int | `1` |
| `v50_sefh_identity_init_gate` | bool | `True` |
| `loss.v50_sefh_loss_weight` | float | `0.01` |

**Integration points**:

1. In `motionflow_mv/fusion/omniview_fusion_v5.py`:
   - Instantiate `SelfEvolutionFeedbackHeadV50(...)` after the v47/v48 pose output when `use_v50_self_evolution_feedback_head=True`.
   - Feed it the current 3-D pose estimate, 2-D keypoints, and camera parameters.
   - Multiply the returned per-view reliability into the next triangulation/aggregation step; add the per-joint log-variance as an uncertainty scale on the final supervised loss.
   - Add a config check that raises if `use_v37_self_critique_reliability=True` or `use_v39_reliability_coupled_refinement=True` while SEFH is enabled, to avoid redundant reliability heads.

2. In `experiments/train_omniview_fusion_v5_webbridge_multi.py`:
   - Add the auxiliary loss: `L_sefh = loss.v50_sefh_loss_weight * (L_reproj_nll + L_residual_smooth + L_reliability_entropy)`.
   - Freeze base model weights for the first epoch so only the SEFH head updates.
   - Log `Spearman(reliability, residual)` and per-joint log-variance calibration as diagnostics.

**Smoke test plan**:

- **Config file**: `configs/benchmark_v50_self_evolution_feedback_head_smoke.yaml`
- **Command**: `bash scripts/run_v50_self_evolution_feedback_head_smoke_local_4090.sh` (or the equivalent `python experiments/train_omniview_fusion_v5_webbridge_multi.py --config configs/benchmark_v50_self_evolution_feedback_head_smoke.yaml`).
- **Warm-start**: from the best available v46/v48 checkpoint.
- **Acceptance thresholds**:
  - `val_MPJPE@full` within 1 mm of the baseline.
  - `MPJPE@2` improves by ≥2 mm.
  - `Spearman(reliability, reprojection_residual)` > 0.3.
  - No NaN/OOM and wall-clock per-step overhead <15 %.

**A800 full-run plan**:

- **Base config**: v46-SVG or v48-domain full-run YAML.
- **train_samples**: 10000.
- **epochs**: 5, with early stopping if validation does not improve for 2 epochs.
- **Model size**: `d=128`, matching the current best v25/v42 A800 runs.
- **Flags enabled**: `use_v50_self_evolution_feedback_head=True`, `v50_sefh_hidden=64`, `v50_sefh_num_layers=2`, `loss.v50_sefh_loss_weight=0.01`.
- **Warm-start**: load the best v48-lite checkpoint; freeze base weights for epoch 0, then unfreeze.
- **Validation**: run `experiments/eval_variable_views.py` every epoch and report `MPJPE@k` for k = 2,3,4,full plus per-domain metrics.

**Main risk and mitigation**:

- **Risk**: The feedback head collapses to uniform reliability and provides no gain, or it destabilizes the already-strong v46/v48 baseline.
- **Mitigation**: Enforce identity-at-init via a zero-initialized residual gate; clamp reliability to [0.05, 1.0]; cap refinement to a single step; freeze the base model for the first epoch; start ablations with `loss.v50_sefh_loss_weight=0.001` before committing the default 0.01.

## 5. Paper-Story Paragraph

MotionFlow-MultiView v50 advances the self-evolution narrative by turning the pose estimator into its own critic. Instead of relying on a frozen triangulation step and a separately trained reliability head, v50 feeds reprojection, temporal, and epipolar residuals back into a lightweight Self-Evolution Feedback Head that refines per-view reliability and per-joint uncertainty in the same training graph. This closes the loop between prediction and geometric evidence, letting the model know which views to trust and how uncertain each joint is when only two or three views are available. By keeping the module identity-at-init and gradient-safe, v50 preserves the strong v46/v48 full-view baseline while pushing sparse-view accuracy and cross-domain robustness forward — a direct step toward a multi-view pose system that adapts to its own mistakes across views, time, and domains.
