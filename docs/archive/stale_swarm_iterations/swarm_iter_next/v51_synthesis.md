# v51 Design Space Synthesis

Synthesized from the 20 sub-agent notes in `docs/swarm_iter_next/v51_*.md`. The project is at the v50 Self-Evolution Feedback Head milestone: v46-SVG local smoke is `val_MPJPE@full ≈ 32.97 mm`, v47 temporal smoke is in progress, the A800 v46 full run is training, and v50 SEFH is queued.

## 1. Executive Summary

v51 should keep the proven v46/v47/v48/v50 backbone intact and attack the **cross-domain sparse-view reliability gap**: the v50 self-evolution loop works well on the training distribution, but its residual-to-reliability mapping can fail on out-of-domain data such as 3DPW actual, especially when only 2–3 views are available. The 20 proposals fall into four natural clusters:

1. **Reliability / triangulation refinement** (CDSVR, MLTR, NBAv2, OVR/UGVD, GAAP, VCC, SEFH-3DPW).
2. **Temporal / efficiency** (long-clip memory, lightweight temporal transformer, probabilistic forecaster).
3. **Domain / data** (domain-agnostic ensemble, real-world 3DPW refiner, cross-dataset normalization, DALR, self-supervised pretraining, sparse-view mixup).
4. **Calibration / robustness** (camera noise, skeleton-aware uncertainty, dynamic view selection, view consistency).

Most are warm-startable and identity-at-init, so they can be smoke-tested on the local RTX 4090 without waiting for A800 results. The highest-priority work is the module that explicitly makes v50 reliability **domain-aware** for sparse views, because it directly fills the largest remaining paper gap with the smallest architectural change.

## 2. Ranking Table of the 20 v51 Proposals

Ranked by expected sparse-view/cross-domain impact per unit risk.

| Rank | Proposal (file) | Key idea | Expected MPJPE impact | Risk |
|---:|---|---|---|---|
| 1 | **Cross-Domain Sparse-View Reliability** (`v51_paper_gap_analysis_v51.md`) | Condition v50 reliability/uncertainty on a domain embedding via a 2-layer cross-attention block. | 3DPW actual `MPJPE@2` −5 to −7 mm; in-domain full-view ±0.5 mm. | Medium |
| 2 | **Real-World 3DPW Actual Refiner** (`v51_real_world_3dpw_actual.md`) | Test-time geometric self-refinement (reprojection + bone + temporal) for 3DPW actual. | 3DPW actual `MPJPE@2` −4 to −7 mm; `MPJPE@full` −2 to −3 mm. | Medium |
| 3 | **Reliability-Guided Sparse-View Mixup** (`v51_sparse_view_data_augmentation.md`) | Use v50 reliability to weight pose mixup and synthesize harder sparse-view training samples. | `MPJPE@2` −3 to −5 mm; 3DPW actual −4 to −6 mm. | Medium |
| 4 | **Camera Noise Robustness v2** (`v51_camera_noise_robustness_v2.md`) | Synthetic calibration-noise augmentation + bounded per-camera extrinsic correction head. | `MPJPE@2` under noise −3 to −6 mm; full-view ±0.5 mm. | Medium |
| 5 | **Domain-Agnostic Ensemble** (`v51_domain_agnostic_ensemble.md`) | Learnable ensemble over v25/v46/v47/v48/v50 experts gated by geometric evidence. | `MPJPE@2` −3 to −5 mm; 3DPW actual −4 to −6 mm. | Medium-High |
| 6 | **Test-Time Self-Evolution Refiner** (`v51_test_time_adaptation.md`) | Per-sequence Adam refinement of v50 reliability/uncertainty at inference. | 3DPW actual `MPJPE@2` −3 to −5 mm; in-domain −1 to −2 mm. | Medium-High |
| 7 | **Model-Level Triangulation Refinement** (`v51_model_level_triangulation_refinement.md`) | Learned differentiable triangulation refiner with per-view per-joint precision weights. | `MPJPE@2` −3 to −5 mm; `MPJPE@3` −2 to −4 mm. | Medium |
| 8 | **Self-Supervised Multi-View Pretraining** (`v51_self_supervised_pretraining.md`) | Pretrain on unlabeled multi-view video via 2-D reconstruction + temporal + epipolar heads. | `MPJPE@2` −3 to −5 mm; 3DPW actual −4 to −7 mm. | High |
| 9 | **Cross-Dataset Pose Normalization** (`v51_cross_dataset_pose_normalization.md`) | Per-dataset scale/translation/bone-length canonicalization before the pose head. | 3DPW actual `MPJPE@2` −3 to −5 mm; full-view −0.8 to −1.5 mm. | Medium |
| 10 | **Neural Bundle Adjustment v2** (`v51_neural_bundle_adjustment_v2.md`) | Lightweight differentiable BA step after v50 using uncertainty-weighted reprojection residuals. | `MPJPE@2` −2 to −4 mm; 3DPW actual −3 to −5 mm. | Medium |
| 11 | **Multi-View Diffusion Refiner** (`v51_multi_view_diffusion_refiner.md`) | Few-step diffusion denoiser on the fused 3-D pose, conditioned on v50 reliability. | `MPJPE@2` −2 to −4 mm; 3DPW actual −3 to −5 mm. | Medium-High |
| 12 | **Uncertainty-Guided View Dropout** (`v51_uncertainty_guided_view_dropout.md`) | Replace v46 uniform dropout with a reliability-conditioned dropout policy. | `MPJPE@2` −2 to −4 mm; `MPJPE@3` −1 to −2 mm. | Medium |
| 13 | **View Consistency Constraint** (`v51_view_consistency_constraint.md`) | Learned geometric-consistency weights over view-pair residuals. | `MPJPE@2` −2 to −4 mm; 3DPW actual up to −5 mm. | Medium |
| 14 | **Geometry-Aware Attention Pooling** (`v51_geometry_aware_attention_pooling.md`) | Replace mean/max pooling with geometry-biased attention over views. | `MPJPE@2` −2 to −4 mm; `MPJPE@3` −1 to −2 mm. | Medium |
| 15 | **Skeleton-Aware Uncertainty Gating** (`v51_skeleton_aware_uncertainty.md`) | GNN over kinematic graph to refine v50 per-joint uncertainty. | `MPJPE@2` −2 to −4 mm; `MPJPE@3` −1 to −2 mm. | Medium |
| 16 | **Dynamic View Selection Policy** (`v51_dynamic_view_selection_policy.md`) | Gumbel-softmax policy that selects camera subsets for triangulation. | `MPJPE@2` −2 to −4 mm; `MPJPE@full` ±0.3 mm. | Medium-High |
| 17 | **Long-Clip Efficient Temporal Memory** (`v51_long_clip_efficiency.md`) | Perceiver-style local compressor + causal memory bank for long clips. | `MPJPE@full` (clip_len=25) −1.5 to −3.0 mm; `MPJPE@2/3` −2 to −3 mm. | Medium |
| 18 | **Lightweight Temporal Transformer** (`v51_lightweight_temporal_transformer.md`) | Causal per-joint sliding-window attention + cross-joint MLP. | `MPJPE@2/3/4` −1.5 to −2.5 mm; full-view −0.8 to −1.5 mm. | Low-Medium |
| 19 | **Adaptive Learning Rate Per Domain** (`v51_adaptive_learning_rate_per_domain.md`) | Domain-conditioned optimizer LR scaler based on gradient norms. | 3DPW actual `MPJPE@2` −2 to −4 mm. | Low-Medium |
| 20 | **Probabilistic Pose Forecaster** (`v51_probabilistic_pose_forecasting.md`) | Causal future-pose distribution used as a temporal prior. | `MPJPE@2` −1 to −2 mm; full-view ±0.5 mm. | Medium |

## 3. Top-1 Recommendation: Cross-Domain Sparse-View Reliability (CDSVR)

**Module:** `CrossDomainSparseViewReliabilityV51` → `motionflow_mv/fusion/cross_domain_sparse_view_reliability_v51.py`

**Why:** It is the smallest architectural extension of v50 SEFH that directly targets the largest remaining paper gap — out-of-domain sparse-view reliability on 3DPW actual. It is identity-at-init, warm-startable, and only adds a 2-layer cross-attention block.

### New config flags and defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_cross_domain_sparse_view_reliability` | bool | `False` |
| `v51_cdsvr_hidden` | int | `64` |
| `v51_cdsvr_num_heads` | int | `4` |
| `v51_cdsvr_dropout` | float | `0.1` |
| `v51_cdsvr_offset_min` | float | `0.05` |
| `v51_cdsvr_use_domain_label` | bool | `True` |
| `v51_cdsvr_uncertainty_temperature` | float | `1.0` |
| `v51_cdsvr_identity_init_gate` | bool | `True` |
| `loss.v51_cdsvr_loss_weight` | float | `0.01` |

### Loss term

```
L_cdsvr = λ · [ (1/V) Σ_v w'_v · Huber(||ε_v||, δ)
              − (1/J) Σ_j log α_j
              + γ · Var(w') ]
```

where `w'_v = sigmoid(r'_v / τ)`, `r'_v` is the domain-conditioned reliability, `α_j` is the per-joint uncertainty rescale, and `λ`, `δ`, `γ` are hyperparameters.

### Evaluation metrics

- `MPJPE@k` for `k = 2, 3, 4, full` via `experiments/eval_variable_views.py`.
- Per-domain `MPJPE@k` (H36M / MPI / 3DPW actual).
- `Spearman(r'_v, ||ε_v||) > 0.35` and ECE-style calibration of `exp(−σ')`.

### Implementation plan

1. **Create module** `motionflow_mv/fusion/cross_domain_sparse_view_reliability_v51.py` implementing the 2-layer cross-attention block.
2. **Wire into `motionflow_mv/fusion/omniview_fusion_v5.py`**:
   - Call `CrossDomainSparseViewReliabilityV51` after `SelfEvolutionFeedbackHeadV50` when `use_v51_cross_domain_sparse_view_reliability=True`.
   - Consume v48 domain embedding when available; fall back to one-hot domain label.
   - Apply residual reliability offset and multiplicative uncertainty rescale.
   - Raise a config error if `use_v50_self_evolution_feedback_head=False`.
3. **Trainer changes** (`experiments/train_omniview_fusion_v5_webbridge_multi.py`):
   - Add `L_cdsvr` weighted by `loss.v51_cdsvr_loss_weight`.
   - Freeze base model weights for the first epoch; train only the CDSVR head.
   - Log per-domain `MPJPE@k` and reliability-offset diagnostics.
4. **Smoke test**:
   - Config: `configs/benchmark_v51_cdsvr_smoke.yaml` (copy from v50 SEFH smoke).
   - Warm-start from the best available v50 SEFH checkpoint.
   - Acceptance: `val_MPJPE@full` within 1 mm of v50 baseline, 3DPW actual `MPJPE@2` improves ≥1 mm, no NaN/OOM.
5. **A800 full-run plan**:
   - Add entry to `scripts/launch_v33_a800_queue.py` after v50 SEFH smoke.
   - Settings: `d=128`, `train_samples=10000`, 5 epochs, early stopping after  epochs without improvement.
   - Flags: `use_v51_cross_domain_sparse_view_reliability=True`, `v51_cdsvr_hidden=64`, `loss.v51_cdsvr_loss_weight=0.01`.
   - Run `experiments/eval_variable_views.py` every epoch and report `MPJPE@k` and per-domain metrics.

### Main risk and mitigation

- **Risk:** The module conflates domain shift with view dropout and collapses to a single reliability mode on small domains.
- **Mitigation:** Identity-at-init (`Δr_v = 0`, `α_j = 1`), clamp `Δr_v` to `[-2, 2]`, freeze the base for the first epoch, and require `use_v50_self_evolution_feedback_head=True`.

## 4. Paper-Story Paragraph

MotionFlow-MultiView v51 turns the v50 self-evolution loop into a **domain-aware critic**. Where v50 learned to trust each view from reprojection, temporal, and epipolar residuals, v51 recognizes that the residual-to-reliability mapping itself changes across domains: a reprojection residual that signals a bad view in H36M may mean something different in the wilds of 3DPW actual. The Cross-Domain Sparse-View Reliability module adds a lightweight, identity-at-init cross-attention block that modulates v50's per-view reliability and per-joint uncertainty with a domain embedding, so the model knows not only *which views to trust* but also *how to interpret that trust* when cameras, subjects, and backgrounds change. Because it leaves the proven v46/v47/v48/v50 backbone untouched at startup, it preserves the strong full-view baseline while closing the 2–3 view cross-domain gap — the next concrete step toward a multi-view pose system that adapts to its own evidence across views, time, and domains.
