# v49: Self-Evolving View Dropout for Sparse Generalization

**Status:** Proposal / ready for design review  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #167 (proposed)  
**Depends on:** #160 (v46-SVG), #162 (v47-temporal), #164 (v48-domain), v37 self-critique view reliability

---

## 1. Problem Statement

v46 Sparse-View Generalization (SVG) makes `OmniMultiViewFusionV5` robust to missing views by randomly dropping views during training and re-weighting survivors with a lightweight reliability head. In practice, the fixed random dropout has three weaknesses:

1. **Blind to sample difficulty.** A uniformly random mask may drop the only two reliable views on a hard sequence, or keep four redundant views on an easy one. The model never adapts its dropout to the current pose, motion, or camera geometry.
2. **Ignores self-critique.** v37 already predicts per-(view, joint) reliability from reprojection residuals, but v46 does not use these scores to decide which views to drop during training.
3. **Sparse-view tail is under-sampled.** The curriculum only ramps the global dropout rate; it does not actively mine the hardest 2–3 view subsets, so few-view inference remains noisy.

v49 closes these gaps with a **self-evolving view dropout policy** that uses the model's own reliability and reprojection feedback to choose which views to drop, and a hard-negative sparse-view miner that forces the model to train on the most informative view subsets.

---

## 2. Proposed Approach

v49 keeps the v46 reliability head and dropout augmentation, but replaces the fixed Bernoulli mask with a **learned, sample-conditional dropout policy**. The policy is small, identity-at-init, and trained with the main pose loss so no extra annotated data is required.

```text
Input: (B, T, V, J, C) tokens + cameras + v37 reliability r(B, T, V, J)
        |
        ▼
[ v49 Adaptive View Dropout Policy ]
        |
        ├── Geometry-aware features (epipolar residual, ray angles, baseline)
        ├── v37 reliability scores
        └── Per-view keep probability p_v ∈ (0, 1)
        |
        ▼
[ v49 Hard-Negative Sparse-View Miner ]
        |
        ├── Keep the smallest subset that still triangulates well
        ├── With probability α, force the hardest 2–3 view subset
        └── Return binary view_mask
        |
        ▼
[ v46 Sparse-View Generalization ]
        └── Triangulated pose P_t
        |
        ▼
[ v47 / v48 downstream heads ]
```

### 2.1 Adaptive dropout policy

A tiny MLP predicts a per-view keep probability from:

- **v37 self-critique reliability** (current estimate of each view's trustworthiness)
- **Geometric redundancy** (how much a view adds given the other active views)
- **Temporal stability** (reliability variance over the clip)

At initialization the policy outputs `p_v = 1 - v46_dropout_rate`, so training starts near the v46 baseline.

### 2.2 Hard-negative sparse-view miner

With probability `v49_hard_negative_prob` (default 0.25), the miner overrides the policy and selects the worst-case subset of `min_views` views according to the current v37 reliability. This guarantees exposure to the 2–3 view regime that matters at inference, without waiting for random dropout to produce it.

### 2.3 Fit with v46–v48

- **v46:** v49 is a drop-in replacement for `ViewDropoutAugmentationV46`. The same masks are produced; downstream modules are unchanged.
- **v47:** Temporal aggregation still receives per-frame triangulated poses; the temporal head is unaffected.
- **v48:** Domain generalization still sees the same mixed manifest. Per-domain dropout rates can be left in place or overridden by the adaptive policy.

---

## 3. Concrete Code-Level Changes

### New files

- `motionflow_mv/data/adaptive_view_dropout_v49.py`
  - `AdaptiveViewDropoutV49`: predicts per-view keep probabilities and returns a binary mask.
  - `HardNegativeSparseViewMiner`: forces worst-case `min_views` subsets.
  - Helper `geometry_redundancy_features(points_2d, K, R, t)` to compute view-geometry features.

### Modified files

- `motionflow_mv/data/view_dropout_augmentation_v46.py`
  - Add an optional `adaptive_policy: Optional[AdaptiveViewDropoutV49]` argument to `ViewDropoutAugmentationV46.__init__`.
  - When the policy is present, call it before falling back to Bernoulli dropout.

- `motionflow_mv/fusion/omniview_fusion_v5.py`
  - Add flags:
    - `use_v49_adaptive_view_dropout` (default `False`)
    - `v49_dropout_policy_hidden` (default `32`)
    - `v49_hard_negative_prob` (default `0.25`)
    - `v49_min_views` (default `2`)
    - `v49_adaptive_loss_weight` (default `0.01`)
  - Pass v37 reliability scores to the dropout policy when available.

- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
  - Expose CLI flags and instantiate `AdaptiveViewDropoutV49` when enabled.
  - Add a small auxiliary loss that encourages the policy to keep views with high v37 reliability:
    ```python
    L_policy = -v49_adaptive_loss_weight * mean(r * log(p) + (1 - r) * log(1 - p))
    ```
    where `r` is the v37 reliability and `p` is the predicted keep probability.

- `configs/benchmark_v49_view_dropout_sparse_generalization_smoke.yaml`
  - Smoke config: `d=64`, `batch_size=4`, `clip_len=9`, `train_samples=500`, 5 epochs.

- `scripts/run_v49_view_dropout_sparse_generalization_smoke_local_4090.sh`
  - Smoke launch script.

### New training flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_adaptive_view_dropout` | bool | `False` | Master switch for v49 adaptive view dropout. |
| `v49_dropout_policy_hidden` | int | `32` | Hidden dim of the dropout policy MLP. |
| `v49_hard_negative_prob` | float | `0.25` | Probability of forcing the worst-case `min_views` subset. |
| `v49_min_views` | int | `2` | Minimum surviving views after adaptive dropout. |
| `v49_adaptive_loss_weight` | float | `0.01` | Weight of the reliability-supervised policy loss. |

---

## 4. Risks / Failure Modes

| Risk | Failure mode | Mitigation |
|------|--------------|------------|
| Policy collapses to keeping all views | Sparse-view MPJPE@2 does not improve | Initialize policy near v46 Bernoulli dropout; clamp minimum keep probability to `1 - v46_dropout_rate` early in training. |
| Hard-negative mining is too aggressive | Training loss diverges on 2-view samples | Cap mining probability at 0.25; only mine when the current val MPJPE@2 is below a threshold. |
| v37 reliability is noisy early | Policy chases poor targets | Freeze the policy for the first epoch; use only the v37 score after it has been warm-started. |
| Adaptive dropout interacts badly with v48 per-domain dropout | Domain labels leak through dropout rate | Let the adaptive policy predict domain-agnostic keep probabilities; keep per-domain rate as a soft prior. |

---

## 5. Success Metrics and Experiments

### Smoke experiment (RTX 4090)

| Field | Value |
|-------|-------|
| Config | `configs/benchmark_v49_view_dropout_sparse_generalization_smoke.yaml` |
| Hardware | Local RTX 4090 (24 GiB) |
| Duration | ~1–2 hours |
| Goal | `val_MPJPE < 80 mm`, no NaN/OOM, `MPJPE@2` ≤ v46 smoke baseline |

Run:

```bash
bash scripts/run_v49_view_dropout_sparse_generalization_smoke_local_4090.sh
```

### Full experiment (A800-D)

| Field | Value |
|-------|-------|
| Config | `configs/benchmark_v49_view_dropout_sparse_generalization_full.yaml` |
| Hardware | A800-D |
| Duration | ~1–2 days |
| Goal | Match v46 full-view MPJPE; improve `MPJPE@2` by ≥10% over v46 |

### Evaluation

Extend `experiments/eval_variable_views.py` to report:

| Metric | Description |
|--------|-------------|
| `MPJPE@full` | Full-view accuracy; should not regress vs. v46. |
| `MPJPE@2`, `MPJPE@3`, `MPJPE@4` | Sparse-view accuracy; target ≥10% improvement at –3 views. |
| `mean_kept_views` | Average number of active views per sample during training. |
| `policy_entropy` | Should stay bounded; collapse indicates degenerate policy. |

---

## 6. Self-Evolution Feedback Loop

v49 closes a self-evolution loop between view selection and pose estimation:

1. **Forward pass:** The model predicts per-view reliability with the v37 self-critique head and a 3-D pose with v46.
2. **Reprojection feedback:** The 3-D pose is reprojected to each view; large residuals lower v37 reliability for that view.
3. **Policy update:** The v49 dropout policy uses the updated reliability to learn which views are informative.
4. **Sparse training:** The policy and hard-negative miner produce the next epoch's view masks, biasing training toward subsets the model currently finds hardest.
5. **Loop repeat:** The improved model produces better reprojection feedback, refining both v37 reliability and the v49 policy.

No extra labels or offline stages are required; the loop runs inside the standard training step.

---

## 7. Paper Story Fit

v49 supports the paper claim: *Our multi-view pose estimator actively learns to reason about sparse and unreliable views, using its own reprojection feedback to discover which camera subsets are most informative.* It turns the passive v46 dropout augmentation into an adaptive, self-improving component of the pipeline.

---

## 8. Next Steps

1. Wait for v46-SVG smoke results (#160) and v37 self-critique reliability to be stable.
2. Implement `AdaptiveViewDropoutV49` and the hard-negative miner.
3. Wire v49 flags into `ViewDropoutAugmentationV46` and the trainer.
4. Run smoke on RTX 4090 and compare `MPJPE@2` with the v46 smoke baseline.
5. If smoke meets targets, queue the full A800 run.
